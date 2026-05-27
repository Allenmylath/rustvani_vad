//! Pure Rust Silero VAD inference engine.
//!
//! Loads weights from a flat binary format (extracted from the ONNX model).
//! Architecture: STFT → 4×Conv1d encoder → LSTM → linear decoder → sigmoid.

use crate::simd::{dot, dot_relu};

// ─── Constants ────────────────────────────────────────────────────────

const H: usize = 128;
const GATES: usize = 4 * H;
const CONTEXT_16K: usize = 64;
const COMBINED: usize = 2 * H;

/// Number of PCM samples per inference window at 16 kHz.
pub const SAMPLES_PER_WINDOW: usize = 512;

/// Number of bytes per inference window (512 samples × 2 bytes/sample).
pub const BYTES_PER_WINDOW: usize = SAMPLES_PER_WINDOW * 2;

// ─── Recurrent state ─────────────────────────────────────────────────

pub(crate) struct InferState {
    h: Vec<f32>,
    c: Vec<f32>,
    context: Vec<f32>,
}

impl InferState {
    pub fn new() -> Self {
        Self {
            h: vec![0.0; H],
            c: vec![0.0; H],
            context: vec![0.0; CONTEXT_16K],
        }
    }

    /// Reset LSTM and context state to zeros.
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.context.fill(0.0);
    }
}

// ─── Conv1d weight block ──────────────────────────────────────────────

struct Conv1dW {
    weight: Vec<f32>,
    bias: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
}

// ─── Engine ───────────────────────────────────────────────────────────

pub(crate) struct Engine {
    stft_basis: Vec<f32>,
    enc: [Conv1dW; 4],
    lstm_w: Vec<f32>,
    lstm_bias: Vec<f32>,
    dec_weight: Vec<f32>,
    dec_bias: f32,
}

impl Engine {
    /// Parse weights from a flat little-endian f32 binary blob.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        if floats.len() < 309_633 {
            return Err(format!(
                "Weight file too small: {} floats, expected >= 309633",
                floats.len()
            ));
        }

        let mut o = 0usize;
        let mut take = |n: usize| -> Vec<f32> {
            let v = floats[o..o + n].to_vec();
            o += n;
            v
        };

        let stft_basis = take(258 * 256);
        let e0w = take(128 * 129 * 3);
        let e0b = take(128);
        let e1w = take(64 * 128 * 3);
        let e1b = take(64);
        let e2w = take(64 * 64 * 3);
        let e2b = take(64);
        let e3w = take(128 * 64 * 3);
        let e3b = take(128);
        let wih = take(512 * 128);
        let whh = take(512 * 128);
        let bih = take(512);
        let bhh = take(512);
        let dw = take(128);
        let db = floats[o];

        let mut bias = vec![0.0f32; GATES];
        for i in 0..GATES {
            bias[i] = bih[i] + bhh[i];
        }

        let mut lstm_w = vec![0.0f32; GATES * COMBINED];
        for g in 0..GATES {
            let dst = g * COMBINED;
            lstm_w[dst..dst + H].copy_from_slice(&wih[g * H..(g + 1) * H]);
            lstm_w[dst + H..dst + COMBINED].copy_from_slice(&whh[g * H..(g + 1) * H]);
        }

        Ok(Self {
            stft_basis,
            enc: [
                Conv1dW { weight: e0w, bias: e0b, out_ch: 128, in_ch: 129 },
                Conv1dW { weight: e1w, bias: e1b, out_ch: 64, in_ch: 128 },
                Conv1dW { weight: e2w, bias: e2b, out_ch: 64, in_ch: 64 },
                Conv1dW { weight: e3w, bias: e3b, out_ch: 128, in_ch: 64 },
            ],
            lstm_w,
            lstm_bias: bias,
            dec_weight: dw,
            dec_bias: db,
        })
    }

    /// Run inference on exactly 512 f32 samples. Updates state in-place.
    pub fn infer(&self, samples: &[f32], st: &mut InferState) -> f32 {
        debug_assert_eq!(samples.len(), SAMPLES_PER_WINDOW);

        // Prepend context
        let mut input = Vec::with_capacity(CONTEXT_16K + SAMPLES_PER_WINDOW);
        input.extend_from_slice(&st.context);
        input.extend_from_slice(samples);
        st.context
            .copy_from_slice(&input[input.len() - CONTEXT_16K..]);

        // STFT
        let padded = reflect_pad_right(&input, 64);
        let stft_len = (padded.len() - 256) / 128 + 1;
        let mag = self.stft_magnitude(&padded, stft_len);

        // Encoder
        let strides = [1usize, 2, 2, 1];
        let mut x = mag;
        let mut ch = 129usize;
        let mut len = stft_len;
        for (i, e) in self.enc.iter().enumerate() {
            let new_len = (len + 2 - 3) / strides[i] + 1;
            x = conv1d_k3_pad1_relu(&x, ch, len, e, strides[i], new_len);
            ch = e.out_ch;
            len = new_len;
        }

        // Decoder
        let mut prob_sum = 0.0f32;
        for t in 0..len {
            let mut frame = [0.0f32; H];
            for c in 0..ch {
                frame[c] = x[c * len + t];
            }
            self.lstm_cell(&frame, &mut st.h, &mut st.c);
            let logit = self.dec_bias + dot_relu(&self.dec_weight, &st.h);
            prob_sum += sigmoid(logit);
        }
        prob_sum / len as f32
    }

    fn stft_magnitude(&self, padded: &[f32], out_len: usize) -> Vec<f32> {
        let mut mag = vec![0.0f32; 129 * out_len];
        for t in 0..out_len {
            let x_off = t * 128;
            let x_slice = &padded[x_off..x_off + 256];
            for f in 0..129 {
                let re = dot(&self.stft_basis[f * 256..(f + 1) * 256], x_slice);
                let im =
                    dot(&self.stft_basis[(f + 129) * 256..(f + 130) * 256], x_slice);
                mag[f * out_len + t] = re.mul_add(re, im * im).sqrt();
            }
        }
        mag
    }

    fn lstm_cell(&self, input: &[f32], h: &mut Vec<f32>, c: &mut Vec<f32>) {
        let mut xh = [0.0f32; COMBINED];
        xh[..H].copy_from_slice(input);
        xh[H..].copy_from_slice(h);

        let mut gates = [0.0f32; GATES];
        let mut g = 0;
        while g + 4 <= GATES {
            let r0 = g * COMBINED;
            let r1 = (g + 1) * COMBINED;
            let r2 = (g + 2) * COMBINED;
            let r3 = (g + 3) * COMBINED;
            gates[g] =
                dot(&self.lstm_w[r0..r0 + COMBINED], &xh) + self.lstm_bias[g];
            gates[g + 1] =
                dot(&self.lstm_w[r1..r1 + COMBINED], &xh) + self.lstm_bias[g + 1];
            gates[g + 2] =
                dot(&self.lstm_w[r2..r2 + COMBINED], &xh) + self.lstm_bias[g + 2];
            gates[g + 3] =
                dot(&self.lstm_w[r3..r3 + COMBINED], &xh) + self.lstm_bias[g + 3];
            g += 4;
        }

        for i in 0..H {
            let ig = sigmoid(gates[i]);
            let fg = sigmoid(gates[H + i]);
            let gg = gates[2 * H + i].tanh();
            let og = sigmoid(gates[3 * H + i]);
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * c[i].tanh();
        }
    }
}

// ─── Small ops ────────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn reflect_pad_right(input: &[f32], pad: usize) -> Vec<f32> {
    let n = input.len();
    let mut out = Vec::with_capacity(n + pad);
    out.extend_from_slice(input);
    for i in 0..pad {
        out.push(input[n - 2 - i]);
    }
    out
}

fn conv1d_k3_pad1_relu(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    e: &Conv1dW,
    stride: usize,
    ol: usize,
) -> Vec<f32> {
    let pl = in_len + 2;
    let mut padded = vec![0.0f32; in_ch * pl];
    for ci in 0..in_ch {
        let src = ci * in_len;
        let dst = ci * pl + 1;
        padded[dst..dst + in_len].copy_from_slice(&input[src..src + in_len]);
    }
    let out_ch = e.out_ch;
    let mut output = vec![0.0f32; out_ch * ol];
    for co in 0..out_ch {
        let b = e.bias[co];
        for t in 0..ol {
            let ps = t * stride;
            let mut sum = b;
            for ci in 0..in_ch {
                let wb = (co * in_ch + ci) * 3;
                let xb = ci * pl + ps;
                unsafe {
                    sum += *e.weight.get_unchecked(wb) * *padded.get_unchecked(xb);
                    sum +=
                        *e.weight.get_unchecked(wb + 1) * *padded.get_unchecked(xb + 1);
                    sum +=
                        *e.weight.get_unchecked(wb + 2) * *padded.get_unchecked(xb + 2);
                }
            }
            output[co * ol + t] = sum.max(0.0);
        }
    }
    output
}
