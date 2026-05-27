//! # rustvani-vad
//!
//! Pure Rust Silero VAD with bundled weights — zero external runtime dependencies.
//!
//! SIMD-accelerated (AVX2+FMA on x86_64), ~90% less memory than Python Pipecat.
//!
//! ## Quick start
//!
//! ```ignore
//! use rustvani_vad::{SileroVad, StateMachine, VadParams, VadState};
//!
//! // Inference engine (weights are compiled in — nothing to download)
//! let vad = SileroVad::new()?;
//!
//! // State machine for stream processing
//! let mut sm = StateMachine::new(16000, VadParams::default());
//!
//! // Feed audio chunks in a loop
//! for chunk in audio_chunks {
//!     if let Some(window) = sm.next_window(&chunk) {
//!         let confidence = vad.infer(&window)?;
//!         match sm.advance(confidence, &window) {
//!             VadState::Speaking => { /* user is talking */ }
//!             VadState::Quiet    => { /* silence */ }
//!             _ => {}
//!         }
//!     }
//! }
//! ```
//!
//! ## Async (requires `tokio` feature)
//!
//! ```ignore
//! let confidence = vad.infer_async(window).await?;
//! ```

mod engine;
mod params;
mod simd;
mod state;

pub use engine::{BYTES_PER_WINDOW, SAMPLES_PER_WINDOW};
pub use params::VadParams;
pub use state::{calculate_audio_volume, exp_smoothing, StateMachine, VadState};

use std::sync::{Arc, Mutex};

use engine::{Engine, InferState};

// ─── Bundled weights ──────────────────────────────────────────────────

/// Silero VAD v5 weights for 16 kHz, embedded at compile time.
///
/// Place `silero_vad_16k.bin` in the `models/` directory before building.
static WEIGHTS: &[u8] = include_bytes!("../models/silero_vad_16k.bin");

// ─── Public API ───────────────────────────────────────────────────────

struct Inner {
    engine: Engine,
    state: InferState,
}

/// Pure Rust Silero VAD — no ONNX runtime, bundled weights.
///
/// Thread-safe (`Clone` shares the engine via `Arc<Mutex<_>>`).
/// Supports 16 kHz mono PCM only.
#[derive(Clone)]
pub struct SileroVad {
    inner: Arc<Mutex<Inner>>,
}

impl SileroVad {
    /// Create a new VAD instance with the bundled weights.
    pub fn new() -> Result<Self, String> {
        Self::from_bytes(WEIGHTS)
    }

    /// Create from custom weight bytes (e.g. a different model version).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let engine = Engine::from_bytes(data)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                engine,
                state: InferState::new(),
            })),
        })
    }

    /// Run inference on i16 LE mono PCM bytes.
    ///
    /// Input must be exactly [`BYTES_PER_WINDOW`] bytes (1024 bytes = 512 samples).
    /// Returns confidence in the range 0.0–1.0.
    pub fn infer(&self, audio_bytes: &[u8]) -> Result<f32, String> {
        let mut guard = self.inner.lock().unwrap();
        if audio_bytes.len() != BYTES_PER_WINDOW {
            return Err(format!(
                "Expected {} bytes, got {}",
                BYTES_PER_WINDOW,
                audio_bytes.len()
            ));
        }

        let samples: Vec<f32> = audio_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        let inner = &mut *guard;
        Ok(inner.engine.infer(&samples, &mut inner.state))
    }

    /// Async inference — offloads to a blocking thread.
    ///
    /// Requires the `tokio` feature.
    #[cfg(feature = "tokio")]
    pub async fn infer_async(&self, audio_bytes: Vec<u8>) -> Result<f32, String> {
        let inner = self.inner.clone();
        ::tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock().unwrap();
            if audio_bytes.len() != BYTES_PER_WINDOW {
                return Err(format!(
                    "Expected {} bytes, got {}",
                    BYTES_PER_WINDOW,
                    audio_bytes.len()
                ));
            }
            let samples: Vec<f32> = audio_bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            let inner = &mut *guard;
            Ok(inner.engine.infer(&samples, &mut inner.state))
        })
        .await
        .map_err(|e| format!("spawn_blocking error: {}", e))?
    }

    /// Reset the LSTM hidden state. Call between separate audio streams
    /// to avoid state bleed.
    pub fn reset_state(&self) {
        self.inner.lock().unwrap().state.reset();
    }

    /// Number of PCM samples per inference window (512 at 16 kHz).
    pub fn num_frames_required(&self) -> usize {
        SAMPLES_PER_WINDOW
    }
}
