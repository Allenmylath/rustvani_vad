//! VAD state machine.
//!
//! Drives the Quiet → Starting → Speaking → Stopping → Quiet lifecycle
//! based on confidence scores from the inference engine.
//!
//! # Volume calculation
//!
//! Approximates Python Pipecat's EBU R128 loudness with dBFS:
//!
//!   rms_linear = RMS(samples) / 32768          → 0.0–1.0
//!   db = 20 × log10(rms_linear).clamp(-60, 0)  → -60..0 dBFS
//!   volume = (db + 60) / 60                    → 0.0..1.0
//!
//! Normal conversational speech maps to ~0.3–0.6, compatible with the
//! default `min_volume = 0.6`.

use crate::engine::BYTES_PER_WINDOW;
use crate::params::VadParams;

// ─── VadState ─────────────────────────────────────────────────────────

/// Voice Activity Detection states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No voice activity.
    Quiet,
    /// Speech beginning — accumulating confirmation frames.
    Starting,
    /// Active voice confirmed.
    Speaking,
    /// Speech ending — accumulating silence frames.
    Stopping,
}

impl Default for VadState {
    fn default() -> Self {
        Self::Quiet
    }
}

// ─── Audio helpers ────────────────────────────────────────────────────

/// Calculate audio volume as a normalised 0.0–1.0 value (dBFS-based).
pub fn calculate_audio_volume(audio: &[u8]) -> f32 {
    let samples: Vec<i16> = audio
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32 / 32768.0;

    if rms < 1e-9 {
        return 0.0;
    }

    let db = (20.0 * rms.log10()).clamp(-60.0, 0.0);
    (db + 60.0) / 60.0
}

/// Exponential smoothing: `current × factor + prev × (1 - factor)`.
#[inline]
pub fn exp_smoothing(current: f32, prev: f32, factor: f32) -> f32 {
    current * factor + prev * (1.0 - factor)
}

// ─── StateMachine ─────────────────────────────────────────────────────

const SMOOTHING_FACTOR: f32 = 0.2;

/// Stateful VAD state machine.
///
/// One instance per audio stream. Buffers incoming audio, tracks frame
/// counts, smooths volume, and drives state transitions.
pub struct StateMachine {
    params: VadParams,
    bytes_required: usize,
    start_frames: usize,
    stop_frames: usize,
    starting_count: usize,
    stopping_count: usize,
    prev_volume: f32,
    buffer: Vec<u8>,
    /// Current VAD state.
    pub state: VadState,
}

impl StateMachine {
    /// Create a new state machine.
    ///
    /// `sample_rate` must be 16000 (the only rate supported by the native engine).
    pub fn new(sample_rate: u32, params: VadParams) -> Self {
        let frames_required: usize = if sample_rate == 16000 { 512 } else { 256 };
        let bytes_required = frames_required * 2;

        let frames_per_sec = frames_required as f32 / sample_rate as f32;
        let start_frames = (params.start_secs / frames_per_sec).round() as usize;
        let stop_frames = (params.stop_secs / frames_per_sec).round() as usize;

        Self {
            params,
            bytes_required,
            start_frames,
            stop_frames,
            starting_count: 0,
            stopping_count: 0,
            prev_volume: 0.0,
            buffer: Vec::with_capacity(bytes_required * 2),
            state: VadState::Quiet,
        }
    }

    /// Feed a PCM chunk into the buffer.
    ///
    /// Returns `Some(window)` when a full inference window is ready.
    pub fn next_window(&mut self, chunk: &[u8]) -> Option<Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        if self.buffer.len() >= self.bytes_required {
            let window: Vec<u8> = self.buffer.drain(..self.bytes_required).collect();
            Some(window)
        } else {
            None
        }
    }

    /// Advance the state machine with a confidence value.
    ///
    /// Mirrors Python Pipecat's `_run_analyzer()` transitions exactly.
    pub fn advance(&mut self, confidence: f32, audio_window: &[u8]) -> VadState {
        let volume = exp_smoothing(
            calculate_audio_volume(audio_window),
            self.prev_volume,
            SMOOTHING_FACTOR,
        );
        self.prev_volume = volume;

        let speaking =
            confidence >= self.params.confidence && volume >= self.params.min_volume;

        // State transitions
        if speaking {
            match self.state {
                VadState::Quiet => {
                    self.state = VadState::Starting;
                    self.starting_count = 1;
                }
                VadState::Starting => {
                    self.starting_count += 1;
                }
                VadState::Stopping => {
                    self.state = VadState::Speaking;
                    self.stopping_count = 0;
                }
                VadState::Speaking => {}
            }
        } else {
            match self.state {
                VadState::Starting => {
                    self.state = VadState::Quiet;
                    self.starting_count = 0;
                }
                VadState::Speaking => {
                    self.state = VadState::Stopping;
                    self.stopping_count = 1;
                }
                VadState::Stopping => {
                    self.stopping_count += 1;
                }
                VadState::Quiet => {}
            }
        }

        // Threshold checks
        if self.state == VadState::Starting && self.starting_count >= self.start_frames {
            self.state = VadState::Speaking;
            self.starting_count = 0;
        }

        if self.state == VadState::Stopping && self.stopping_count >= self.stop_frames {
            self.state = VadState::Quiet;
            self.stopping_count = 0;
        }

        self.state
    }

    /// Reset the state machine to initial conditions.
    pub fn reset(&mut self) {
        self.state = VadState::Quiet;
        self.starting_count = 0;
        self.stopping_count = 0;
        self.prev_volume = 0.0;
        self.buffer.clear();
    }
}
