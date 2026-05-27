//! Tunable VAD parameters.

/// Parameters controlling VAD sensitivity and timing.
///
/// All values have sensible defaults for conversational speech at 16 kHz.
#[derive(Debug, Clone)]
pub struct VadParams {
    /// Confidence threshold (0.0–1.0) above which a frame is considered speech.
    pub confidence: f32,

    /// Minimum volume (0.0–1.0) required alongside confidence to count as speech.
    /// Filters out model false-positives on near-silent audio.
    pub min_volume: f32,

    /// Seconds of consecutive speech needed to transition from Starting → Speaking.
    pub start_secs: f32,

    /// Seconds of consecutive silence needed to transition from Stopping → Quiet.
    pub stop_secs: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            confidence: 0.5,
            min_volume: 0.6,
            start_secs: 0.2,
            stop_secs: 0.8,
        }
    }
}
