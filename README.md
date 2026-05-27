# rustvani-vad

Pure Rust Silero VAD — zero ONNX runtime, bundled weights, SIMD-accelerated.

~90% less memory than Python Pipecat's Silero VAD. Compiles to a static binary with no external dependencies.

## Features

- **Bundled weights** — `silero_vad_16k.bin` is compiled into the binary via `include_bytes!`. No downloads, no file paths, no cache directories.
- **SIMD** — AVX2+FMA dot products on x86_64, scalar fallback on other architectures.
- **State machine** — Full Pipecat-compatible `Quiet → Starting → Speaking → Stopping → Quiet` lifecycle with volume gating.
- **Optional async** — `tokio` feature enables `infer_async()` via `spawn_blocking`.

## Usage

```toml
[dependencies]
rustvani-vad = "0.1"

# For async support:
# rustvani-vad = { version = "0.1", features = ["tokio"] }
```

### Sync

```rust
use rustvani_vad::{SileroVad, StateMachine, VadParams, VadState};

let vad = SileroVad::new().unwrap();
let mut sm = StateMachine::new(16000, VadParams::default());

// In your audio read loop:
// `chunk` is a slice of i16 LE mono PCM at 16 kHz
if let Some(window) = sm.next_window(&chunk) {
    let confidence = vad.infer(&window).unwrap();
    match sm.advance(confidence, &window) {
        VadState::Speaking => println!("speech detected"),
        VadState::Quiet    => println!("silence"),
        _                  => {}
    }
}
```

### Async (with `tokio` feature)

```rust
let confidence = vad.infer_async(window).await.unwrap();
```

### Custom parameters

```rust
let params = VadParams {
    confidence: 0.6,   // stricter threshold
    min_volume: 0.4,   // lower volume gate
    start_secs: 0.3,   // slower speech start confirmation
    stop_secs: 1.2,    // longer silence before stop
};
let mut sm = StateMachine::new(16000, params);
```

## Building

Place `silero_vad_16k.bin` in the `models/` directory before building:

```
models/silero_vad_16k.bin
```

This is the flat f32 LE binary extracted from the Silero ONNX model (≥309,633 floats, ~1.2 MB).

## Integrating with Rustvani

In Rustvani proper, wrap `SileroVad` behind the `VadAnalyzer` trait:

```rust
use rustvani_vad::{SileroVad, BYTES_PER_WINDOW, SAMPLES_PER_WINDOW};

#[async_trait]
impl VadAnalyzer for SileroVad {
    fn num_frames_required(&self) -> usize {
        SAMPLES_PER_WINDOW
    }

    async fn voice_confidence(&self, audio: Vec<u8>) -> f32 {
        self.infer_async(audio).await.unwrap_or(0.0)
    }
}
```

## License

MIT OR Apache-2.0
