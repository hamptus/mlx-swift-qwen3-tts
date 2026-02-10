# mlx-swift-qwen3-tts

A Swift Package for running [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) text-to-speech models on Apple Silicon using [MLX](https://github.com/ml-explore/mlx-swift).

## Features

- High-quality text-to-speech with multiple built-in voices
- Streaming audio generation for low-latency playback
- Voice cloning via speaker embeddings or ICL (in-context learning) reference audio
- VoiceDesign: generate voices from text descriptions (1.7B only)
- CustomVoice: named speakers with style/emotion control (1.7B only)
- Memory-efficient long text generation with automatic chunking
- Pre-quantized and runtime quantization support (4-bit/6-bit)
- macOS 14+ and iOS 17+ (Apple Silicon required)

## Requirements

- Apple Silicon Mac or iOS device
- macOS 14+ or iOS 17+
- Swift 5.9+
- A Qwen3-TTS model — 0.6B or 1.7B (see [Model Download](#model-download))

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/hamptus/mlx-swift-qwen3-tts", from: "0.2.0"),
]
```

Then add `Qwen3TTS` to your target dependencies:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "Qwen3TTS", package: "mlx-swift-qwen3-tts"),
    ]
),
```

## Quick Start

```swift
import Qwen3TTS

// Load model
let pipeline = try Qwen3TTSPipeline(modelPath: modelURL)

// Generate speech with a built-in voice
let samples = pipeline.generate(text: "Hello world!", speaker: "Aiden")

// Save to WAV file
try AudioSampleWriter.write(samples: samples, to: outputURL)
```

### Streaming

```swift
for try await chunk in pipeline.generateStream(text: "Hello world!", speaker: "Aiden") {
    // Play chunk.samples (Float array at 24kHz)
    if chunk.isFinal { break }
}
```

### Voice Cloning

```swift
// Extract speaker embedding from reference audio
if let embedding = pipeline.extractSpeakerEmbedding(audioSamples: referenceAudio) {
    let samples = pipeline.generate(text: "Hello!", speakerEmbedding: embedding)
}
```

### ICL Voice Cloning

Clone a voice by encoding reference audio into codes, then passing them to generation.

```swift
// Encode reference audio (24kHz float samples)
if let codes = pipeline.encodeReferenceAudio(audioSamples: referenceAudio) {
    // Generate speech in the cloned voice
    let sampleCount = try await pipeline.generateToFile(
        text: "Hello in a cloned voice!",
        speaker: "Aiden",
        referenceTranscript: "The transcript of the reference audio.",
        referenceAudioCodes: codes,
        outputURL: outputURL
    )
}
```

### VoiceDesign (1.7B only)

Generate speech using a natural language voice description. Requires a VoiceDesign model.

```swift
let pipeline = try Qwen3TTSPipeline(modelPath: voiceDesignModelURL)

// Generate with a described voice
let samples = pipeline.generateVoiceDesign(
    text: "Hello world!",
    voiceDescription: "A deep male voice with a British accent, speaking slowly and calmly"
)

// Streaming
for try await chunk in pipeline.generateStreamVoiceDesign(
    text: "Hello world!",
    voiceDescription: "A cheerful young female voice"
) {
    // Play chunk.samples
    if chunk.isFinal { break }
}
```

### CustomVoice (1.7B only)

Combine a named speaker with style/emotion instructions. Requires a CustomVoice model.

```swift
let pipeline = try Qwen3TTSPipeline(modelPath: customVoiceModelURL)

// Generate with speaker + style control
let samples = pipeline.generateCustomVoice(
    text: "I can't believe it!",
    speaker: "Vivian",
    instruct: "Say it with excitement and surprise"
)

// Streaming
for try await chunk in pipeline.generateStreamCustomVoice(
    text: "I'm so sorry to hear that.",
    speaker: "Aiden",
    instruct: "Speak with empathy and concern"
) {
    // Play chunk.samples
    if chunk.isFinal { break }
}
```

### Long Text to File

```swift
// Memory-efficient generation for long text
let sampleCount = try await pipeline.generateToFile(
    text: longArticleText,
    speaker: "Serena",
    outputURL: outputURL,
    onProgress: { progress in print("\(Int(progress * 100))%") }
)
```

## Model Download

Download a pre-quantized model from HuggingFace:

```bash
# Install git-lfs if needed
brew install git-lfs
git lfs install

# 0.6B models
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit  # ~2GB
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit  # ~1.7GB

# 1.7B models (higher quality, more memory)
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit  # ~3.1GB
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit  # ~2.3GB

# 1.7B VoiceDesign (generate voices from text descriptions)
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit

# 1.7B CustomVoice (named speakers + style/emotion instruct)
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
```

The model directory should contain:
- `config.json` - Model configuration
- `model.safetensors` - Model weights
- `tokenizer.json` - BPE tokenizer
- `speech_tokenizer/` - Vocoder (config.json + model.safetensors)

## Available Speakers

Built-in speakers (varies by model): aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

## API Reference

### `Qwen3TTSPipeline`

The main entry point. Load a model and generate speech.

| Method | Description |
|--------|-------------|
| `init(modelPath:configuration:)` | Load model from directory |
| `generate(text:speaker:)` | Simple generation with built-in voice |
| `generate(text:speakerEmbedding:)` | Generation with cloned voice |
| `generateStream(text:speaker:)` | Streaming generation |
| `generateVoiceDesign(text:voiceDescription:)` | Generate with described voice (VoiceDesign) |
| `generateStreamVoiceDesign(text:voiceDescription:)` | Stream with described voice (VoiceDesign) |
| `generateCustomVoice(text:speaker:instruct:)` | Generate with speaker + style (CustomVoice) |
| `generateStreamCustomVoice(text:speaker:instruct:)` | Stream with speaker + style (CustomVoice) |
| `generateToFile(text:speaker:outputURL:)` | Memory-efficient file output |
| `generateBatch(text:speaker:)` | Batch generation for long text |
| `extractSpeakerEmbedding(audioSamples:)` | Extract voice embedding |
| `encodeReferenceAudio(audioSamples:)` | Encode audio for ICL cloning |
| `clearCache()` | Free cached memory |

### Properties

| Property | Description |
|----------|-------------|
| `availableSpeakers` | Built-in speaker names |
| `supportsVoiceCloning` | Whether speaker encoder is loaded |
| `supportsICL` | Whether audio encoder is loaded |
| `supportsVoiceDesign` | Whether model supports VoiceDesign |
| `supportsCustomVoice` | Whether model supports CustomVoice |
| `modelType` | Raw model type string from config |
| `sampleRate` | Audio sample rate (24000 Hz) |

### `AudioSampleWriter`

Utility for writing WAV files.

```swift
// One-shot
try AudioSampleWriter.write(samples: samples, to: url)

// Streaming
let writer = try StreamingWAVWriter(to: url)
try writer.write(samples: chunk1)
try writer.write(samples: chunk2)
let result = writer.finalize()
```

## Configuration

```swift
let config = Qwen3TTSPipelineConfiguration(
    applyRuntimeQuantization: true,  // Quantize non-quantized models at load
    defaultTemperature: 0.85,        // Generation temperature
    defaultMaxTokens: 2400,          // Max tokens (~200s audio at 12Hz)
    defaultStreamingChunkSize: 12,   // Frames per streaming yield
    crossfadeSamples: 480            // Crossfade between text chunks (20ms)
)
let pipeline = try Qwen3TTSPipeline(modelPath: modelURL, configuration: config)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MLX_DEVICE=cpu` | Force CPU-only mode |

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen team
- [MLX Swift](https://github.com/ml-explore/mlx-swift) by Apple
