import Foundation
import Qwen3TTS

// MARK: - Basic Qwen3 TTS Generation Example
//
// Prerequisites:
//   Download a model from HuggingFace, e.g.:
//     git lfs install
//     git clone https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
//
// Usage:
//   swift run BasicGeneration /path/to/Qwen3-TTS-12Hz-0.6B-Base-8bit

guard CommandLine.arguments.count > 1 else {
    print("Usage: BasicGeneration <model-path> [output.wav] [speaker] [text]")
    print("")
    print("  model-path  Path to the Qwen3-TTS model directory")
    print("  output.wav  Output WAV file (default: output.wav)")
    print("  speaker     Speaker name (default: Aiden)")
    print("  text        Text to synthesize (default: hello world)")
    exit(1)
}

let modelPath = URL(fileURLWithPath: CommandLine.arguments[1])
let outputPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "output.wav"
let speaker = CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : "Aiden"
let text = CommandLine.arguments.count > 4 ? CommandLine.arguments[4] : "Hello! This is a test of the Qwen3 text to speech system."

print("Loading model from: \(modelPath.path)")
let startLoad = CFAbsoluteTimeGetCurrent()

do {
    let pipeline = try Qwen3TTSPipeline(modelPath: modelPath)
    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    print("Model loaded in \(String(format: "%.1f", loadTime))s")
    print("Available speakers: \(pipeline.availableSpeakers.joined(separator: ", "))")
    print("Voice cloning: \(pipeline.supportsVoiceCloning ? "supported" : "not available")")

    print("\nGenerating speech...")
    print("  Speaker: \(speaker)")
    print("  Text: \(text)")

    let startGen = CFAbsoluteTimeGetCurrent()
    let samples = pipeline.generate(text: text, speaker: speaker)
    let genTime = CFAbsoluteTimeGetCurrent() - startGen

    guard !samples.isEmpty else {
        print("Error: No audio samples generated")
        exit(1)
    }

    let durationSec = Double(samples.count) / Double(Qwen3TTSPipeline.sampleRate)
    print("Generated \(samples.count) samples (\(String(format: "%.1f", durationSec))s audio) in \(String(format: "%.1f", genTime))s")
    print("Real-time factor: \(String(format: "%.2f", genTime / durationSec))x")

    // Write to WAV file
    let outputURL = URL(fileURLWithPath: outputPath)
    try AudioSampleWriter.write(samples: samples, to: outputURL)
    print("Saved to: \(outputURL.path)")

} catch {
    print("Error: \(error.localizedDescription)")
    exit(1)
}
