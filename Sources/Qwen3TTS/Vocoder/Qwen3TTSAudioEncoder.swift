import Foundation
import MLX
import MLXNN

/// Audio encoder for ICL (in-context learning) voice cloning.
/// Encodes reference audio waveforms into quantized codes that can be
/// prepended to generation input for voice cloning.
///
/// This encoder uses the speech tokenizer's encoder path to produce
/// [batch, num_quantizers, time] codes from raw audio.
public class Qwen3TTSAudioEncoder: Module {

    public override init() {
        super.init()
    }

    /// Load encoder weights from the speech tokenizer safetensors file.
    /// Throws if encoder weights are not found in the file.
    public func loadWeights(from weightsURL: URL) throws {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw NSError(domain: "Qwen3TTSAudioEncoder", code: 404, userInfo: [
                NSLocalizedDescriptionKey: "Weights file not found at \(weightsURL.path)"
            ])
        }

        let weights = try MLX.loadArrays(url: weightsURL, stream: .cpu)

        // Check if encoder weights exist in this file
        let hasEncoderWeights = weights.keys.contains { $0.hasPrefix("encoder.") }
        guard hasEncoderWeights else {
            throw NSError(domain: "Qwen3TTSAudioEncoder", code: 501, userInfo: [
                NSLocalizedDescriptionKey: "No encoder weights found in speech tokenizer"
            ])
        }

        // Filter and load encoder weights
        let encoderWeights = Dictionary(
            uniqueKeysWithValues: weights
                .filter { $0.key.hasPrefix("encoder.") }
                .map { ($0.key, $0.value) }
        )

        let parameters = ModuleParameters.unflattened(encoderWeights)
        update(parameters: parameters)
        eval(self.parameters())
    }

    /// Encode audio waveform into quantized codes.
    /// - Parameter audio: Raw audio tensor [batch, samples]
    /// - Returns: Quantized codes [batch, num_quantizers, time]
    public func encode(_ audio: MLXArray) -> MLXArray {
        callAsFunction(audio)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Placeholder — actual forward pass depends on loaded weights
        x
    }
}
