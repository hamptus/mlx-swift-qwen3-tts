import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct AudioDecoderConfig: Codable, Sendable {
    public struct DecoderInternalConfig: Codable, Sendable {
        public var upsample_rates: [Int]?
        public var upsampling_ratios: [Int]?
        public var decoder_dim: Int?
        public var codebook_size: Int?
        public var codebook_dim: Int?
        public var num_hidden_layers: Int?
        public var num_attention_heads: Int?
        public var num_key_value_heads: Int?
        public var hidden_size: Int?
        public var intermediate_size: Int?
        public var latent_dim: Int?
        public var num_quantizers: Int?
        public var num_semantic_quantizers: Int?
        public var head_dim: Int?
        public var rms_norm_eps: Float?
        public var rope_theta: Float?
        public var layer_scale_initial_scale: Float?
        public var max_position_embeddings: Int?
        public var attention_bias: Bool?
        public var sliding_window: Int?
        public var quantization: QuantizationConfig?

        public init() {}

        public func toQwen3Config() -> Qwen3TTSTokenizerDecoderConfig {
            var config = Qwen3TTSTokenizerDecoderConfig()
            if let v = upsample_rates { config.upsample_rates = v }
            if let v = upsampling_ratios { config.upsampling_ratios = v }
            if let v = decoder_dim { config.decoder_dim = v }
            if let v = codebook_size { config.codebook_size = v }
            if let v = codebook_dim { config.codebook_dim = v }
            if let v = num_hidden_layers { config.num_hidden_layers = v }
            if let v = num_attention_heads { config.num_attention_heads = v }
            if let v = num_key_value_heads { config.num_key_value_heads = v }
            if let v = hidden_size { config.hidden_size = v }
            if let v = intermediate_size { config.intermediate_size = v }
            if let v = latent_dim { config.latent_dim = v }
            if let v = num_quantizers { config.num_quantizers = v }
            if let v = num_semantic_quantizers { config.num_semantic_quantizers = v }
            if let v = head_dim { config.head_dim = v }
            if let v = rms_norm_eps { config.rms_norm_eps = v }
            if let v = rope_theta { config.rope_theta = v }
            if let v = layer_scale_initial_scale { config.layer_scale_initial_scale = v }
            if let v = max_position_embeddings { config.max_position_embeddings = v }
            if let v = attention_bias { config.attention_bias = v }
            if let v = sliding_window { config.sliding_window = v }
            config.quantization = quantization
            return config
        }
    }

    public var decoder_config: DecoderInternalConfig?
    public var input_sample_rate: Int?
    public var output_sample_rate: Int?
    public var decode_upsample_rate: Int?
    public var encoder_valid_num_quantizers: Int?

    public init() {}

    enum CodingKeys: String, CodingKey {
        case decoder_config
        case input_sample_rate
        case output_sample_rate
        case decode_upsample_rate
        case encoder_valid_num_quantizers
    }

    public var codebookDim: Int {
        decoder_config?.codebook_dim ?? 512
    }
    public var decoderDim: Int {
        decoder_config?.decoder_dim ?? 1536
    }
    public var upsampleRates: [Int] {
        decoder_config?.upsample_rates ?? [8, 5, 4, 3]
    }
    public var codebookSize: Int {
        decoder_config?.codebook_size ?? 2048
    }

    public func toQwen3TokenizerConfig() -> Qwen3TTSTokenizerConfig {
        var config = Qwen3TTSTokenizerConfig()
        if let dc = decoder_config {
            config.decoder_config = dc.toQwen3Config()
        }
        if let v = input_sample_rate { config.input_sample_rate = v }
        if let v = output_sample_rate { config.output_sample_rate = v }
        if let v = decode_upsample_rate { config.decode_upsample_rate = v }
        if let v = encoder_valid_num_quantizers { config.encoder_valid_num_quantizers = v }
        return config
    }
}

// MARK: - AudioDecoder

public class AudioDecoder: Module {
    private let config: AudioDecoderConfig
    private var mlxDecoder: Qwen3TTSSpeechTokenizerDecoder?

    public var hasMLXDecoder: Bool {
        mlxDecoder != nil
    }

    public required nonisolated override init() {
        self.config = AudioDecoderConfig()
        super.init()
    }

    nonisolated public init(config: AudioDecoderConfig) {
        self.config = config
        super.init()
    }

    public func unload() {
        mlxDecoder = nil
    }

    public func clearCompiledCache() {
        mlxDecoder?.clearCompiledCache()
    }

    @discardableResult
    public func loadMLXDecoder(configURL: URL, weightsURL: URL) -> Bool {
        do {
            let data = try Data(contentsOf: configURL)
            let jsonConfig = try JSONDecoder().decode(AudioDecoderConfig.self, from: data)
            let decoderConfig = jsonConfig.decoder_config?.toQwen3Config() ?? Qwen3TTSTokenizerDecoderConfig()

            let decoder = Qwen3TTSSpeechTokenizerDecoder(config: decoderConfig)

            var weights = try MLX.loadArrays(url: weightsURL, stream: .cpu)
            let sanitized = AudioDecoder.sanitize(weights: weights)
            let parameters = ModuleParameters.unflattened(sanitized)
            try decoder.update(parameters: parameters, verify: .noUnusedKeys)

            mlxDecoder = decoder
            weights = [:]
            Memory.clearCache()

            return true
        } catch {
            mlxDecoder = nil
            return false
        }
    }

    public func chunkedDecode(codes: MLXArray, chunkSize: Int = 100, leftContextSize: Int = 10) -> MLXArray {
        guard let mlxDecoder = mlxDecoder else {
            return MLXArray.zeros([1, 0, 1], dtype: .float32)
        }

        let transposedCodes = codes.transposed(0, 2, 1)
        let wav = mlxDecoder.chunkedDecode(transposedCodes, chunkSize: chunkSize, leftContextSize: leftContextSize)
        return wav
    }

    public func mlxDecode(codes: MLXArray) -> MLXArray {
        guard let mlxDecoder = mlxDecoder else {
            return MLXArray.zeros([1, 0, 1], dtype: .float32)
        }

        let transposedCodes = codes.transposed(0, 2, 1)
        let wav = mlxDecoder(transposedCodes)
        return wav
    }

    public func decode(codes: MLXArray) -> MLXArray {
        guard let _ = mlxDecoder else {
            return MLXArray.zeros([1, 1, 0], dtype: .float32)
        }
        return mlxDecode(codes: codes)
    }

    private static func snakeToCamel(_ input: String) -> String {
        let parts = input.split(separator: "_")
        guard !parts.isEmpty else { return input }

        let first = String(parts[0])
        let rest = parts.dropFirst().map { part in
            guard let firstChar = part.first else { return String(part) }
            return String(firstChar).uppercased() + String(part.dropFirst())
        }
        return first + rest.joined()
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var codebookData: [String: [String: MLXArray]] = [:]

        for (key, value) in weights {
            var v = value
            var workingKey = key

            if workingKey.hasPrefix("audio_decoder.") {
                workingKey = String(workingKey.dropFirst("audio_decoder.".count))
            }

            if workingKey.hasPrefix("decoder.") && !workingKey.hasPrefix("decoder.decoder.") {
                workingKey = String(workingKey.dropFirst("decoder.".count))
            } else if workingKey.hasPrefix("decoder.decoder.") {
                workingKey = String(workingKey.dropFirst("decoder.".count))
            }

            if workingKey.hasPrefix("encoder.") || workingKey.contains(".encoder.") {
                continue
            }

            if workingKey.contains("_codebook.cluster_usage") || workingKey.contains("_codebook.embedding_sum") {
                let parts = workingKey.components(separatedBy: "._codebook.")
                if parts.count == 2 {
                    let basePath = parts[0]
                    if codebookData[basePath] == nil {
                        codebookData[basePath] = [:]
                    }
                    if workingKey.contains("cluster_usage") {
                        codebookData[basePath]?["cluster_usage"] = v
                    } else {
                        codebookData[basePath]?["embedding_sum"] = v
                    }
                }
                continue
            }

            var newKey = workingKey

            let components = newKey.components(separatedBy: ".")
            let camelComponents = components.map { component -> String in
                if Int(component) != nil {
                    return component
                }
                return snakeToCamel(component)
            }
            newKey = camelComponents.joined(separator: ".")

            newKey = newKey
                .replacingOccurrences(of: "rvqFirst", with: "rvqFirst")
                .replacingOccurrences(of: "rvqRest", with: "rvqRest")
                .replacingOccurrences(of: "selfAttn", with: "selfAttn")
                .replacingOccurrences(of: "inputLayernorm", with: "inputLayernorm")
                .replacingOccurrences(of: "postAttentionLayernorm", with: "postAttentionLayernorm")
                .replacingOccurrences(of: "selfAttnLayerScale", with: "selfAttnLayerScale")
                .replacingOccurrences(of: "mlpLayerScale", with: "mlpLayerScale")
                .replacingOccurrences(of: "qProj", with: "qProj")
                .replacingOccurrences(of: "kProj", with: "kProj")
                .replacingOccurrences(of: "vProj", with: "vProj")
                .replacingOccurrences(of: "oProj", with: "oProj")
                .replacingOccurrences(of: "gateProj", with: "gateProj")
                .replacingOccurrences(of: "upProj", with: "upProj")
                .replacingOccurrences(of: "downProj", with: "downProj")
                .replacingOccurrences(of: "inputProj", with: "inputProj")
                .replacingOccurrences(of: "outputProj", with: "outputProj")
                .replacingOccurrences(of: "preTransformer", with: "preTransformer")
                .replacingOccurrences(of: "preConv", with: "preConv")
                .replacingOccurrences(of: "rotaryEmb", with: "rotaryEmb")
                .replacingOccurrences(of: "invFreq", with: "invFreq")
                .replacingOccurrences(of: "projectOut", with: "projectOut")
                .replacingOccurrences(of: "dwconv", with: "dwconv")
                .replacingOccurrences(of: "pwconv1", with: "pwconv1")
                .replacingOccurrences(of: "pwconv2", with: "pwconv2")

            let isTransposeConv = (workingKey.contains("upsample") && workingKey.contains(".0.conv.")) ||
                                  workingKey.contains(".block.1.conv.")

            if isTransposeConv && workingKey.hasSuffix(".weight") && v.ndim == 3 {
                v = v.transposed(1, 2, 0)
            } else if workingKey.contains("conv") && workingKey.hasSuffix(".weight") && v.ndim == 3 {
                v = v.transposed(0, 2, 1)
            } else if workingKey.contains("_proj") && workingKey.hasSuffix(".weight") && v.ndim == 3 {
                v = v.transposed(0, 2, 1)
            }

            sanitized[newKey] = v
        }

        let eps: Float = 1e-5
        for (basePath, data) in codebookData {
            if let clusterUsage = data["cluster_usage"],
               let embeddingSum = data["embedding_sum"] {
                let usage = clip(clusterUsage, min: eps, max: Float.greatestFiniteMagnitude)
                let embedding = embeddingSum / usage.expandedDimensions(axis: -1)

                let components = basePath.components(separatedBy: ".")
                let camelComponents = components.map { component -> String in
                    if Int(component) != nil { return component }
                    return snakeToCamel(component)
                }
                let camelPath = camelComponents.joined(separator: ".")

                let newKey = "\(camelPath).codebook.embed.weight"
                sanitized[newKey] = embedding
            }
        }

        return sanitized
    }
}
