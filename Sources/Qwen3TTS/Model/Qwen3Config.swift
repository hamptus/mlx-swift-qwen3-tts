import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// Code predictor config parsed from JSON
public struct CodePredictorConfigJSON: Codable, Sendable {
    public var hidden_size: Int
    public var num_hidden_layers: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var intermediate_size: Int
    public var rms_norm_eps: Float
    public var max_position_embeddings: Int
    public var rope_theta: Float
    public var vocab_size: Int
    public var num_code_groups: Int

    public init(
        hidden_size: Int = 1024,
        num_hidden_layers: Int = 5,
        num_attention_heads: Int = 16,
        num_key_value_heads: Int = 8,
        head_dim: Int = 128,
        intermediate_size: Int = 3072,
        rms_norm_eps: Float = 1e-6,
        max_position_embeddings: Int = 65536,
        rope_theta: Float = 1000000.0,
        vocab_size: Int = 2048,
        num_code_groups: Int = 16
    ) {
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.num_code_groups = num_code_groups
    }
}

public struct QuantizationConfig: Codable, Sendable {
    public var group_size: Int?
    public var bits: Int?
    public var mode: String?

    public init(group_size: Int? = nil, bits: Int? = nil, mode: String? = nil) {
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
    }

    /// Convert to runtime QuantizationSettings
    public var settings: QuantizationSettings {
        QuantizationSettings(from: self)
    }
}

public struct Qwen3TTSConfig: Codable, Sendable {
    public var hidden_size: Int
    public var num_hidden_layers: Int
    public var vocab_size: Int
    public var text_vocab_size: Int
    public var text_hidden_size: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var intermediate_size: Int
    public var rms_norm_eps: Float
    public var max_position_embeddings: Int
    public var rope_theta: Float

    // Special token IDs
    public var tts_bos_token_id: Int
    public var tts_eos_token_id: Int
    public var tts_pad_token_id: Int
    public var codec_bos_id: Int
    public var codec_eos_token_id: Int
    public var codec_pad_id: Int
    public var codec_nothink_id: Int
    public var codec_think_bos_id: Int
    public var codec_think_eos_id: Int

    // Speaker IDs
    public var spk_id: [String: Int]

    // Code predictor config
    public var code_predictor_config: CodePredictorConfigJSON

    // MRoPE section sizes
    public var mrope_section: [Int]?
    public var quantization: QuantizationConfig?
    public var quantization_config: QuantizationConfig?

    public static let standard = Qwen3TTSConfig(
        hidden_size: 1024,
        num_hidden_layers: 28,
        vocab_size: 3072,
        text_vocab_size: 151936,
        text_hidden_size: 2048,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        intermediate_size: 3072,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 32768,
        rope_theta: 1000000.0,
        tts_bos_token_id: 151672,
        tts_eos_token_id: 151673,
        tts_pad_token_id: 151671,
        codec_bos_id: 2149,
        codec_eos_token_id: 2150,
        codec_pad_id: 2148,
        codec_nothink_id: 2155,
        codec_think_bos_id: 2156,
        codec_think_eos_id: 2157,
        spk_id: ["serena": 3066, "vivian": 3065, "uncle_fu": 3010, "ryan": 3061, "aiden": 2861, "ono_anna": 2873, "sohee": 2864, "eric": 2875, "dylan": 2878],
        code_predictor_config: CodePredictorConfigJSON()
    )

    enum CodingKeys: String, CodingKey {
        case hidden_size
        case num_hidden_layers
        case vocab_size
        case text_vocab_size
        case text_hidden_size
        case num_attention_heads
        case num_key_value_heads
        case head_dim
        case intermediate_size
        case rms_norm_eps
        case max_position_embeddings
        case rope_theta
        case talker_config
        case tts_bos_token_id
        case tts_eos_token_id
        case tts_pad_token_id
        case codec_bos_id
        case codec_eos_token_id
        case codec_pad_id
        case codec_nothink_id
        case codec_think_bos_id
        case codec_think_eos_id
        case spk_id
        case code_predictor_config
        case rope_scaling
        case quantization
        case quantization_config
    }

    struct RopeScaling: Codable {
        var mrope_section: [Int]?
        var interleaved: Bool?
    }

    public init(
        hidden_size: Int, num_hidden_layers: Int, vocab_size: Int, text_vocab_size: Int,
        text_hidden_size: Int = 2048, num_attention_heads: Int, num_key_value_heads: Int = 8,
        head_dim: Int = 128, intermediate_size: Int, rms_norm_eps: Float,
        max_position_embeddings: Int, rope_theta: Float,
        tts_bos_token_id: Int = 151672, tts_eos_token_id: Int = 151673, tts_pad_token_id: Int = 151671,
        codec_bos_id: Int = 2149, codec_eos_token_id: Int = 2150, codec_pad_id: Int = 2148,
        codec_nothink_id: Int = 2155, codec_think_bos_id: Int = 2156, codec_think_eos_id: Int = 2157,
        spk_id: [String: Int] = [:],
        code_predictor_config: CodePredictorConfigJSON = CodePredictorConfigJSON(),
        mrope_section: [Int]? = nil
    ) {
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.text_hidden_size = text_hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.codec_bos_id = codec_bos_id
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_pad_id = codec_pad_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.spk_id = spk_id
        self.code_predictor_config = code_predictor_config
        self.mrope_section = mrope_section
        self.quantization = nil
        self.quantization_config = nil
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let sourceContainer: KeyedDecodingContainer<CodingKeys>
        if container.contains(.talker_config) {
            sourceContainer = try container.nestedContainer(keyedBy: CodingKeys.self, forKey: .talker_config)
        } else {
            sourceContainer = container
        }

        self.hidden_size = try sourceContainer.decode(Int.self, forKey: .hidden_size)
        self.num_hidden_layers = try sourceContainer.decode(Int.self, forKey: .num_hidden_layers)
        self.vocab_size = try sourceContainer.decode(Int.self, forKey: .vocab_size)
        self.text_vocab_size = try sourceContainer.decode(Int.self, forKey: .text_vocab_size)
        self.text_hidden_size = try sourceContainer.decodeIfPresent(Int.self, forKey: .text_hidden_size) ?? 2048
        self.num_attention_heads = try sourceContainer.decode(Int.self, forKey: .num_attention_heads)
        self.num_key_value_heads = try sourceContainer.decodeIfPresent(Int.self, forKey: .num_key_value_heads) ?? 8
        self.head_dim = try sourceContainer.decodeIfPresent(Int.self, forKey: .head_dim) ?? 128
        self.intermediate_size = try sourceContainer.decode(Int.self, forKey: .intermediate_size)
        self.rms_norm_eps = try sourceContainer.decode(Float.self, forKey: .rms_norm_eps)
        self.max_position_embeddings = try sourceContainer.decode(Int.self, forKey: .max_position_embeddings)
        self.rope_theta = try sourceContainer.decode(Float.self, forKey: .rope_theta)

        self.tts_bos_token_id = try container.decodeIfPresent(Int.self, forKey: .tts_bos_token_id) ?? 151672
        self.tts_eos_token_id = try container.decodeIfPresent(Int.self, forKey: .tts_eos_token_id) ?? 151673
        self.tts_pad_token_id = try container.decodeIfPresent(Int.self, forKey: .tts_pad_token_id) ?? 151671

        self.codec_bos_id = try sourceContainer.decodeIfPresent(Int.self, forKey: .codec_bos_id) ?? 2149
        self.codec_eos_token_id = try sourceContainer.decodeIfPresent(Int.self, forKey: .codec_eos_token_id) ?? 2150
        self.codec_pad_id = try sourceContainer.decodeIfPresent(Int.self, forKey: .codec_pad_id) ?? 2148
        self.codec_nothink_id = try sourceContainer.decodeIfPresent(Int.self, forKey: .codec_nothink_id) ?? 2155
        self.codec_think_bos_id = try sourceContainer.decodeIfPresent(Int.self, forKey: .codec_think_bos_id) ?? 2156
        self.codec_think_eos_id = try sourceContainer.decodeIfPresent(Int.self, forKey: .codec_think_eos_id) ?? 2157
        self.spk_id = try sourceContainer.decodeIfPresent([String: Int].self, forKey: .spk_id) ?? [:]

        self.code_predictor_config = try sourceContainer.decodeIfPresent(CodePredictorConfigJSON.self, forKey: .code_predictor_config) ?? CodePredictorConfigJSON()

        if let ropeScaling = try sourceContainer.decodeIfPresent(RopeScaling.self, forKey: .rope_scaling) {
            self.mrope_section = ropeScaling.mrope_section
        } else {
            self.mrope_section = nil
        }
        self.quantization = try container.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        self.quantization_config = try container.decodeIfPresent(QuantizationConfig.self, forKey: .quantization_config)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hidden_size, forKey: .hidden_size)
        try container.encode(num_hidden_layers, forKey: .num_hidden_layers)
        try container.encode(vocab_size, forKey: .vocab_size)
        try container.encode(text_vocab_size, forKey: .text_vocab_size)
        try container.encode(text_hidden_size, forKey: .text_hidden_size)
        try container.encode(num_attention_heads, forKey: .num_attention_heads)
        try container.encode(num_key_value_heads, forKey: .num_key_value_heads)
        try container.encode(head_dim, forKey: .head_dim)
        try container.encode(intermediate_size, forKey: .intermediate_size)
        try container.encode(rms_norm_eps, forKey: .rms_norm_eps)
        try container.encode(max_position_embeddings, forKey: .max_position_embeddings)
        try container.encode(rope_theta, forKey: .rope_theta)
        try container.encodeIfPresent(quantization, forKey: .quantization)
        try container.encodeIfPresent(quantization_config, forKey: .quantization_config)
    }

    /// Get quantization settings (prefers quantization_config over quantization)
    public var quantizationSettings: QuantizationSettings {
        if let config = quantization_config ?? quantization {
            return config.settings
        }
        return .fullPrecision
    }
}

/// Code predictor configuration (runtime)
public struct CodePredictorConfig: Sendable {
    public var hidden_size: Int = 1024
    public var num_hidden_layers: Int = 5
    public var num_attention_heads: Int = 16
    public var num_key_value_heads: Int = 8
    public var head_dim: Int = 128
    public var intermediate_size: Int = 3072
    public var rms_norm_eps: Float = 1e-6
    public var max_position_embeddings: Int = 65536
    public var rope_theta: Float = 1000000.0
    public var vocab_size: Int = 2048
    public var num_code_groups: Int = 16
    public var quantization: QuantizationSettings = .fullPrecision

    public init(
        hidden_size: Int = 1024, num_hidden_layers: Int = 5, num_attention_heads: Int = 16,
        num_key_value_heads: Int = 8, head_dim: Int = 128, intermediate_size: Int = 3072,
        rms_norm_eps: Float = 1e-6, max_position_embeddings: Int = 65536,
        rope_theta: Float = 1000000.0, vocab_size: Int = 2048, num_code_groups: Int = 16,
        quantization: QuantizationSettings = .fullPrecision
    ) {
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.num_code_groups = num_code_groups
        self.quantization = quantization
    }
}
