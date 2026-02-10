import XCTest
@testable import Qwen3TTS

final class ConfigTests: XCTestCase {

    func testStandardConfig() {
        let config = Qwen3TTSConfig.standard
        XCTAssertEqual(config.hidden_size, 1024)
        XCTAssertEqual(config.num_hidden_layers, 28)
        XCTAssertEqual(config.vocab_size, 3072)
        XCTAssertEqual(config.text_vocab_size, 151936)
        XCTAssertEqual(config.num_attention_heads, 16)
        XCTAssertEqual(config.num_key_value_heads, 8)
        XCTAssertEqual(config.head_dim, 128)
        XCTAssertEqual(config.codec_bos_id, 2149)
        XCTAssertEqual(config.codec_eos_token_id, 2150)
        XCTAssertEqual(config.codec_pad_id, 2148)
    }

    func testConfigDecoding() throws {
        let json = """
        {
            "talker_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "vocab_size": 3072,
                "text_vocab_size": 151936,
                "text_hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 3072,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "codec_bos_id": 2149,
                "codec_eos_token_id": 2150,
                "codec_pad_id": 2148,
                "spk_id": {"aiden": 2861, "serena": 3066},
                "code_predictor_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "intermediate_size": 3072,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 65536,
                    "rope_theta": 1000000.0,
                    "vocab_size": 2048,
                    "num_code_groups": 16
                }
            },
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "tts_pad_token_id": 151671
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)

        XCTAssertEqual(config.hidden_size, 1024)
        XCTAssertEqual(config.num_hidden_layers, 28)
        XCTAssertEqual(config.text_vocab_size, 151936)
        XCTAssertEqual(config.tts_bos_token_id, 151672)
        XCTAssertEqual(config.spk_id["aiden"], 2861)
        XCTAssertEqual(config.spk_id["serena"], 3066)
        XCTAssertEqual(config.code_predictor_config.num_code_groups, 16)
    }

    func testConfigDecodingWithQuantization() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "quantization": {
                "bits": 4,
                "group_size": 64
            }
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)

        XCTAssertNotNil(config.quantization)
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.quantization?.group_size, 64)

        let settings = config.quantizationSettings
        XCTAssertTrue(settings.enabled)
        XCTAssertEqual(settings.bits, 4)
        XCTAssertEqual(settings.groupSize, 64)
    }

    func testQuantizationSettings() {
        let q4 = QuantizationSettings.quantized4Bit
        XCTAssertTrue(q4.enabled)
        XCTAssertEqual(q4.bits, 4)
        XCTAssertEqual(q4.groupSize, 64)

        let q6 = QuantizationSettings.quantized6Bit
        XCTAssertTrue(q6.enabled)
        XCTAssertEqual(q6.bits, 6)

        let fp = QuantizationSettings.fullPrecision
        XCTAssertFalse(fp.enabled)
    }

    func testStandardConfigModelTypeIsNil() {
        let config = Qwen3TTSConfig.standard
        XCTAssertNil(config.tts_model_type)
    }

    func testConfigDecodingBaseModelType() throws {
        // No tts_model_type key → nil (base model)
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)
        XCTAssertNil(config.tts_model_type)
    }

    func testConfigDecodingVoiceDesignModelType() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tts_model_type": "voice_design"
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)
        XCTAssertEqual(config.tts_model_type, "voice_design")
    }

    func testConfigDecodingCustomVoiceModelType() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "vocab_size": 3072,
            "text_vocab_size": 151936,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tts_model_type": "custom_voice"
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)
        XCTAssertEqual(config.tts_model_type, "custom_voice")
    }

    func testConfigDecodingModelTypeInNestedTalkerConfig() throws {
        // tts_model_type is at root level, even when talker_config is nested
        let json = """
        {
            "talker_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "vocab_size": 3072,
                "text_vocab_size": 151936,
                "num_attention_heads": 16,
                "intermediate_size": 3072,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "spk_id": {}
            },
            "tts_model_type": "voice_design",
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "tts_pad_token_id": 151671
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)
        XCTAssertEqual(config.tts_model_type, "voice_design")
        XCTAssertEqual(config.hidden_size, 1024)
    }

    func testCodePredictorConfigDefaults() {
        let config = CodePredictorConfigJSON()
        XCTAssertEqual(config.hidden_size, 1024)
        XCTAssertEqual(config.num_hidden_layers, 5)
        XCTAssertEqual(config.num_attention_heads, 16)
        XCTAssertEqual(config.num_key_value_heads, 8)
        XCTAssertEqual(config.head_dim, 128)
        XCTAssertEqual(config.num_code_groups, 16)
        XCTAssertEqual(config.vocab_size, 2048)
    }
}
