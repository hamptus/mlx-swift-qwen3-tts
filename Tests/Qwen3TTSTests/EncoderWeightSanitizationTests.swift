import XCTest
import MLX
@testable import Qwen3TTS

/// Model-free unit tests for `Qwen3TTSAudioEncoder.sanitizeEncoderWeights`.
///
/// These verify the weight-key handling that previously caused the Mimi encoder
/// codebooks to load on random init weights (zero of 96 encoder codebook keys
/// matched the decoder-format pattern). They feed synthetic weight dictionaries
/// instead of downloading a model.
///
/// `sanitizeEncoderWeights` builds `MLXArray`s, and instantiating an `MLXArray`
/// initialises the Metal backend. The native SwiftPM test runner cannot locate
/// mlx-swift's `default.metallib` (it ships in the Xcode build system's resource
/// bundle), so these tests are gated behind `QWEN3TTS_RUN_MLX_TESTS=1` to avoid a
/// hard abort under `swift test`. Run them with:
///
///     QWEN3TTS_RUN_MLX_TESTS=1 swift test --build-system xcode
///
/// or from Xcode / xcodebuild, where the Metal library is available.
final class EncoderWeightSanitizationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["QWEN3TTS_RUN_MLX_TESTS"] == "1",
            "MLX Metal runtime unavailable under the native SwiftPM test runner. "
            + "Set QWEN3TTS_RUN_MLX_TESTS=1 and run with --build-system xcode (or in Xcode)."
        )
    }

    func testEncoderFormatCodebookIsReconstructed() throws {
        try requireMLXRuntime()
        // encoder format: ".codebook.{cluster_usage,embed_sum}"
        let clusterUsage = MLXArray([2.0, 4.0] as [Float])               // [2]
        let embedSum = MLXArray([2, 2, 2, 8, 8, 8] as [Float]).reshaped([2, 3])

        let sanitized = Qwen3TTSAudioEncoder.sanitizeEncoderWeights([
            "encoder.quantizer.codebook.cluster_usage": clusterUsage,
            "encoder.quantizer.codebook.embed_sum": embedSum,
        ])

        let key = "quantizer.codebook.embed.weight"
        guard let embedding = sanitized[key] else {
            return XCTFail("encoder-format codebook produced no \(key)")
        }
        XCTAssertEqual(embedding.shape, [2, 3])
        eval(embedding)
        // embed_sum / cluster_usage -> [[1,1,1],[2,2,2]]
        XCTAssertEqual(embedding.asArray(Float.self), [1, 1, 1, 2, 2, 2])
    }

    func testDecoderFormatCodebookIsReconstructed() throws {
        try requireMLXRuntime()
        // decoder format: "._codebook.{cluster_usage,embedding_sum}"
        let clusterUsage = MLXArray([1.0, 5.0] as [Float])
        let embeddingSum = MLXArray([3, 3, 3, 5, 5, 5] as [Float]).reshaped([2, 3])

        let sanitized = Qwen3TTSAudioEncoder.sanitizeEncoderWeights([
            "encoder.quantizer._codebook.cluster_usage": clusterUsage,
            "encoder.quantizer._codebook.embedding_sum": embeddingSum,
        ])

        guard let embedding = sanitized["quantizer.codebook.embed.weight"] else {
            return XCTFail("decoder-format codebook produced no embed.weight")
        }
        eval(embedding)
        XCTAssertEqual(embedding.asArray(Float.self), [3, 3, 3, 1, 1, 1])
    }

    func testInitializedFlagsAreDropped() throws {
        try requireMLXRuntime()
        let sanitized = Qwen3TTSAudioEncoder.sanitizeEncoderWeights([
            "encoder.quantizer.codebook.initialized": MLXArray([1] as [Int32]),
        ])
        XCTAssertTrue(sanitized.keys.allSatisfy { !$0.contains("initialized") })
    }

    func testConvWeightsAreTransposedToMLXLayout() throws {
        try requireMLXRuntime()
        // safetensors conv weight is [out, in, k]; MLX Conv1d wants [out, k, in].
        let convWeight = MLXArray((0..<24).map { Float($0) }).reshaped([2, 3, 4])
        let sanitized = Qwen3TTSAudioEncoder.sanitizeEncoderWeights([
            "encoder.conv.weight": convWeight,
        ])
        guard let transposed = sanitized["conv.weight"] else {
            return XCTFail("conv weight missing after sanitisation")
        }
        XCTAssertEqual(transposed.shape, [2, 4, 3])
    }

    func testNonEncoderKeysAreDropped() throws {
        try requireMLXRuntime()
        let sanitized = Qwen3TTSAudioEncoder.sanitizeEncoderWeights([
            "decoder.layer.weight": MLXArray([0.0] as [Float]),
            "quantizer.weight": MLXArray([0.0] as [Float]),
        ])
        XCTAssertTrue(sanitized.isEmpty, "only encoder.* keys should be retained")
    }
}
