import XCTest
@testable import Qwen3TTS

/// Model-free unit tests for the chunking and decode-stitching helpers added to
/// fix multi-chunk ICL degeneration (`effectiveChunkWords`) and the chunk-boundary
/// dead-air that stacked into audible gaps (`trimEdgeSilence`). Both are pure
/// `[Float]`/`Int` functions, so they run without the Metal backend.
final class DecodeAndChunkingTests: XCTestCase {

    // MARK: effectiveChunkWords (Fix B)

    func testICLFloorsSmallChunkSize() {
        // A small requested size is raised to the ICL minimum so chunks stay long
        // enough for the talker to emit EOS instead of over-generating.
        XCTAssertEqual(Qwen3TTSPipeline.effectiveChunkWords(12, isICL: true), Qwen3TTSPipeline.iclMinChunkWords)
        XCTAssertEqual(Qwen3TTSPipeline.effectiveChunkWords(30, isICL: true), Qwen3TTSPipeline.iclMinChunkWords)
    }

    func testICLHonoursLargerRequest() {
        // A caller asking for larger chunks (e.g. 120) is left untouched.
        XCTAssertEqual(Qwen3TTSPipeline.effectiveChunkWords(120, isICL: true), 120)
    }

    func testNonICLNeverFloored() {
        // Plain generation has the per-step text-consumption stop signal, so its
        // chunk size is never altered.
        XCTAssertEqual(Qwen3TTSPipeline.effectiveChunkWords(12, isICL: false), 12)
        XCTAssertEqual(Qwen3TTSPipeline.effectiveChunkWords(200, isICL: false), 200)
    }

    // MARK: trimEdgeSilence (Fix A')

    func testTrimsTrailingSilenceLeavingMargin() {
        let speech = [Float](repeating: 0.3, count: 5000)
        let silence = [Float](repeating: 0.0, count: 10000)
        let trimmed = Qwen3TTSPipeline.trimEdgeSilence(speech + silence, fromStart: false, threshold: 0.02, keepSamples: 1200, maxTrimSamples: 14400)
        // 10000 trailing silent samples, keep 1200 -> drop 8800.
        XCTAssertEqual(trimmed.count, 5000 + 1200)
    }

    func testTrimsLeadingSilenceLeavingMargin() {
        let silence = [Float](repeating: 0.0, count: 6000)
        let speech = [Float](repeating: 0.3, count: 5000)
        let trimmed = Qwen3TTSPipeline.trimEdgeSilence(silence + speech, fromStart: true, threshold: 0.02, keepSamples: 1200, maxTrimSamples: 14400)
        XCTAssertEqual(trimmed.count, 1200 + 5000)
    }

    func testNeverTrimsMoreThanMax() {
        let silence = [Float](repeating: 0.0, count: 50000)
        let trimmed = Qwen3TTSPipeline.trimEdgeSilence(silence, fromStart: false, threshold: 0.02, keepSamples: 0, maxTrimSamples: 14400)
        XCTAssertEqual(trimmed.count, 50000 - 14400)
    }

    func testLeavesAllSpeechUntouched() {
        let speech = (0..<8000).map { Float(($0 % 100) - 50) / 100.0 }
        let trimmed = Qwen3TTSPipeline.trimEdgeSilence(speech, fromStart: false)
        XCTAssertEqual(trimmed.count, speech.count)
    }

    func testEmptyInputIsSafe() {
        XCTAssertTrue(Qwen3TTSPipeline.trimEdgeSilence([], fromStart: true).isEmpty)
        XCTAssertTrue(Qwen3TTSPipeline.trimEdgeSilence([], fromStart: false).isEmpty)
    }

    func testKeepLargerThanRunIsNoOp() {
        // Trailing silent run shorter than the keep margin -> nothing dropped.
        let s = [Float](repeating: 0.3, count: 3000) + [Float](repeating: 0.0, count: 500)
        let trimmed = Qwen3TTSPipeline.trimEdgeSilence(s, fromStart: false, threshold: 0.02, keepSamples: 1200, maxTrimSamples: 14400)
        XCTAssertEqual(trimmed.count, s.count)
    }
}
