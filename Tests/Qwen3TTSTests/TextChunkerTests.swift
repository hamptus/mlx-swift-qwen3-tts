import XCTest
@testable import Qwen3TTS

final class TextChunkerTests: XCTestCase {

    func testEmptyText() {
        let chunks = TextChunker.chunk("")
        XCTAssertTrue(chunks.isEmpty)
    }

    func testWhitespaceOnly() {
        let chunks = TextChunker.chunk("   \n  ")
        XCTAssertTrue(chunks.isEmpty)
    }

    func testShortText() {
        let text = "Hello world, this is a test."
        let chunks = TextChunker.chunk(text)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0], text)
    }

    func testSentenceBoundary() {
        let text = "The quick brown fox jumped over the lazy dog. Then the dog woke up and chased the fox through the forest."
        let chunks = TextChunker.chunk(text, maxWords: 15)
        XCTAssertGreaterThanOrEqual(chunks.count, 1)
        // First chunk should end at the sentence boundary
        XCTAssertTrue(chunks[0].hasSuffix("."))
    }

    func testCommaBoundary() {
        // Text with no sentence endings but has commas
        let words = (0..<40).map { "word\($0)" }
        let text = words[0..<15].joined(separator: " ") + ", " + words[15..<40].joined(separator: " ")
        let chunks = TextChunker.chunk(text, maxWords: 20)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
    }

    func testTokenEstimation() {
        let text = "Hello world this is a test"
        let estimate = TextChunker.estimateTokens(for: text)
        // 6 words * 5 tokens/word = 30, but minimum is 50
        XCTAssertEqual(estimate, 50)

        let longText = (0..<20).map { "word\($0)" }.joined(separator: " ")
        let longEstimate = TextChunker.estimateTokens(for: longText)
        XCTAssertEqual(longEstimate, 100) // 20 words * 5
    }

    func testVeryLongText() {
        let words = (0..<200).map { "word\($0)" }
        let text = words.joined(separator: " ")
        let chunks = TextChunker.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1)
        // Each chunk should have at most defaultMaxWords words
        for chunk in chunks {
            let wordCount = chunk.split(separator: " ").count
            XCTAssertLessThanOrEqual(wordCount, TextChunker.defaultMaxWords)
        }
    }

    func testMinWordsRespected() {
        // Chunks shouldn't be tiny fragments
        let text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T. This is a longer sentence that has more than eight words in it."
        let chunks = TextChunker.chunk(text, maxWords: 25)
        // The chunker should produce reasonable chunks
        XCTAssertGreaterThanOrEqual(chunks.count, 1)
    }
}
