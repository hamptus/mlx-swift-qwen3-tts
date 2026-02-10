import XCTest
@testable import Qwen3TTS

final class TokenizerTests: XCTestCase {

    func testManualInitialization() {
        let vocab: [String: Int] = [
            "hello": 0,
            "world": 1,
            "Ġhello": 2,
            "Ġworld": 3,
            "Ġ": 4,
            "h": 5,
            "e": 6,
            "l": 7,
            "o": 8,
            "w": 9,
            "r": 10,
            "d": 11,
        ]
        let merges = ["h e", "l l", "he ll", "hell o", "w o", "r l", "wo rl", "worl d"]
        let tokenizer = Qwen3Tokenizer(vocab: vocab, merges: merges)

        // Simple decode test
        let decoded = tokenizer.decode(ids: [2, 3])
        XCTAssertEqual(decoded, " hello world")
    }

    func testEmptyEncode() {
        let tokenizer = Qwen3Tokenizer(vocab: [:], merges: [])
        let ids: [Int32] = tokenizer.encode(text: "")
        XCTAssertTrue(ids.isEmpty)
    }

    func testEmptyDecode() {
        let tokenizer = Qwen3Tokenizer(vocab: [:], merges: [])
        let text = tokenizer.decode(ids: [])
        XCTAssertEqual(text, "")
    }

    func testQuoteNormalization() {
        // Smart quotes should be normalized to ASCII
        let vocab: [String: Int] = [
            "I": 0,
            "'": 1,
            "m": 2,
            "Ġ": 3,
        ]
        let tokenizer = Qwen3Tokenizer(vocab: vocab, merges: [])

        // Curly apostrophe should be normalized to straight
        let text1 = "I\u{2019}m"  // RIGHT SINGLE QUOTATION MARK
        let ids1: [Int32] = tokenizer.encode(text: text1)
        let text2 = "I'm"         // ASCII apostrophe
        let ids2: [Int32] = tokenizer.encode(text: text2)

        XCTAssertEqual(ids1, ids2)
    }

    func testUninitializedTokenizer() {
        // Tokenizer without model path falls back to UTF-8 bytes
        let tokenizer = Qwen3Tokenizer()
        let ids: [Int32] = tokenizer.encode(text: "Hi")
        // Without loading, encode returns UTF-8 bytes
        XCTAssertEqual(ids, [72, 105]) // 'H' = 72, 'i' = 105
    }
}
