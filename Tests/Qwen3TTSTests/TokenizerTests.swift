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
    
    func testByteLevelEncodingForNonLatinText() {
        // Build GPT/Qwen byte-to-unicode map used by byte-level BPE.
        var bs: [UInt8] = []
        bs += Array(UInt8(33)...UInt8(126))
        bs += Array(UInt8(161)...UInt8(172))
        bs += Array(UInt8(174)...UInt8(255))

        var cs = bs.map { Int($0) }
        var n = 0
        for b in UInt8.min...UInt8.max {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }

        var byteToUnicode: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) {
            if let scalar = UnicodeScalar(c) {
                byteToUnicode[b] = Character(scalar)
            }
        }

        let text = "あ"
        let utf8Bytes = Array(text.utf8)

        // Vocab intentionally excludes <0xXX> tokens and raw "あ".
        // It only includes byte-level mapped chars, which should be sufficient.
        var vocab: [String: Int] = [:]
        for (idx, byte) in utf8Bytes.enumerated() {
            vocab[String(byteToUnicode[byte]!)] = idx
        }

        let tokenizer = Qwen3Tokenizer(vocab: vocab, merges: [])
        let ids: [Int32] = tokenizer.encode(text: text)

        XCTAssertEqual(ids, [0, 1, 2])
    }
}
