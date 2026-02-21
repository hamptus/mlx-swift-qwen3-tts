import Foundation

public class Qwen3Tokenizer {
    private var vocab: [String: Int] = [:]
    private var tokens: [Int: String] = [:]
    private var merges: [String: Int] = [:]
    private var loaded: Bool = false

    // Cache for BPE results to speed up repeated words
    // Limited to prevent unbounded memory growth in long sessions
    private var cache: [String: [String]] = [:]
    private let maxCacheSize = 10000

    // Special tokens that should be looked up directly, not BPE'd
    // These are sorted by length (longest first) to match greedily
    private var specialTokens: [String] = []

    // JSON structures for decoding
    private struct TokenizerJSON: Codable {
        let model: ModelJSON
        let added_tokens: [AddedToken]?
    }

    private struct ModelJSON: Codable {
        let vocab: [String: Int]
        let merges: [[String]]
    }

    private struct AddedToken: Codable {
        let id: Int
        let content: String
        let special: Bool?
    }

    private struct TokenizerConfigJSON: Codable {
        let added_tokens_decoder: [String: AddedTokenDecoder]?
    }

    private struct AddedTokenDecoder: Codable {
        let content: String
        let special: Bool?
    }

    public init(modelPath: URL? = nil) {
        if let modelPath = modelPath {
            do {
                try load(from: modelPath)
                self.loaded = true
            } catch {
                // Tokenizer load failed
            }
        }
    }

    // For manual initialization
    public init(vocab: [String: Int], merges: [String]) {
        self.vocab = vocab
        self.tokens = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        for (i, merge) in merges.enumerated() {
            self.merges[merge] = i
        }

        // Extract special tokens
        self.specialTokens = vocab.keys.filter { key in
            (key.hasPrefix("<|") && key.hasSuffix("|>")) ||
            (key.hasPrefix("<") && key.hasSuffix(">") && !key.contains(" "))
        }.sorted { $0.count > $1.count }

        self.loaded = true
    }

    private func load(from url: URL) throws {
        let tokenizerURL = url.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerURL.path) {
            let data = try Data(contentsOf: tokenizerURL)

            let decoder = JSONDecoder()
            let tokenizerData = try decoder.decode(TokenizerJSON.self, from: data)

            self.vocab = tokenizerData.model.vocab
            self.tokens = Dictionary(uniqueKeysWithValues: self.vocab.map { ($1, $0) })

            // Parse merges
            // Merges are lists of pairs ["a", "b"] -> "a b"
            for (i, pair) in tokenizerData.model.merges.enumerated() {
                if pair.count == 2 {
                    let mergeKey = pair[0] + " " + pair[1]
                    self.merges[mergeKey] = i
                }
            }

            // Load added_tokens - these include special tokens like <|im_start|>
            var addedSpecialTokens: [String] = []
            if let addedTokens = tokenizerData.added_tokens {
                for token in addedTokens {
                    // Add to vocab
                    vocab[token.content] = token.id
                    tokens[token.id] = token.content

                    // Track special tokens
                    if token.special == true {
                        addedSpecialTokens.append(token.content)
                    }
                }
            }

            // Special tokens are from added_tokens with special=true
            // Sort by length descending for greedy matching
            self.specialTokens = addedSpecialTokens.sorted { $0.count > $1.count }
        } else {
            let vocabURL = url.appendingPathComponent("vocab.json")
            let mergesURL = url.appendingPathComponent("merges.txt")

            guard FileManager.default.fileExists(atPath: vocabURL.path),
                  FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw NSError(domain: "Tokenizer", code: 404, userInfo: [
                    NSLocalizedDescriptionKey: "Tokenizer files not found."
                ])
            }

            let vocabData = try Data(contentsOf: vocabURL)
            self.vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
            self.tokens = Dictionary(uniqueKeysWithValues: self.vocab.map { ($1, $0) })

            let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
            let mergeLines = mergesText.split(separator: "\n").map { String($0) }
            for (i, line) in mergeLines.enumerated() where !line.isEmpty {
                let parts = line.split(separator: " ")
                if parts.count == 2 {
                    let mergeKey = String(parts[0]) + " " + String(parts[1])
                    self.merges[mergeKey] = i
                }
            }

            // Pull in special tokens from tokenizer_config.json if present.
            let configURL = url.appendingPathComponent("tokenizer_config.json")
            if FileManager.default.fileExists(atPath: configURL.path) {
                if let configData = try? Data(contentsOf: configURL),
                   let config = try? JSONDecoder().decode(TokenizerConfigJSON.self, from: configData),
                   let addedTokens = config.added_tokens_decoder {
                    var addedSpecialTokens: [String] = []
                    for (idString, token) in addedTokens {
                        if let id = Int(idString) {
                            vocab[token.content] = id
                            tokens[id] = token.content
                            if token.special == true {
                                addedSpecialTokens.append(token.content)
                            }
                        }
                    }
                    self.specialTokens = addedSpecialTokens.sorted { $0.count > $1.count }
                }
            }
        }

        if specialTokens.isEmpty {
            // Fallback: detect special tokens by naming convention.
            self.specialTokens = vocab.keys.filter { key in
                (key.hasPrefix("<|") && key.hasSuffix("|>")) ||
                (key.hasPrefix("<") && key.hasSuffix(">") && !key.contains(" "))
            }.sorted { $0.count > $1.count }
        }
    }

    public func encode(text: String) -> [Int32] {
        if !loaded {
            return text.utf8.map { Int32($0) }
        }

        // Normalize text: convert smart quotes/apostrophes to ASCII equivalents
        // This ensures contractions like "I'm" (with curly apostrophe) are tokenized correctly
        let normalizedText = normalizeQuotes(text)

        // First, split text into segments of special tokens and regular text
        let segments = splitWithSpecialTokens(normalizedText)
        var allIds: [Int32] = []

        for segment in segments {
            if let id = vocab[segment] {
                // This is a special token - add directly
                allIds.append(Int32(id))
            } else {
                // Regular text - apply BPE tokenization
                let ids = encodeRegularText(segment)
                allIds.append(contentsOf: ids)
            }
        }

        return allIds
    }

    /// Split text into segments, separating special tokens from regular text
    private func splitWithSpecialTokens(_ text: String) -> [String] {
        if specialTokens.isEmpty { return [text] }

        var segments: [String] = []
        var remaining = text

        // Simple optimization: If no special token start char (<) is in string, return early
        if !remaining.contains("<") {
             return [remaining]
        }

        while !remaining.isEmpty {
            // Check for prefix match (fastest)
            var matchedSpecial: String? = nil
            for special in specialTokens {
                if remaining.hasPrefix(special) {
                    matchedSpecial = special
                    break
                }
            }

            if let special = matchedSpecial {
                segments.append(special)
                remaining = String(remaining.dropFirst(special.count))
                continue
            }

            // No prefix match, find next occurrence of ANY special token
            if let range = remaining.range(of: "<") {
                let startDist = remaining.distance(from: remaining.startIndex, to: range.lowerBound)

                if startDist == 0 {
                    let nextIndex = remaining.index(after: remaining.startIndex)
                    if let nextRange = remaining.range(of: "<", range: nextIndex..<remaining.endIndex) {
                        let chunk = String(remaining[..<nextRange.lowerBound])
                        segments.append(chunk)
                        remaining = String(remaining[nextRange.lowerBound...])
                    } else {
                        segments.append(remaining)
                        remaining = ""
                    }
                } else {
                    // Take everything before the '<'
                    let chunk = String(remaining[..<range.lowerBound])
                    segments.append(chunk)
                    remaining = String(remaining[range.lowerBound...])
                }
            } else {
                // No more '<', rest is regular
                segments.append(remaining)
                remaining = ""
            }
        }

        return segments
    }

    // Cached regex for splitting text
    private static let splitRegex = try! NSRegularExpression(pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)

    // GPT-2/Qwen byte-to-unicode mapping used for byte-level BPE
    private static let byteToUnicode: [UInt8: Character] = {
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

        var map: [UInt8: Character] = [:]
        map.reserveCapacity(256)
        for (b, c) in zip(bs, cs) {
            if let scalar = UnicodeScalar(c) {
                map[b] = Character(scalar)
            }
        }
        return map
    }()

    /// Encode regular text (non-special tokens) using BPE
    private func encodeRegularText(_ text: String) -> [Int32] {
        var allIds: [Int32] = []

        // Use cached regex
        let range = NSRange(text.startIndex..., in: text)
        let matches = Qwen3Tokenizer.splitRegex.matches(in: text, range: range)

        let subTokens: [String]
        if !matches.isEmpty {
            subTokens = matches.map {
                if let r = Range($0.range, in: text) {
                    return String(text[r])
                }
                return ""
            }
        } else {
            subTokens = [text] // Fallback
        }

        for token in subTokens {
            let bpeTokens = bpe(encodeByteLevelToken(token))
            for bToken in bpeTokens {
                if let id = vocab[bToken] {
                    allIds.append(Int32(id))
                } else {
                    // Byte-level fallback
                    for char in bToken {
                        if let id = vocab[String(char)] {
                            allIds.append(Int32(id))
                        }
                    }
                }
            }
        }

        return allIds
    }

    private func encodeByteLevelToken(_ token: String) -> String {
        var encoded = String()
        encoded.reserveCapacity(token.utf8.count)

        for byte in token.utf8 {
            if let mapped = Self.byteToUnicode[byte] {
                encoded.append(mapped)
            }
        }
        return encoded
    }

    public func decode(ids: [Int32]) -> String {
        if !loaded { return "" }

        var result = ""
        for id in ids {
            if let token = tokens[Int(id)] {
                result += token
            }
        }

        // Reverse specific mappings
        // Replace "Ġ" with space, "Ċ" with newline
        result = result.replacingOccurrences(of: "Ġ", with: " ")
        result = result.replacingOccurrences(of: "Ċ", with: "\n")
        return result
    }

    /// Normalize smart quotes and apostrophes to ASCII equivalents
    private func normalizeQuotes(_ text: String) -> String {
        var result = text
        result = result.replacingOccurrences(of: "\u{2019}", with: "'")
        result = result.replacingOccurrences(of: "\u{2018}", with: "'")
        result = result.replacingOccurrences(of: "\u{201B}", with: "'")
        result = result.replacingOccurrences(of: "\u{201C}", with: "\"")
        result = result.replacingOccurrences(of: "\u{201D}", with: "\"")
        result = result.replacingOccurrences(of: "\u{201F}", with: "\"")
        return result
    }

    private func bpe(_ token: String) -> [String] {
        if let cached = cache[token] { return cached }

        // Initial split into characters (grapheme clusters)
        var word: [String] = token.map { String($0) }

        // Map special characters to their vocab representation
        // GPT-2/Qwen style: space -> Ġ, newline -> Ċ
        let spaceChar = vocab["Ġ"] != nil ? "Ġ" : " "
        let newlineChar = vocab["Ċ"] != nil ? "Ċ" : "\n"
        word = word.map { char in
            if char == " " { return spaceChar }
            if char == "\n" { return newlineChar }
            return char
        }

        if word.isEmpty { return [] }

        while word.count > 1 {
            var minRank = Int.max
            var bestPairIdx: Int? = nil

            for i in 0..<word.count - 1 {
                let pair = word[i] + " " + word[i+1]

                if let rank = merges[pair] {
                    if rank < minRank {
                        minRank = rank
                        bestPairIdx = i
                    }
                }
            }

            guard let idx = bestPairIdx else {
                break
            }

            // Merge
            let mergedToken = word[idx] + word[idx+1]
            word[idx] = mergedToken
            word.remove(at: idx + 1)
        }

        // Limit cache size to prevent unbounded memory growth
        if cache.count >= maxCacheSize {
            let keysToRemove = Array(cache.keys.prefix(maxCacheSize / 2))
            for key in keysToRemove {
                cache.removeValue(forKey: key)
            }
        }
        cache[token] = word
        return word
    }
}
