import Foundation

/// Smart text chunking for memory-efficient long text generation.
/// Finds natural break points to split text while maintaining prosody.
public struct TextChunker {

    /// Default maximum words per chunk
    public static let defaultMaxWords = 35

    /// Minimum words to consider for a chunk (avoid tiny fragments)
    public static let minWords = 8

    /// Conjunctions that indicate clause boundaries (split BEFORE these)
    private static let conjunctions = [
        " and then ", " and ", " but ", " or ", " so ", " because ",
        " when ", " while ", " although ", " however ", " therefore ",
        " meanwhile ", " afterwards ", " finally ", " then "
    ]

    /// Prepositions that often start new phrases (split BEFORE these, lower priority)
    private static let phraseStarters = [
        " in the ", " on the ", " at the ", " for the ", " with the ",
        " to the ", " from the ", " into the ", " onto the "
    ]

    /// Split text into natural chunks for TTS generation.
    /// - Parameters:
    ///   - text: The input text to chunk
    ///   - maxWords: Maximum words per chunk (default 35)
    /// - Returns: Array of text chunks
    public static func chunk(_ text: String, maxWords: Int = defaultMaxWords) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        // If short enough, return as single chunk
        let wordCount = trimmed.split(separator: " ").count
        if wordCount <= maxWords {
            return [trimmed]
        }

        var chunks: [String] = []
        var remaining = trimmed

        while !remaining.isEmpty {
            let chunk = findNaturalBreak(remaining, maxWords: maxWords)
            let trimmedChunk = chunk.trimmingCharacters(in: .whitespacesAndNewlines)

            if !trimmedChunk.isEmpty {
                chunks.append(trimmedChunk)
            }

            // Remove the chunk from remaining text
            remaining = String(remaining.dropFirst(chunk.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return chunks
    }

    /// Find a natural break point within the text.
    private static func findNaturalBreak(_ text: String, maxWords: Int) -> String {
        let words = text.split(separator: " ", omittingEmptySubsequences: true)

        // If text is short enough, return all of it
        if words.count <= maxWords {
            return text
        }

        // Take up to maxWords for our search window
        let windowWords = words.prefix(maxWords)
        let window = windowWords.joined(separator: " ")

        // Priority 1: Sentence endings (. ! ?)
        if let breakPoint = findSentenceEnd(in: window) {
            let chunk = String(window.prefix(breakPoint))
            if chunk.split(separator: " ").count >= minWords {
                return chunk
            }
        }

        // Priority 2: Semicolon or colon (strong clause break)
        if let lastSemi = window.lastIndex(of: ";") {
            let chunk = String(window[...lastSemi])
            if chunk.split(separator: " ").count >= minWords {
                return chunk
            }
        }
        if let lastColon = window.lastIndex(of: ":") {
            let chunk = String(window[...lastColon])
            if chunk.split(separator: " ").count >= minWords {
                return chunk
            }
        }

        // Priority 3: Last comma (clause separator)
        if let lastComma = window.lastIndex(of: ",") {
            let chunk = String(window[...lastComma])
            if chunk.split(separator: " ").count >= minWords {
                return chunk
            }
        }

        // Priority 4: Conjunctions (split BEFORE the conjunction)
        for conjunction in conjunctions {
            if let range = window.range(of: conjunction, options: [.backwards, .caseInsensitive]) {
                let chunk = String(window[..<range.lowerBound])
                if chunk.split(separator: " ").count >= minWords {
                    return chunk
                }
            }
        }

        // Priority 5: Phrase starters (lower priority, split BEFORE)
        for starter in phraseStarters {
            if let range = window.range(of: starter, options: [.backwards, .caseInsensitive]) {
                let chunk = String(window[..<range.lowerBound])
                if chunk.split(separator: " ").count >= minWords {
                    return chunk
                }
            }
        }

        // Priority 6: Hard limit at word boundary
        return window
    }

    /// Find sentence ending position in text.
    /// Returns the position after the sentence-ending punctuation.
    private static func findSentenceEnd(in text: String) -> Int? {
        var lastEnd: Int? = nil
        let minChunkLength = minWords * 4  // Rough character estimate

        for (index, char) in text.enumerated() {
            if char == "." || char == "!" || char == "?" {
                let nextIndex = text.index(text.startIndex, offsetBy: index + 1, limitedBy: text.endIndex)
                if nextIndex == text.endIndex || (nextIndex != nil && text[nextIndex!].isWhitespace) {
                    if index >= minChunkLength {
                        lastEnd = index + 1
                    }
                }
            }
        }

        return lastEnd
    }

    /// Estimate the number of audio tokens needed for text.
    /// At 12Hz, approximately 12 tokens = 1 second of audio.
    /// Average speech is ~150 words per minute = 2.5 words per second.
    /// So roughly 5 tokens per word.
    public static func estimateTokens(for text: String) -> Int {
        let wordCount = text.split(separator: " ").count
        return max(50, wordCount * 5)
    }
}
