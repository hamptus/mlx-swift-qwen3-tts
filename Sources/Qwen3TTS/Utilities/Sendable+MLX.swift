import Foundation
import MLX

extension Qwen3Talker: @unchecked Sendable {}
extension Qwen3Tokenizer: @unchecked Sendable {}
extension AudioDecoder: @unchecked Sendable {}
extension SpeakerEncoder: @unchecked Sendable {}

public struct MLXArrayBox: @unchecked Sendable {
    public let array: MLXArray
}
