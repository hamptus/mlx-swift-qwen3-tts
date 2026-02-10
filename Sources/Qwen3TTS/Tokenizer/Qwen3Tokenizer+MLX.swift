import Foundation
import MLX

extension Qwen3Tokenizer {
    public func encode(text: String) -> MLXArray {
        let ids: [Int32] = self.encode(text: text)
        if ids.isEmpty {
            return MLXArray.zeros([1, 0], dtype: .int32)
        }
        return MLXArray(ids).expandedDimensions(axis: 0)
    }

    public func decode(ids: MLXArray) -> String {
        let indices = ids.reshaped([-1]).asArray(Int32.self)
        return self.decode(ids: indices)
    }
}
