import Foundation
import MLX
import MLXNN

/// Configuration for quantized model loading (runtime settings)
public struct QuantizationSettings: Sendable {
    /// Whether to use quantized layers
    public let enabled: Bool

    /// Bits for quantization (4, 6, or 8)
    public let bits: Int

    /// Group size for quantization
    public let groupSize: Int

    /// Default settings (4-bit quantization enabled)
    public static let quantized4Bit = QuantizationSettings(enabled: true, bits: 4, groupSize: 64)

    /// 6-bit quantization (better quality)
    public static let quantized6Bit = QuantizationSettings(enabled: true, bits: 6, groupSize: 64)

    /// Full precision (no quantization)
    public static let fullPrecision = QuantizationSettings(enabled: false, bits: 4, groupSize: 64)

    public init(enabled: Bool = false, bits: Int = 4, groupSize: Int = 64) {
        self.enabled = enabled
        self.bits = bits
        self.groupSize = groupSize
    }

    /// Create settings from the JSON-loaded QuantizationConfig
    public init(from config: QuantizationConfig?) {
        if let config = config, let bits = config.bits {
            self.enabled = true
            self.bits = bits
            self.groupSize = config.group_size ?? 64
        } else {
            self.enabled = false
            self.bits = 4
            self.groupSize = 64
        }
    }
}

/// Factory for creating Linear or QuantizedLinear layers based on settings
public enum QuantizedLayerFactory {

    /// Create a Linear or QuantizedLinear layer based on quantization settings
    public static func linear(
        _ inputDims: Int,
        _ outputDims: Int,
        bias: Bool = true,
        settings: QuantizationSettings
    ) -> Linear {
        if settings.enabled {
            return QuantizedLinear(
                inputDims,
                outputDims,
                bias: bias,
                groupSize: settings.groupSize,
                bits: settings.bits
            )
        } else {
            return Linear(inputDims, outputDims, bias: bias)
        }
    }

    /// Create a Linear layer (convenience for non-quantized)
    public static func linear(
        _ inputDims: Int,
        _ outputDims: Int,
        bias: Bool = true
    ) -> Linear {
        Linear(inputDims, outputDims, bias: bias)
    }
}
