import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - ELU Activation

nonisolated public class ELUActivation: Module, UnaryLayer {
    let alpha: Float

    public init(alpha: Float = 1.0) {
        self.alpha = alpha
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ELU: max(0, x) + min(0, alpha * (exp(x) - 1))
        maximum(x, 0) + minimum(alpha * (exp(x) - 1), 0)
    }
}

// MARK: - MimiConv1d (Causal Conv1d matching HuggingFace MimiConv1d)

nonisolated public class MimiConv1d: Module {
    let conv: Conv1d
    let paddingLeft: Int
    let paddingRight: Int
    let stride: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        let effectiveKernel = (kernelSize - 1) * dilation + 1

        // Causal padding: all padding on the left
        self.paddingLeft = effectiveKernel - stride
        self.paddingRight = 0

        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
        super.init()
    }

    private func getExtraPadding(length: Int) -> Int {
        let effectiveKernel = paddingLeft + stride
        let nFrames = Float(length - effectiveKernel + paddingLeft) / Float(stride) + 1
        let idealLength = (Int(ceil(nFrames)) - 1) * stride + (effectiveKernel - paddingLeft)
        return max(0, idealLength - length)
    }

    /// Input: [B, C, T] (channels-first), Output: [B, C_out, T']
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let extraPadding = getExtraPadding(length: x.shape[2])

        // Pad time dimension: (paddingLeft, paddingRight + extraPadding)
        var padded = MLX.padded(x, widths: [
            .init((0, 0)),
            .init((0, 0)),
            .init((paddingLeft, paddingRight + extraPadding))
        ])

        // MLX Conv1d expects [B, T, C] — transpose from [B, C, T]
        padded = padded.transposed(0, 2, 1)
        var result = conv(padded)
        // Transpose back to [B, C, T]
        result = result.transposed(0, 2, 1)
        return result
    }
}

// MARK: - MimiResnetBlock

nonisolated public class MimiResnetBlock: Module {
    let block: [Module]

    public init(dim: Int, dilations: [Int] = [1, 1], compress: Int = 2) {
        let hiddenDim = dim / compress
        self.block = [
            ELUActivation(),
            MimiConv1d(inChannels: dim, outChannels: hiddenDim, kernelSize: 3, dilation: dilations[0]),
            ELUActivation(),
            MimiConv1d(inChannels: hiddenDim, outChannels: dim, kernelSize: 1, dilation: dilations[1])
        ]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let elu = layer as? ELUActivation {
                h = elu(h)
            } else if let conv = layer as? MimiConv1d {
                h = conv(h)
            }
        }
        return x + h
    }
}

// MARK: - MimiSEANetEncoder

nonisolated public class MimiSEANetEncoder: Module {
    let layers: [Module]

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        var layerList: [Module] = []

        let numFilters = config.num_filters     // 64
        let ratios = config.upsampling_ratios   // [8, 6, 5, 4] — for encoder these are downsample ratios
        let numResLayers = config.num_residual_layers  // 1

        // Layer 0: initial conv
        layerList.append(MimiConv1d(
            inChannels: config.audio_channels,
            outChannels: numFilters,
            kernelSize: config.kernel_size
        ))

        // For each ratio: ResnetBlock(s) + ELU + downsampling Conv1d
        var currentChannels = numFilters
        for (i, ratio) in ratios.reversed().enumerated() {
            let mult = Int(pow(2.0, Double(i + 1)))
            let outChannels = numFilters * mult

            // Residual blocks
            for j in 0..<numResLayers {
                let dilation = Int(pow(Double(config.dilation_growth_rate), Double(j)))
                layerList.append(MimiResnetBlock(dim: currentChannels, dilations: [dilation, 1]))
            }

            // ELU activation
            layerList.append(ELUActivation())

            // Downsampling conv: kernel_size = 2*ratio, stride = ratio
            layerList.append(MimiConv1d(
                inChannels: currentChannels,
                outChannels: outChannels,
                kernelSize: 2 * ratio,
                stride: ratio
            ))

            currentChannels = outChannels
        }

        // Final ELU + conv to hidden_size
        layerList.append(ELUActivation())
        layerList.append(MimiConv1d(
            inChannels: currentChannels,
            outChannels: config.hidden_size,
            kernelSize: config.last_kernel_size
        ))

        self.layers = layerList
        super.init()
    }

    /// Input: [B, 1, L] -> Output: [B, hidden_size, L/960]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let conv = layer as? MimiConv1d {
                h = conv(h)
            } else if let resnet = layer as? MimiResnetBlock {
                h = resnet(h)
            } else if let elu = layer as? ELUActivation {
                h = elu(h)
            }
        }
        return h
    }
}

// MARK: - Encoder Attention

nonisolated public class EncoderAttention: Module {
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear

    public init(config: Qwen3TTSTokenizerEncoderConfig, layerIdx: Int) {
        self.headDim = config.head_dim
        self.numHeads = config.num_attention_heads
        self.numKVHeads = config.num_key_value_heads
        self.scale = pow(Float(headDim), -0.5)

        let quantization = config.quantizationSettings
        let hiddenSize = config.hidden_size
        self.qProj = QuantizedLayerFactory.linear(hiddenSize, numHeads * headDim, bias: false, settings: quantization)
        self.kProj = QuantizedLayerFactory.linear(hiddenSize, numKVHeads * headDim, bias: false, settings: quantization)
        self.vProj = QuantizedLayerFactory.linear(hiddenSize, numKVHeads * headDim, bias: false, settings: quantization)
        self.oProj = QuantizedLayerFactory.linear(numHeads * headDim, hiddenSize, bias: false, settings: quantization)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

        var q = qProj(x).reshaped([batch, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([batch, seqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([batch, seqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)

        let (cos, sin) = positionEmbeddings
        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

        var output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped([batch, seqLen, -1])
        return oProj(output)
    }
}

// MARK: - Encoder MLP (fc1/fc2 with GELU, NOT gate/up/down SiLU)

nonisolated public class EncoderMLP: Module {
    let fc1: Linear
    let fc2: Linear

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        let quantization = config.quantizationSettings
        self.fc1 = QuantizedLayerFactory.linear(config.hidden_size, config.intermediate_size, bias: true, settings: quantization)
        self.fc2 = QuantizedLayerFactory.linear(config.intermediate_size, config.hidden_size, bias: true, settings: quantization)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

// MARK: - Encoder Transformer Layer

nonisolated public class EncoderTransformerLayer: Module {
    let inputLayernorm: LayerNorm
    let postAttentionLayernorm: LayerNorm
    let selfAttn: EncoderAttention
    let mlp: EncoderMLP
    let selfAttnLayerScale: LayerScale
    let mlpLayerScale: LayerScale

    public init(config: Qwen3TTSTokenizerEncoderConfig, layerIdx: Int) {
        self.inputLayernorm = LayerNorm(dimensions: config.hidden_size, eps: config.norm_eps)
        self.postAttentionLayernorm = LayerNorm(dimensions: config.hidden_size, eps: config.norm_eps)
        self.selfAttn = EncoderAttention(config: config, layerIdx: layerIdx)
        self.mlp = EncoderMLP(config: config)
        self.selfAttnLayerScale = LayerScale(channels: config.hidden_size, initialScale: config.layer_scale_initial_scale)
        self.mlpLayerScale = LayerScale(channels: config.hidden_size, initialScale: config.layer_scale_initial_scale)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil
    ) -> MLXArray {
        var h = x
        var residual = h
        h = inputLayernorm(h)
        h = selfAttn(h, positionEmbeddings: positionEmbeddings, mask: mask)
        h = residual + selfAttnLayerScale(h)

        residual = h
        h = postAttentionLayernorm(h)
        h = mlp(h)
        h = residual + mlpLayerScale(h)

        return h
    }
}

// MARK: - Encoder Transformer

nonisolated public class EncoderTransformer: Module {
    let layers: [EncoderTransformerLayer]
    let rotaryEmb: DecoderRotaryEmbedding

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        self.layers = (0..<config.num_hidden_layers).map { i in
            EncoderTransformerLayer(config: config, layerIdx: i)
        }
        self.rotaryEmb = DecoderRotaryEmbedding(
            dim: config.head_dim,
            maxPositionEmbeddings: config.max_position_embeddings,
            base: config.rope_theta
        )
        super.init()
    }

    /// Input: [B, T, hidden_size] -> Output: [B, T, hidden_size]
    /// Non-causal (bidirectional) — no mask applied
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

        let positionIds = MLXArray(Array(0..<seqLen).map { Int32($0) }).expandedDimensions(axis: 0)
        let broadcastedIds = broadcast(positionIds, to: [batch, seqLen])
        let positionEmbeddings = rotaryEmb(x, positionIds: broadcastedIds)

        var h = x
        // No causal mask — encoder uses bidirectional attention
        for layer in layers {
            h = layer(h, positionEmbeddings: positionEmbeddings, mask: nil)
        }

        return h
    }
}

// MARK: - Encoder Downsample

nonisolated public class EncoderDownsample: Module {
    let conv: MimiConv1d

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        // Conv1d(512→512, kernel=2*compress, stride=compress) for 2× temporal compression
        self.conv = MimiConv1d(
            inChannels: config.hidden_size,
            outChannels: config.hidden_size,
            kernelSize: 2 * config.compress,
            stride: config.compress
        )
        super.init()
    }

    /// Input: [B, C, T] -> Output: [B, C, T/compress]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

// MARK: - Encoder Vector Quantization

nonisolated public class EncoderVectorQuantization: Module {
    let codebook: EuclideanCodebook

    public init(dim: Int, codebookSize: Int) {
        self.codebook = EuclideanCodebook(dim: dim, codebookSize: codebookSize)
        super.init()
    }

    /// Encode input to codebook indices and return (indices, quantized)
    /// - Parameter x: [B, T, dim]
    /// - Returns: (indices [B, T], quantized [B, T, dim])
    public func encode(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let indices = codebook.encode(x)       // [B, T]
        let quantized = codebook.decode(indices) // [B, T, dim]
        return (indices, quantized)
    }
}

// MARK: - Encoder Residual Vector Quantizer

nonisolated public class EncoderResidualVectorQuantizer: Module {
    let inputProj: Conv1d
    let outputProj: Conv1d
    let layers: [EncoderVectorQuantization]

    public init(numQuantizers: Int, dim: Int, inputDim: Int, codebookSize: Int) {
        // inputProj/outputProj are Conv1d with kernel=1, matching HF's MimiResidualVectorQuantizer
        self.inputProj = Conv1d(inputChannels: inputDim, outputChannels: dim, kernelSize: 1, bias: false)
        self.outputProj = Conv1d(inputChannels: dim, outputChannels: inputDim, kernelSize: 1, bias: false)
        self.layers = (0..<numQuantizers).map { _ in
            EncoderVectorQuantization(dim: dim, codebookSize: codebookSize)
        }
        super.init()
    }

    /// Encode input through all codebooks with residual subtraction.
    /// - Parameter x: [B, C, T] (channels-first)
    /// - Returns: codes [numQuantizers, B, T]
    public func encode(_ x: MLXArray) -> MLXArray {
        // Project input: Conv1d expects [B, T, C]
        var projected = x.transposed(0, 2, 1)   // [B, T, C]
        projected = inputProj(projected)          // [B, T, dim]

        var residual = projected
        var allCodes: [MLXArray] = []

        for layer in layers {
            let (indices, quantized) = layer.encode(residual)  // indices: [B, T]
            allCodes.append(indices)
            residual = residual - quantized
        }

        return stacked(allCodes, axis: 0)  // [numQuantizers, B, T]
    }
}

// MARK: - Encoder Split Residual Vector Quantizer

nonisolated public class EncoderSplitResidualVectorQuantizer: Module {
    let semanticResidualVectorQuantizer: EncoderResidualVectorQuantizer
    let acousticResidualVectorQuantizer: EncoderResidualVectorQuantizer

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        let nQSemantic = config.num_semantic_quantizers
        let nQAcoustic = config.num_quantizers - nQSemantic
        let codebookDim = config.vector_quantization_hidden_dimension

        self.semanticResidualVectorQuantizer = EncoderResidualVectorQuantizer(
            numQuantizers: nQSemantic,
            dim: codebookDim,
            inputDim: config.hidden_size,
            codebookSize: config.codebook_size
        )
        self.acousticResidualVectorQuantizer = EncoderResidualVectorQuantizer(
            numQuantizers: nQAcoustic,
            dim: codebookDim,
            inputDim: config.hidden_size,
            codebookSize: config.codebook_size
        )
        super.init()
    }

    /// Encode input through semantic + acoustic codebooks.
    /// - Parameter x: [B, C, T] (channels-first)
    /// - Returns: codes [B, numQuantizers, T]
    public func encode(_ x: MLXArray) -> MLXArray {
        let semanticCodes = semanticResidualVectorQuantizer.encode(x)  // [nQSemantic, B, T]
        let acousticCodes = acousticResidualVectorQuantizer.encode(x)  // [nQAcoustic, B, T]
        let allCodes = concatenated([semanticCodes, acousticCodes], axis: 0) // [numQ, B, T]
        return allCodes.transposed(1, 0, 2)  // [B, numQ, T]
    }
}

// MARK: - Qwen3TTSAudioEncoder

/// Audio encoder for ICL (in-context learning) voice cloning.
/// Encodes reference audio waveforms into quantized codes that can be
/// prepended to generation input for voice cloning.
///
/// Architecture: MimiModel encoder (SEANet pattern)
/// Forward pass: audio [B, 1, L] -> CNN encoder -> transformer -> downsample -> quantizer -> codes [B, 16, T]
nonisolated public class Qwen3TTSAudioEncoder: Module {
    private var encoder: MimiSEANetEncoder?
    private var encoderTransformer: EncoderTransformer?
    private var downsample: EncoderDownsample?
    private var quantizer: EncoderSplitResidualVectorQuantizer?
    private var validNumQuantizers: Int = 16

    public override init() {
        super.init()
    }

    /// Load encoder weights from the speech tokenizer safetensors file.
    public func loadWeights(from weightsURL: URL, configURL: URL) throws {
        // Parse encoder config from the speech tokenizer config
        let configData = try Data(contentsOf: configURL)
        let tokenizerConfig = try JSONDecoder().decode(AudioDecoderConfig.self, from: configData)

        let encoderConfig: Qwen3TTSTokenizerEncoderConfig
        if let ec = tokenizerConfig.encoder_config {
            encoderConfig = ec
        } else {
            encoderConfig = Qwen3TTSTokenizerEncoderConfig()
        }

        self.validNumQuantizers = tokenizerConfig.encoder_valid_num_quantizers ?? 16

        // Create modules
        let enc = MimiSEANetEncoder(config: encoderConfig)
        let transformer = EncoderTransformer(config: encoderConfig)
        let ds = EncoderDownsample(config: encoderConfig)
        let quant = EncoderSplitResidualVectorQuantizer(config: encoderConfig)

        // Evaluate random init weights first
        eval(enc.parameters())
        eval(transformer.parameters())
        eval(ds.parameters())
        eval(quant.parameters())
        Memory.clearCache()

        // Load and sanitize weights
        let allWeights = try MLX.loadArrays(url: weightsURL, stream: .cpu)
        let sanitized = Qwen3TTSAudioEncoder.sanitizeEncoderWeights(allWeights)

        // Load into a container module that holds all submodules
        self.encoder = enc
        self.encoderTransformer = transformer
        self.downsample = ds
        self.quantizer = quant

        let parameters = ModuleParameters.unflattened(sanitized)
        try self.update(parameters: parameters, verify: .noUnusedKeys)

        Memory.clearCache()
    }

    /// Encode audio waveform into quantized codes.
    /// - Parameter audio: Raw audio tensor [B, samples] or [B, 1, samples]
    /// - Returns: Quantized codes [B, num_quantizers, time]
    public func encode(_ audio: MLXArray) -> MLXArray {
        callAsFunction(audio)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let encoder = encoder,
              let encoderTransformer = encoderTransformer,
              let downsample = downsample,
              let quantizer = quantizer else {
            return x
        }

        // Ensure input is [B, 1, L]
        var h = x
        if h.ndim == 2 {
            h = h.expandedDimensions(axis: 1)  // [B, L] -> [B, 1, L]
        }

        // CNN encoder: [B, 1, L] -> [B, hidden_size, L/960]
        h = encoder(h)

        // Transpose to [B, T, hidden_size] for transformer
        h = h.transposed(0, 2, 1)

        // Encoder transformer (non-causal): [B, T, hidden_size] -> [B, T, hidden_size]
        h = encoderTransformer(h)

        // Transpose back to [B, hidden_size, T] for downsample
        h = h.transposed(0, 2, 1)

        // Downsample: [B, hidden_size, T] -> [B, hidden_size, T/2]
        h = downsample(h)

        // Quantize: [B, hidden_size, T/2] -> [B, numQuantizers, T/2]
        var codes = quantizer.encode(h)

        // Keep only first validNumQuantizers
        if codes.shape[1] > validNumQuantizers {
            codes = codes[0..., 0..<validNumQuantizers, 0...]
        }

        return codes
    }

    // MARK: - Weight Sanitization

    private static func snakeToCamel(_ input: String) -> String {
        let parts = input.split(separator: "_")
        guard !parts.isEmpty else { return input }
        let first = String(parts[0])
        let rest = parts.dropFirst().map { part in
            guard let firstChar = part.first else { return String(part) }
            return String(firstChar).uppercased() + String(part.dropFirst())
        }
        return first + rest.joined()
    }

    /// Sanitize weights: extract encoder-only keys, camelCase, compute codebook embeddings, transpose convs.
    public static func sanitizeEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var codebookData: [String: [String: MLXArray]] = [:]

        for (key, value) in weights {
            var v = value

            // Only keep encoder.* keys
            guard key.hasPrefix("encoder.") else { continue }

            let workingKey = String(key.dropFirst("encoder.".count))

            // Handle codebook data (cluster_usage + embedding_sum -> embed.weight)
            if workingKey.contains("_codebook.cluster_usage") || workingKey.contains("_codebook.embedding_sum") {
                let parts = workingKey.components(separatedBy: "._codebook.")
                if parts.count == 2 {
                    let basePath = parts[0]
                    if codebookData[basePath] == nil {
                        codebookData[basePath] = [:]
                    }
                    if workingKey.contains("cluster_usage") {
                        codebookData[basePath]?["cluster_usage"] = v
                    } else {
                        codebookData[basePath]?["embedding_sum"] = v
                    }
                }
                continue
            }

            // Convert snake_case to camelCase
            let components = workingKey.components(separatedBy: ".")
            let camelComponents = components.map { component -> String in
                if Int(component) != nil { return component }
                return snakeToCamel(component)
            }
            let newKey = camelComponents.joined(separator: ".")

            // Transpose conv weights: safetensors stores [out, in, k], MLX Conv1d wants [out, k, in]
            if workingKey.contains("conv") && workingKey.hasSuffix(".weight") && v.ndim == 3 {
                v = v.transposed(0, 2, 1)
            } else if workingKey.contains("_proj") && workingKey.hasSuffix(".weight") && v.ndim == 3 {
                // Conv1d proj weights (inputProj, outputProj)
                v = v.transposed(0, 2, 1)
            }

            sanitized[newKey] = v
        }

        // Compute codebook embeddings from cluster_usage + embedding_sum
        let eps: Float = 1e-5
        for (basePath, data) in codebookData {
            if let clusterUsage = data["cluster_usage"],
               let embeddingSum = data["embedding_sum"] {
                let usage = clip(clusterUsage, min: eps, max: Float.greatestFiniteMagnitude)
                let embedding = embeddingSum / usage.expandedDimensions(axis: -1)

                let components = basePath.components(separatedBy: ".")
                let camelComponents = components.map { component -> String in
                    if Int(component) != nil { return component }
                    return snakeToCamel(component)
                }
                let camelPath = camelComponents.joined(separator: ".")

                let newKey = "\(camelPath).codebook.embed.weight"
                sanitized[newKey] = embedding
            }
        }

        return sanitized
    }
}
