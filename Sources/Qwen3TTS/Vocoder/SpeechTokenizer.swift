import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXFast

// MARK: - Configuration

public struct Qwen3TTSTokenizerDecoderConfig: Codable, Sendable {
    public var attention_bias: Bool = false
    public var attention_dropout: Float = 0.0
    public var latent_dim: Int = 1024
    public var codebook_dim: Int = 512
    public var codebook_size: Int = 2048
    public var decoder_dim: Int = 1536
    public var hidden_act: String = "silu"
    public var hidden_size: Int = 512
    public var intermediate_size: Int = 1024
    public var layer_scale_initial_scale: Float = 0.01
    public var max_position_embeddings: Int = 8000
    public var head_dim: Int = 64
    public var num_attention_heads: Int = 16
    public var num_hidden_layers: Int = 8
    public var num_key_value_heads: Int = 16
    public var num_quantizers: Int = 16
    public var num_semantic_quantizers: Int = 1
    public var rms_norm_eps: Float = 1e-5
    public var rope_theta: Float = 10000.0
    public var semantic_codebook_size: Int = 4096
    public var sliding_window: Int = 72
    public var upsample_rates: [Int] = [8, 5, 4, 3]
    public var upsampling_ratios: [Int] = [2, 2]
    public var vector_quantization_hidden_dimension: Int = 512
    public var quantization: QuantizationConfig?

    public init() {}

    public var quantizationSettings: QuantizationSettings {
        QuantizationSettings(from: quantization)
    }
}

public struct Qwen3TTSTokenizerConfig: Codable, Sendable {
    public var decoder_config: Qwen3TTSTokenizerDecoderConfig?
    public var encoder_valid_num_quantizers: Int = 16
    public var input_sample_rate: Int = 24000
    public var output_sample_rate: Int = 24000
    public var decode_upsample_rate: Int = 1920
    public var encode_downsample_rate: Int = 1920

    public init() {
        self.decoder_config = Qwen3TTSTokenizerDecoderConfig()
    }
}

// MARK: - SnakeBeta Activation

public class SnakeBeta: Module, UnaryLayer {
    let channels: Int
    var alpha: MLXArray
    var beta: MLXArray
    let eps: Float = 1e-9

    public init(channels: Int) {
        self.channels = channels
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaExp = exp(alpha).expandedDimensions(axes: [0, 2])
        let betaExp = exp(beta).expandedDimensions(axes: [0, 2])
        return x + (1.0 / (betaExp + eps)) * pow(sin(x * alphaExp), 2)
    }
}

// MARK: - CausalConv1d

public class CausalConv1d: Module {
    let conv: Conv1d
    let groups: Int
    let inChannels: Int
    let outChannels: Int
    let stride: Int
    let kernelSizeVal: Int
    let kernelSize: Int
    let dilation: Int
    let padding: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.groups = groups
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.stride = stride
        self.kernelSizeVal = kernelSize
        self.kernelSize = (kernelSize - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernelSize - stride

        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups
        )
        super.init()
    }

    private func getExtraPadding(length: Int) -> Int {
        let nFrames = Float(length - kernelSize + padding) / Float(stride) + 1
        let idealLength = (Int(ceil(nFrames)) - 1) * stride + (kernelSize - padding)
        return idealLength - length
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let extraPadding = getExtraPadding(length: x.shape[2])

        var padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((padding, extraPadding))])

        padded = padded.transposed(0, 2, 1)
        var result = conv(padded)
        result = result.transposed(0, 2, 1)
        return result
    }
}

// MARK: - CausalTransposeConv1d

public class CausalTransposeConv1d: Module {
    let conv: ConvTransposed1d
    let trimRight: Int

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.conv = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0
        )
        self.trimRight = kernelSize - stride
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x.transposed(0, 2, 1)
        result = conv(result)
        result = result.transposed(0, 2, 1)

        if trimRight > 0 {
            let timeLen = result.shape[2]
            let endIdx = timeLen - trimRight
            if endIdx > 0 {
                result = result[0..., 0..., 0..<endIdx]
            }
        }
        return result
    }
}

// MARK: - ConvNeXtBlock

public class ConvNeXtBlock: Module {
    let dwconv: CausalConv1d
    let norm: LayerNorm
    let pwconv1: Linear
    let pwconv2: Linear
    var gamma: MLXArray

    public init(dim: Int, quantization: QuantizationSettings = .fullPrecision) {
        self.dwconv = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, groups: dim)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        self.pwconv1 = QuantizedLayerFactory.linear(dim, 4 * dim, settings: quantization)
        self.pwconv2 = QuantizedLayerFactory.linear(4 * dim, dim, settings: quantization)
        self.gamma = MLXArray.ones([dim]) * 1e-6
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = dwconv(x)
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        h = gamma * h
        h = h.transposed(0, 2, 1)
        return residual + h
    }
}

// MARK: - DecoderRMSNorm

public class DecoderRMSNorm: Module, UnaryLayer {
    var weight: MLXArray
    let eps: Float

    public init(hiddenSize: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([hiddenSize])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = MLX.mean(pow(xFloat, 2), axis: -1, keepDims: true)
        let xNormed = xFloat * rsqrt(variance + eps)
        return (weight * xNormed).asType(x.dtype)
    }
}

// MARK: - LayerScale

public class LayerScale: Module, UnaryLayer {
    var scale: MLXArray

    public init(channels: Int, initialScale: Float = 0.01) {
        self.scale = MLXArray.ones([channels]) * initialScale
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return scale * x
    }
}

// MARK: - DecoderRotaryEmbedding

public class DecoderRotaryEmbedding: Module {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let invFreq: MLXArray

    public init(dim: Int, maxPositionEmbeddings: Int = 8000, base: Float = 10000.0) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base

        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = 1.0 / pow(base, indices / Float(dim))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let invFreqExpanded = invFreq.expandedDimensions(axes: [0, 2]).asType(.float32)
        let pos = positionIds.expandedDimensions(axis: 1).asType(.float32)
        let freqs = matmul(invFreqExpanded, pos).transposed(0, 2, 1)
        let emb = concatenated([freqs, freqs], axis: -1)
        let cosEmb = cos(emb).asType(x.dtype)
        let sinEmb = sin(emb).asType(x.dtype)
        return (cosEmb, sinEmb)
    }
}

// MARK: - Helper Functions

func speechTokenizerRotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.shape[x.ndim - 1] / 2
    let x1 = x[0..., 0..., 0..., 0..<halfDim]
    let x2 = x[0..., 0..., 0..., halfDim...]
    return concatenated([-x2, x1], axis: -1)
}

func applyRotaryPosEmb(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let cosExpanded = cos.expandedDimensions(axis: 1)
    let sinExpanded = sin.expandedDimensions(axis: 1)
    let qEmbed = (q * cosExpanded) + (speechTokenizerRotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (speechTokenizerRotateHalf(k) * sinExpanded)
    return (qEmbed, kEmbed)
}

// MARK: - DecoderAttention

public class DecoderAttention: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let layerIdx: Int
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear

    public init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.headDim = config.head_dim
        self.numHeads = config.num_attention_heads
        self.numKVHeads = config.num_key_value_heads
        self.scale = pow(Float(headDim), -0.5)

        let quantization = config.quantizationSettings
        self.qProj = QuantizedLayerFactory.linear(config.hidden_size, numHeads * headDim, bias: config.attention_bias, settings: quantization)
        self.kProj = QuantizedLayerFactory.linear(config.hidden_size, numKVHeads * headDim, bias: config.attention_bias, settings: quantization)
        self.vProj = QuantizedLayerFactory.linear(config.hidden_size, numKVHeads * headDim, bias: config.attention_bias, settings: quantization)
        self.oProj = QuantizedLayerFactory.linear(numHeads * headDim, config.hidden_size, bias: config.attention_bias, settings: quantization)
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

// MARK: - DecoderMLP

public class DecoderMLP: Module {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear

    public init(config: Qwen3TTSTokenizerDecoderConfig) {
        let quantization = config.quantizationSettings
        self.gateProj = QuantizedLayerFactory.linear(config.hidden_size, config.intermediate_size, bias: false, settings: quantization)
        self.upProj = QuantizedLayerFactory.linear(config.hidden_size, config.intermediate_size, bias: false, settings: quantization)
        self.downProj = QuantizedLayerFactory.linear(config.intermediate_size, config.hidden_size, bias: false, settings: quantization)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - DecoderTransformerLayer

public class DecoderTransformerLayer: Module {
    let selfAttn: DecoderAttention
    let mlp: DecoderMLP
    let inputLayernorm: DecoderRMSNorm
    let postAttentionLayernorm: DecoderRMSNorm
    let selfAttnLayerScale: LayerScale
    let mlpLayerScale: LayerScale

    public init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self.selfAttn = DecoderAttention(config: config, layerIdx: layerIdx)
        self.mlp = DecoderMLP(config: config)
        self.inputLayernorm = DecoderRMSNorm(hiddenSize: config.hidden_size, eps: config.rms_norm_eps)
        self.postAttentionLayernorm = DecoderRMSNorm(hiddenSize: config.hidden_size, eps: config.rms_norm_eps)
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

// MARK: - DecoderTransformer

public class DecoderTransformer: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let layers: [DecoderTransformerLayer]
    let norm: DecoderRMSNorm
    let rotaryEmb: DecoderRotaryEmbedding
    let inputProj: Linear
    let outputProj: Linear

    public init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self.layers = (0..<config.num_hidden_layers).map { i in
            DecoderTransformerLayer(config: config, layerIdx: i)
        }
        self.norm = DecoderRMSNorm(hiddenSize: config.hidden_size, eps: config.rms_norm_eps)
        self.rotaryEmb = DecoderRotaryEmbedding(
            dim: config.head_dim,
            maxPositionEmbeddings: config.max_position_embeddings,
            base: config.rope_theta
        )
        let quantization = config.quantizationSettings
        self.inputProj = QuantizedLayerFactory.linear(config.latent_dim, config.hidden_size, settings: quantization)
        self.outputProj = QuantizedLayerFactory.linear(config.hidden_size, config.latent_dim, settings: quantization)
        super.init()
    }

    public func callAsFunction(_ inputsEmbeds: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (batch, seqLen, _) = (inputsEmbeds.shape[0], inputsEmbeds.shape[1], inputsEmbeds.shape[2])

        var x = inputProj(inputsEmbeds)

        let positionIds = MLXArray(Array(0..<seqLen).map { Int32($0) }).expandedDimensions(axis: 0)
        let broadcastedIds = broadcast(positionIds, to: [batch, seqLen])

        let positionEmbeddings = rotaryEmb(x, positionIds: broadcastedIds)

        var actualMask = mask
        if seqLen > 1 && mask == nil {
            actualMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(x.dtype)
        }

        for layer in layers {
            x = layer(x, positionEmbeddings: positionEmbeddings, mask: actualMask)
        }

        x = norm(x)
        x = outputProj(x)

        return x
    }
}

// MARK: - EuclideanCodebook

public class EuclideanCodebook: Module {
    let dim: Int
    let codebookSize: Int
    let embed: Embedding

    public init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.codebookSize = codebookSize
        self.embed = Embedding(embeddingCount: codebookSize, dimensions: dim)
        super.init()
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        return embed(codes)
    }
}

// MARK: - VectorQuantization

public class VectorQuantization: Module {
    let projectOut: Linear?
    let codebook: EuclideanCodebook
    let codebookSize: Int

    public init(dim: Int, codebookSize: Int, codebookDim: Int? = nil, quantization: QuantizationSettings = .fullPrecision) {
        let actualCodebookDim = codebookDim ?? dim
        let requiresProjection = actualCodebookDim != dim

        if requiresProjection {
            self.projectOut = Linear(actualCodebookDim, dim)
        } else {
            self.projectOut = nil
        }

        self.codebook = EuclideanCodebook(dim: actualCodebookDim, codebookSize: codebookSize)
        self.codebookSize = codebookSize
        super.init()
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = codebook.decode(codes)
        if let proj = projectOut {
            quantized = proj(quantized)
        }
        quantized = quantized.transposed(0, 2, 1)
        return quantized
    }
}

// MARK: - ResidualVectorQuantization

public class ResidualVectorQuantization: Module {
    let layers: [VectorQuantization]

    public init(numQuantizers: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil, quantization: QuantizationSettings = .fullPrecision) {
        self.layers = (0..<numQuantizers).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim, quantization: quantization)
        }
        super.init()
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        guard !layers.isEmpty else {
            print("CRASH AVOIDED [RVQ.decode]: layers is empty!")
            return MLXArray.zeros([codes.shape[1], 1, codes.shape[2]])
        }
        var quantized = MLXArray.zeros([codes.shape[1], layers[0].codebook.dim, codes.shape[2]])

        let numIter = min(codes.shape[0], layers.count)
        if codes.shape[0] != layers.count {
            print("CRASH AVOIDED [RVQ.decode]: codes.shape[0]=\(codes.shape[0]) != layers.count=\(layers.count), using min=\(numIter)")
        }
        for idx in 0..<numIter {
            let layerCodes = codes[idx]
            quantized = quantized + layers[idx].decode(layerCodes)
        }
        return quantized
    }
}

// MARK: - ResidualVectorQuantizer

public class ResidualVectorQuantizer: Module {
    let nQ: Int
    let dimension: Int
    let inputDimension: Int
    let outputDimension: Int
    let bins: Int

    let inputProj: Conv1d?
    let outputProj: Conv1d?
    let vq: ResidualVectorQuantization

    public init(
        dimension: Int = 128,
        inputDimension: Int? = nil,
        outputDimension: Int? = nil,
        nQ: Int = 8,
        bins: Int = 1024,
        forceProjection: Bool = false,
        quantization: QuantizationSettings = .fullPrecision
    ) {
        self.nQ = nQ
        self.dimension = dimension
        self.inputDimension = inputDimension ?? dimension
        self.outputDimension = outputDimension ?? dimension
        self.bins = bins

        if self.inputDimension == dimension && !forceProjection {
            self.inputProj = nil
        } else {
            self.inputProj = Conv1d(inputChannels: self.inputDimension, outputChannels: dimension, kernelSize: 1, bias: false)
        }

        if self.outputDimension == dimension && !forceProjection {
            self.outputProj = nil
        } else {
            self.outputProj = Conv1d(inputChannels: dimension, outputChannels: self.outputDimension, kernelSize: 1, bias: false)
        }

        self.vq = ResidualVectorQuantization(numQuantizers: nQ, dim: dimension, codebookSize: bins, quantization: quantization)
        super.init()
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        let codesTransposed = codes.transposed(1, 0, 2)
        var quantized = vq.decode(codesTransposed)

        if let proj = outputProj {
            quantized = quantized.transposed(0, 2, 1)
            quantized = proj(quantized)
            quantized = quantized.transposed(0, 2, 1)
        }
        return quantized
    }
}

// MARK: - SplitResidualVectorQuantizer

public class SplitResidualVectorQuantizer: Module {
    let nQSemantic: Int
    let nQAcoustic: Int

    let rvqFirst: ResidualVectorQuantizer
    let rvqRest: ResidualVectorQuantizer

    public init(
        nQ: Int = 8,
        nQSemantic: Int = 1,
        dimension: Int = 128,
        inputDimension: Int? = nil,
        outputDimension: Int? = nil,
        bins: Int = 1024,
        quantization: QuantizationSettings = .fullPrecision
    ) {
        self.nQSemantic = nQSemantic
        self.nQAcoustic = nQ - nQSemantic

        self.rvqFirst = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQSemantic,
            bins: bins,
            forceProjection: true,
            quantization: quantization
        )
        self.rvqRest = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQ - nQSemantic,
            bins: bins,
            forceProjection: true,
            quantization: quantization
        )
        super.init()
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = rvqFirst.decode(codes[0..., 0..<nQSemantic, 0...])

        if codes.shape[1] > nQSemantic {
            quantized = quantized + rvqRest.decode(codes[0..., nQSemantic..., 0...])
        }
        return quantized
    }
}

// MARK: - Decoder Components

public class DecoderResidualUnit: Module {
    let act1: SnakeBeta
    let conv1: CausalConv1d
    let act2: SnakeBeta
    let conv2: CausalConv1d

    public init(dim: Int, dilation: Int = 1) {
        self.act1 = SnakeBeta(channels: dim)
        self.conv1 = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, dilation: dilation)
        self.act2 = SnakeBeta(channels: dim)
        self.conv2 = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = act1(x)
        h = conv1(h)
        h = act2(h)
        h = conv2(h)
        return h + residual
    }
}

public class DecoderBlockUpsample: Module {
    let conv: ConvTransposed1d
    let trimRight: Int

    public init(inDim: Int, outDim: Int, upsampleRate: Int) {
        let kernelSize = 2 * upsampleRate
        self.conv = ConvTransposed1d(
            inputChannels: inDim,
            outputChannels: outDim,
            kernelSize: kernelSize,
            stride: upsampleRate,
            padding: 0
        )
        self.trimRight = kernelSize - upsampleRate
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x.transposed(0, 2, 1)
        result = conv(result)
        result = result.transposed(0, 2, 1)

        if trimRight > 0 {
            let timeLen = result.shape[2]
            let endIdx = timeLen - trimRight
            if endIdx > 0 {
                result = result[0..., 0..., 0..<endIdx]
            }
        }
        return result
    }
}

public class DecoderBlock: Module {
    let block: [Module]

    public init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        let inDim = config.decoder_dim / Int(pow(2.0, Double(layerIdx)))
        let outDim = config.decoder_dim / Int(pow(2.0, Double(layerIdx + 1)))
        let upsampleRate = config.upsample_rates[layerIdx]

        self.block = [
            SnakeBeta(channels: inDim),
            DecoderBlockUpsample(inDim: inDim, outDim: outDim, upsampleRate: upsampleRate),
            DecoderResidualUnit(dim: outDim, dilation: 1),
            DecoderResidualUnit(dim: outDim, dilation: 3),
            DecoderResidualUnit(dim: outDim, dilation: 9),
        ]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let snake = layer as? SnakeBeta {
                h = snake(h)
            } else if let upsample = layer as? DecoderBlockUpsample {
                h = upsample(h)
            } else if let residual = layer as? DecoderResidualUnit {
                h = residual(h)
            }
        }
        return h
    }
}

public class DecoderInitialConv: Module {
    let conv: Conv1d
    let kernelSize: Int

    public init(latentDim: Int, decoderDim: Int, kernelSize: Int = 7) {
        self.conv = Conv1d(inputChannels: latentDim, outputChannels: decoderDim, kernelSize: kernelSize, padding: 0)
        self.kernelSize = kernelSize
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((kernelSize - 1, 0))])
        padded = padded.transposed(0, 2, 1)
        var result = conv(padded)
        result = result.transposed(0, 2, 1)
        return result
    }
}

public class DecoderOutputSnake: Module, UnaryLayer {
    var alpha: MLXArray
    var beta: MLXArray
    let eps: Float = 1e-9

    public init(channels: Int) {
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaExp = exp(alpha).reshaped([1, -1, 1])
        let betaExp = exp(beta).reshaped([1, -1, 1])
        return x + (1.0 / (betaExp + eps)) * pow(sin(x * alphaExp), 2)
    }
}

public class DecoderOutputConv: Module {
    let conv: Conv1d
    let kernelSize: Int

    public init(channels: Int, kernelSize: Int = 7) {
        self.conv = Conv1d(inputChannels: channels, outputChannels: 1, kernelSize: kernelSize, padding: 0)
        self.kernelSize = kernelSize
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((kernelSize - 1, 0))])
        padded = padded.transposed(0, 2, 1)
        var result = conv(padded)
        result = result.transposed(0, 2, 1)
        return result
    }
}

// MARK: - Qwen3TTSSpeechTokenizerDecoder

public class Qwen3TTSSpeechTokenizerDecoder: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let totalUpsample: Int

    let preTransformer: DecoderTransformer
    let quantizer: SplitResidualVectorQuantizer
    let preConv: CausalConv1d
    let upsample: [[Module]]
    let decoder: [Module]

    public init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        let quantization = config.quantizationSettings

        let allRates = config.upsample_rates + config.upsampling_ratios
        self.totalUpsample = allRates.reduce(1, *)

        self.preTransformer = DecoderTransformer(config: config)

        self.quantizer = SplitResidualVectorQuantizer(
            nQ: config.num_quantizers,
            nQSemantic: config.num_semantic_quantizers,
            dimension: config.codebook_dim / 2,
            inputDimension: config.codebook_dim,
            outputDimension: config.codebook_dim,
            bins: config.codebook_size,
            quantization: quantization
        )

        self.preConv = CausalConv1d(
            inChannels: config.codebook_dim,
            outChannels: config.latent_dim,
            kernelSize: 3
        )

        self.upsample = config.upsampling_ratios.map { factor in
            [
                CausalTransposeConv1d(inChannels: config.latent_dim, outChannels: config.latent_dim, kernelSize: factor, stride: factor),
                ConvNeXtBlock(dim: config.latent_dim, quantization: quantization)
            ]
        }

        let outputDim = config.decoder_dim / Int(pow(2.0, Double(config.upsample_rates.count)))
        var decoderLayers: [Module] = [DecoderInitialConv(latentDim: config.latent_dim, decoderDim: config.decoder_dim, kernelSize: 7)]
        for i in 0..<config.upsample_rates.count {
            decoderLayers.append(DecoderBlock(config: config, layerIdx: i))
        }
        decoderLayers.append(DecoderOutputSnake(channels: outputDim))
        decoderLayers.append(DecoderOutputConv(channels: outputDim, kernelSize: 7))
        self.decoder = decoderLayers

        super.init()
    }

    private var compiledDecode: ((MLXArray) -> MLXArray)?

    public func clearCompiledCache() {
        compiledDecode = nil
    }

    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        if ProcessInfo.processInfo.environment["QWEN3TTS_DISABLE_MLX_COMPILE"] == "1" {
            return decodeImpl(codes)
        }
        if compiledDecode == nil {
            compiledDecode = compile { [weak self] x in
                guard let self = self else { return x }
                return self.decodeImpl(x)
            }
        }
        return compiledDecode!(codes)
    }

    private func decodeImpl(_ codes: MLXArray) -> MLXArray {
        guard codes.shape[1] == config.num_quantizers else {
            return MLXArray.zeros([codes.shape[0], 1, 0], dtype: .float32)
        }

        var hidden = quantizer.decode(codes)
        hidden = preConv(hidden)
        hidden = hidden.transposed(0, 2, 1)
        hidden = preTransformer(hidden)
        hidden = hidden.transposed(0, 2, 1)

        for upsampleLayers in upsample {
            for layer in upsampleLayers {
                if let causalTranspose = layer as? CausalTransposeConv1d {
                    hidden = causalTranspose(hidden)
                } else if let convNeXt = layer as? ConvNeXtBlock {
                    hidden = convNeXt(hidden)
                }
            }
        }

        var wav = hidden
        for decoderLayer in decoder {
            if let initialConv = decoderLayer as? DecoderInitialConv {
                wav = initialConv(wav)
            } else if let decoderBlock = decoderLayer as? DecoderBlock {
                wav = decoderBlock(wav)
            } else if let outputSnake = decoderLayer as? DecoderOutputSnake {
                wav = outputSnake(wav)
            } else if let outputConv = decoderLayer as? DecoderOutputConv {
                wav = outputConv(wav)
            }
        }

        return clip(wav, min: -1.0, max: 1.0)
    }

    public func chunkedDecode(_ codes: MLXArray, chunkSize: Int = 100, leftContextSize: Int = 10) -> MLXArray {
        let (B, _, T) = (codes.shape[0], codes.shape[1], codes.shape[2])

        let numChunks = (T + chunkSize - 1) / chunkSize
        let paddedT = numChunks * chunkSize
        let rightPad = paddedT - T

        let paddedCodes = MLX.padded(codes, widths: [.init((0, 0)), .init((0, 0)), .init((leftContextSize, rightPad))])

        var chunkList = [MLXArray]()
        for i in 0..<numChunks {
            let start = i * chunkSize
            let end = start + chunkSize + leftContextSize
            let chunk = paddedCodes[0..., 0..., start..<end]
            chunkList.append(chunk)
        }

        let batchInput = concatenated(chunkList, axis: 0)
        let batchOutput = self(batchInput)
        let dropSamples = leftContextSize * totalUpsample
        let validOutput = batchOutput[0..., 0..., dropSamples...]

        if B == 1 {
            let flat = validOutput.reshaped([1, 1, -1])
            let targetLen = T * totalUpsample
            return flat[0..., 0..., 0..<targetLen]
        } else {
            let reshaped = validOutput.reshaped([numChunks, B, 1, validOutput.shape[2]])
            let transposed = reshaped.transposed(1, 2, 0, 3)
            let flat = transposed.reshaped([B, 1, -1])
            let targetLen = T * totalUpsample
            return flat[0..., 0..., 0..<targetLen]
        }
    }
}

// MARK: - Qwen3TTSSpeechTokenizer

public class Qwen3TTSSpeechTokenizer: Module {
    let config: Qwen3TTSTokenizerConfig
    let encoderValidNumQuantizers: Int
    let inputSampleRate: Int
    let outputSampleRate: Int
    let decodeUpsampleRate: Int

    public let decoder: Qwen3TTSSpeechTokenizerDecoder

    public init(config: Qwen3TTSTokenizerConfig) {
        self.config = config
        self.encoderValidNumQuantizers = config.encoder_valid_num_quantizers
        self.inputSampleRate = config.input_sample_rate
        self.outputSampleRate = config.output_sample_rate
        self.decodeUpsampleRate = config.decode_upsample_rate

        self.decoder = Qwen3TTSSpeechTokenizerDecoder(config: config.decoder_config ?? Qwen3TTSTokenizerDecoderConfig())
        super.init()
    }

    public func decode(_ audioCodes: MLXArray) -> (MLXArray, MLXArray) {
        let codes = audioCodes.transposed(0, 2, 1)
        let chunkSize = Int(ProcessInfo.processInfo.environment["QWEN3TTS_DECODE_CHUNK_SIZE"] ?? "") ?? 100
        let leftContext = Int(ProcessInfo.processInfo.environment["QWEN3TTS_DECODE_LEFT_CONTEXT"] ?? "") ?? 10
        let wav = decoder.chunkedDecode(codes, chunkSize: chunkSize, leftContextSize: leftContext).squeezed(axis: 1)

        let validMask = audioCodes[0..., 0..., 0] .> 0
        let audioLengths = MLX.sum(validMask.asType(.int32), axis: 1) * Int32(decodeUpsampleRate)

        return (wav, audioLengths)
    }
}
