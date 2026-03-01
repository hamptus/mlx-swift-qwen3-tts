import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - RMSNorm

nonisolated class Qwen3RMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalDtype = x.dtype
        let xFloat = x.asType(.float32)
        let meanSqr = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let rsqrtParams = rsqrt(meanSqr + eps)
        let normalized = xFloat * rsqrtParams * weight
        return normalized.asType(originalDtype)
    }
}

// MARK: - Rotary Embedding

nonisolated class Qwen3RotaryEmbedding: Module {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let invFreq: MLXArray
    let mropeSection: [Int]
    let useInterleaved: Bool

    init(dim: Int, maxPositionEmbeddings: Int = 2048, base: Float = 10000.0, mropeSection: [Int]? = nil) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.mropeSection = mropeSection ?? [24, 20, 20]
        self.useInterleaved = mropeSection != nil

        let freqs = (0..<dim/2).map { 1.0 / pow(base, Float($0 * 2) / Float(dim)) }
        self.invFreq = MLXArray(freqs)
        super.init()
    }

    private func applyInterleavedMRoPE(freqs: MLXArray) -> MLXArray {
        let headDimHalf = freqs.shape[3]

        let freqsT = freqs[0]
        let freqsH = freqs[1]
        let freqsW = freqs[2]

        let hLength = mropeSection[1] * 3
        let wLength = mropeSection[2] * 3

        let indices = MLXArray(0..<headDimHalf)

        let hMask = (indices % 3 .== 1) .&& (indices .< hLength)
        let wMask = (indices % 3 .== 2) .&& (indices .< wLength)

        let hMaskExpanded = hMask.reshaped([1, 1, headDimHalf])
        let wMaskExpanded = wMask.reshaped([1, 1, headDimHalf])

        var freqsCombined = which(hMaskExpanded, freqsH, freqsT)
        freqsCombined = which(wMaskExpanded, freqsW, freqsCombined)

        return freqsCombined
    }

    func cosSin(positionIds: MLXArray) -> (MLXArray, MLXArray) {
        if useInterleaved {
            var pos3d: MLXArray
            if positionIds.ndim == 2 {
                pos3d = stacked([positionIds, positionIds, positionIds], axis: 0)
            } else {
                pos3d = positionIds
            }

            let invFreqExpanded = invFreq.expandedDimensions(axes: [0, 1, 3])
            let posExpanded = pos3d.asType(.float32).expandedDimensions(axis: 2)

            var freqs = matmul(invFreqExpanded, posExpanded).transposed(0, 1, 3, 2)
            eval(freqs)

            freqs = applyInterleavedMRoPE(freqs: freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb), sin(emb))
        } else {
            let invFreqExpanded = invFreq.expandedDimensions(axis: 0)
            let posExpanded = positionIds.asType(.float32).expandedDimensions(axis: -1)
            let emb = matmul(posExpanded, invFreqExpanded)
            let result = concatenated([emb, emb], axis: -1)
            return (cos(result), sin(result))
        }
    }
}

// MARK: - KV Cache

public typealias KVCache = (MLXArray, MLXArray)

/// Maximum number of tokens to keep in KV cache (sliding window)
let maxKVCacheWindow: Int = 192

/// Trim KV cache to keep only the most recent tokens within the window.
func trimKVCache(_ cache: [KVCache], maxWindow: Int) -> [KVCache] {
    guard let first = cache.first else { return cache }
    let currentLen = first.0.shape[2]

    guard currentLen > maxWindow else { return cache }

    let startIdx = currentLen - maxWindow
    return cache.map { (k, v) in
        let kTrimmed = k[0..., 0..., startIdx..., 0...]
        let vTrimmed = v[0..., 0..., startIdx..., 0...]
        eval(kTrimmed, vTrimmed)
        return (kTrimmed, vTrimmed)
    }
}

// MARK: - Attention

nonisolated class Qwen3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let numKVGroups: Int

    let q_proj: Linear
    let k_proj: Linear
    let v_proj: Linear
    let o_proj: Linear

    let q_norm: Qwen3RMSNorm
    let k_norm: Qwen3RMSNorm

    let rope: Qwen3RotaryEmbedding

    init(config: Qwen3TTSConfig) {
        self.numHeads = config.num_attention_heads
        self.numKVHeads = config.num_key_value_heads
        self.headDim = config.head_dim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.numKVGroups = numHeads / numKVHeads

        let qs = config.quantizationSettings

        self.q_proj = QuantizedLayerFactory.linear(config.hidden_size, numHeads * headDim, bias: false, settings: qs)
        self.k_proj = QuantizedLayerFactory.linear(config.hidden_size, numKVHeads * headDim, bias: false, settings: qs)
        self.v_proj = QuantizedLayerFactory.linear(config.hidden_size, numKVHeads * headDim, bias: false, settings: qs)
        self.o_proj = QuantizedLayerFactory.linear(numHeads * headDim, config.hidden_size, bias: false, settings: qs)

        self.q_norm = Qwen3RMSNorm(dims: headDim, eps: config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(dims: headDim, eps: config.rms_norm_eps)

        self.rope = Qwen3RotaryEmbedding(dim: headDim, maxPositionEmbeddings: config.max_position_embeddings, base: config.rope_theta, mropeSection: config.mrope_section)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil, positionIds: MLXArray? = nil) -> (MLXArray, KVCache) {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

        var q = q_proj(x).reshaped([B, L, numHeads, headDim])
        var k = k_proj(x).reshaped([B, L, numKVHeads, headDim])
        var v = v_proj(x).reshaped([B, L, numKVHeads, headDim])

        q = q_norm(q)
        k = k_norm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let positions = positionIds ?? MLXArray(0..<L).expandedDimensions(axis: 0)
        let (cos, sin) = rope.cosSin(positionIds: positions)

        let cosB = cos.expandedDimensions(axis: 1)
        let sinB = sin.expandedDimensions(axis: 1)

        func rotateHalf(_ x: MLXArray) -> MLXArray {
            let parts = split(x, indices: [x.shape.last! / 2], axis: -1)
            let x1 = parts[0]
            let x2 = parts[1]
            return concatenated([-x2, x1], axis: -1)
        }

        q = (q * cosB) + (rotateHalf(q) * sinB)
        k = (k * cosB) + (rotateHalf(k) * sinB)

        if let (kCache, vCache) = cache {
            k = concatenated([kCache, k], axis: 2)
            v = concatenated([vCache, v], axis: 2)
        }
        let newCache = (k, v)

        if numKVGroups > 1 {
            k = MLXArray.repeated(k, count: numKVGroups, axis: 1)
            v = MLXArray.repeated(v, count: numKVGroups, axis: 1)
        }

        let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale

        var probs = scores
        if let mask = mask {
            probs = probs + mask
        }
        probs = softmax(probs, axis: -1)

        let output = matmul(probs, v).transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        return (o_proj(output), newCache)
    }
}

// MARK: - MLP

nonisolated class Qwen3MLP: Module {
    let gate_proj: Linear
    let up_proj: Linear
    let down_proj: Linear

    init(hiddenSize: Int, intermediateSize: Int, quantization: QuantizationSettings = .fullPrecision) {
        self.gate_proj = QuantizedLayerFactory.linear(hiddenSize, intermediateSize, bias: false, settings: quantization)
        self.up_proj = QuantizedLayerFactory.linear(hiddenSize, intermediateSize, bias: false, settings: quantization)
        self.down_proj = QuantizedLayerFactory.linear(intermediateSize, hiddenSize, bias: false, settings: quantization)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down_proj(silu(gate_proj(x)) * up_proj(x))
    }
}

// MARK: - Decoder Layer

nonisolated class Qwen3DecoderLayer: Module {
    let self_attn: Qwen3Attention
    let mlp: Qwen3MLP
    let input_layernorm: Qwen3RMSNorm
    let post_attention_layernorm: Qwen3RMSNorm

    init(config: Qwen3TTSConfig) {
        self.self_attn = Qwen3Attention(config: config)
        self.input_layernorm = Qwen3RMSNorm(dims: config.hidden_size, eps: config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(dims: config.hidden_size, eps: config.rms_norm_eps)
        self.mlp = Qwen3MLP(hiddenSize: config.hidden_size, intermediateSize: config.intermediate_size, quantization: config.quantizationSettings)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil, positionIds: MLXArray? = nil) -> (MLXArray, KVCache) {
        let (r, newCache) = self_attn(input_layernorm(x), mask: mask, cache: cache, positionIds: positionIds)
        let h = x + r
        let m = mlp(post_attention_layernorm(h))
        return (h + m, newCache)
    }
}

// MARK: - Text Projection

nonisolated public class Qwen3TextProjection: Module {
    let linear_fc1: Linear
    let linear_fc2: Linear

    init(textHiddenSize: Int, hiddenSize: Int, quantization: QuantizationSettings = .fullPrecision) {
        self.linear_fc1 = QuantizedLayerFactory.linear(textHiddenSize, textHiddenSize, bias: true, settings: quantization)
        self.linear_fc2 = QuantizedLayerFactory.linear(textHiddenSize, hiddenSize, bias: true, settings: quantization)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = silu(linear_fc1(x))
        return linear_fc2(h)
    }
}
