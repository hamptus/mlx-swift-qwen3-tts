import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Code Predictor Rotary Embedding

nonisolated class CodePredictorRotaryEmbedding: Module {
    let dim: Int
    let base: Float
    let invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int = 65536, base: Float = 1000000.0) {
        self.dim = dim
        self.base = base
        let freqs = (0..<dim/2).map { 1.0 / pow(base, Float($0 * 2) / Float(dim)) }
        self.invFreq = MLXArray(freqs)
        super.init()
    }

    func cosSin(positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let invFreqExpanded = invFreq.expandedDimensions(axis: 0)
        let posExpanded = positionIds.expandedDimensions(axis: -1)
        let emb = matmul(posExpanded, invFreqExpanded)
        let result = concatenated([emb, emb], axis: -1)
        return (cos(result), sin(result))
    }
}

// MARK: - Code Predictor Attention

nonisolated class CodePredictorAttention: Module {
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
    let rope: CodePredictorRotaryEmbedding

    init(config: CodePredictorConfig) {
        self.numHeads = config.num_attention_heads
        self.numKVHeads = config.num_key_value_heads
        self.headDim = config.head_dim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.numKVGroups = numHeads / numKVHeads

        let qs = config.quantization
        self.q_proj = QuantizedLayerFactory.linear(config.hidden_size, numHeads * headDim, bias: false, settings: qs)
        self.k_proj = QuantizedLayerFactory.linear(config.hidden_size, numKVHeads * headDim, bias: false, settings: qs)
        self.v_proj = QuantizedLayerFactory.linear(config.hidden_size, numKVHeads * headDim, bias: false, settings: qs)
        self.o_proj = QuantizedLayerFactory.linear(numHeads * headDim, config.hidden_size, bias: false, settings: qs)

        self.q_norm = Qwen3RMSNorm(dims: headDim, eps: config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(dims: headDim, eps: config.rms_norm_eps)
        self.rope = CodePredictorRotaryEmbedding(dim: headDim, maxPositionEmbeddings: config.max_position_embeddings, base: config.rope_theta)

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
            return concatenated([-parts[1], parts[0]], axis: -1)
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

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask = mask {
            scores = scores + mask
        }
        let probs = softmax(scores, axis: -1)
        let output = matmul(probs, v).transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])

        return (o_proj(output), newCache)
    }
}

// MARK: - Code Predictor Decoder Layer

nonisolated class CodePredictorDecoderLayer: Module {
    let self_attn: CodePredictorAttention
    let mlp: Qwen3MLP
    let input_layernorm: Qwen3RMSNorm
    let post_attention_layernorm: Qwen3RMSNorm

    init(config: CodePredictorConfig) {
        self.self_attn = CodePredictorAttention(config: config)
        self.mlp = Qwen3MLP(hiddenSize: config.hidden_size, intermediateSize: config.intermediate_size, quantization: config.quantization)
        self.input_layernorm = Qwen3RMSNorm(dims: config.hidden_size, eps: config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(dims: config.hidden_size, eps: config.rms_norm_eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil, positionIds: MLXArray? = nil) -> (MLXArray, KVCache) {
        let (r, newCache) = self_attn(input_layernorm(x), mask: mask, cache: cache, positionIds: positionIds)
        let h = x + r
        let m = mlp(post_attention_layernorm(h))
        return (h + m, newCache)
    }
}

// MARK: - Code Predictor

nonisolated public class Qwen3CodePredictor: Module {
    let config: CodePredictorConfig
    let talkerHiddenSize: Int

    let codec_embedding: [Embedding]
    let layers: [CodePredictorDecoderLayer]
    let norm: Qwen3RMSNorm
    let lm_head: [Linear]
    let small_to_mtp_projection: Linear?

    public init(config: CodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.talkerHiddenSize = talkerHiddenSize

        let qs = config.quantization

        self.codec_embedding = (0..<(config.num_code_groups - 1)).map { _ in
            Embedding(embeddingCount: config.vocab_size, dimensions: talkerHiddenSize)
        }

        self.layers = (0..<config.num_hidden_layers).map { _ in
            CodePredictorDecoderLayer(config: config)
        }
        self.norm = Qwen3RMSNorm(dims: config.hidden_size, eps: config.rms_norm_eps)

        self.lm_head = (0..<(config.num_code_groups - 1)).map { _ in
            QuantizedLayerFactory.linear(config.hidden_size, config.vocab_size, bias: false, settings: qs)
        }

        if config.hidden_size != talkerHiddenSize {
            self.small_to_mtp_projection = QuantizedLayerFactory.linear(talkerHiddenSize, config.hidden_size, bias: true, settings: qs)
        } else {
            self.small_to_mtp_projection = nil
        }

        super.init()
    }

    func callAsFunction(_ inputEmbeds: MLXArray, cache: [KVCache]?, generationStep: Int) -> (MLXArray, [KVCache]) {
        var x = inputEmbeds

        if let proj = small_to_mtp_projection {
            x = proj(x)
        }

        let L = x.shape[1]
        var mask: MLXArray? = nil
        if L > 1 {
            mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(L, dtype: .float32)
        }

        var offset = 0
        if let cache = cache, let first = cache.first {
            offset = first.0.shape[2]
        }
        let positionIds = MLXArray((offset..<offset+L).map { Int32($0) }).expandedDimensions(axis: 0)

        var newCaches: [KVCache] = []
        for (i, layer) in layers.enumerated() {
            let layerCache: KVCache? = (cache != nil && i < cache!.count) ? cache![i] : nil
            let (out, c) = layer(x, mask: mask, cache: layerCache, positionIds: positionIds)
            x = out
            newCaches.append(c)
        }

        x = norm(x)
        guard generationStep < lm_head.count else {
            print("CRASH AVOIDED [CodePredictor]: generationStep=\(generationStep) >= lm_head.count=\(lm_head.count)")
            return (MLXArray.zeros(like: x), newCaches)
        }
        let logits = lm_head[generationStep](x)

        return (logits, newCaches)
    }
}
