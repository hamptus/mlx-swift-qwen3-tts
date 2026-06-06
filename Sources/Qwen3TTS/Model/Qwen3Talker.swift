import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Qwen3 Talker Model

nonisolated public class Qwen3Talker: Module {
    public let config: Qwen3TTSConfig
    public let codec_embedding: Embedding
    public let text_embedding: Embedding
    public let text_projection: Qwen3TextProjection
    public let codec_head: Linear
    let layers: [Qwen3DecoderLayer]
    let norm: Qwen3RMSNorm

    public let code_predictor: Qwen3CodePredictor

    private var _cachedValidMask: MLXArray?
    private var cachedValidMask: MLXArray {
        if let mask = _cachedValidMask { return mask }
        let vocabSize = config.vocab_size
        let codebookSize = 2048
        let padTokenId: Int32 = 2148
        let eosTokenId: Int32 = 2150
        let indices = MLXArray(Array(0..<Int32(vocabSize)))
        let validCodebook = indices .< Int32(codebookSize)
        let validPad = indices .== padTokenId
        let validEos = indices .== eosTokenId
        let mask = logicalOr(logicalOr(validCodebook, validPad), validEos)
        _cachedValidMask = mask
        return mask
    }

    public init(config: Qwen3TTSConfig) {
        self.config = config

        let qs = config.quantizationSettings

        self.text_embedding = Embedding(embeddingCount: config.text_vocab_size, dimensions: config.text_hidden_size)
        self.text_projection = Qwen3TextProjection(textHiddenSize: config.text_hidden_size, hiddenSize: config.hidden_size, quantization: qs)
        self.codec_embedding = Embedding(embeddingCount: config.vocab_size, dimensions: config.hidden_size)
        self.codec_head = QuantizedLayerFactory.linear(config.hidden_size, config.vocab_size, bias: false, settings: qs)

        self.layers = (0..<config.num_hidden_layers).map { _ in Qwen3DecoderLayer(config: config) }
        self.norm = Qwen3RMSNorm(dims: config.hidden_size, eps: config.rms_norm_eps)

        let cpConfig = config.code_predictor_config
        var codePredictorConfig = CodePredictorConfig(
            hidden_size: cpConfig.hidden_size,
            num_hidden_layers: cpConfig.num_hidden_layers,
            num_attention_heads: cpConfig.num_attention_heads,
            num_key_value_heads: cpConfig.num_key_value_heads,
            head_dim: cpConfig.head_dim,
            intermediate_size: cpConfig.intermediate_size,
            rms_norm_eps: cpConfig.rms_norm_eps,
            max_position_embeddings: cpConfig.max_position_embeddings,
            rope_theta: cpConfig.rope_theta,
            vocab_size: cpConfig.vocab_size,
            num_code_groups: cpConfig.num_code_groups
        )
        codePredictorConfig.quantization = qs
        self.code_predictor = Qwen3CodePredictor(config: codePredictorConfig, talkerHiddenSize: config.hidden_size)

        super.init()
    }

    public func clearGenerationCache() {
        _cachedValidMask = nil
    }

    // Core forward pass with pre-computed embeddings
    public func callAsFunction(_ x: MLXArray, cache: [KVCache]? = nil, positionOffset: Int? = nil) -> (MLXArray, [KVCache]) {
        let (_, L) = (x.shape[0], x.shape[1])

        var mask: MLXArray? = nil
        if L > 1 {
            mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(L, dtype: .float32)
        }

        var offset = positionOffset ?? 0
        if positionOffset == nil, let cache = cache, let first = cache.first {
            offset = first.0.shape[2]
        }

        let positionIds = MLXArray((offset..<offset+L).map { Int32($0) }).expandedDimensions(axis: 0)

        var newCaches: [KVCache] = []
        var h = x

        for (i, layer) in layers.enumerated() {
            let layerCache: KVCache? = (cache != nil && i < cache!.count) ? cache![i] : nil
            let (out, c) = layer(h, mask: mask, cache: layerCache, positionIds: positionIds)
            h = out
            newCaches.append(c)
        }

        let output = norm(h)

        return (output, newCaches)
    }

    public func encodeText(_ inputIds: MLXArray) -> MLXArray {
        let embedded = text_embedding(inputIds)
        return text_projection(embedded)
    }

    public func encodeAudio(_ inputIds: MLXArray) -> MLXArray {
        return codec_embedding(inputIds)
    }

    // MARK: - Weight Loading

    public func load(weights: [String: MLXArray]) {
        var newWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            if key.hasPrefix("audio_decoder.") {
                continue
            }

            if newKey.hasPrefix("talker.") {
                newKey = String(newKey.dropFirst("talker.".count))
            }

            if newKey.hasPrefix("code_predictor.model.") {
                newKey = "code_predictor." + String(newKey.dropFirst("code_predictor.model.".count))
            }

            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }

            newWeights[newKey] = value
        }

        let usePreQuantized = config.quantization != nil

        if !usePreQuantized {
            let quantGroupSize = config.quantization_config?.group_size ?? 64
            let quantBits = config.quantization_config?.bits ?? 8
            let quantMode: QuantizationMode = config.quantization_config?.mode == "mxfp4" ? .mxfp4 : .affine

            var keysToRemove = Set<String>()
            let weightKeys = newWeights.keys.filter { $0.hasSuffix(".weight") }
            for key in weightKeys {
                guard let weight = newWeights[key] else { continue }
                let scalesKey = key.replacingOccurrences(of: ".weight", with: ".scales")
                let biasesKey = key.replacingOccurrences(of: ".weight", with: ".biases")
                guard let scales = newWeights[scalesKey] else { continue }

                if weight.dtype == .uint8 || weight.dtype == .uint16 || weight.dtype == .uint32 {
                    let biases = newWeights[biasesKey]
                    let dq = dequantized(
                        weight,
                        scales: scales,
                        biases: biases,
                        groupSize: quantGroupSize,
                        bits: quantBits,
                        mode: quantMode,
                        dtype: .float16
                    )
                    eval(dq)
                    newWeights[key] = dq
                    keysToRemove.insert(scalesKey)
                    keysToRemove.insert(biasesKey)
                }
            }
            for key in keysToRemove {
                newWeights.removeValue(forKey: key)
            }
            newWeights = newWeights.filter { !($0.key.hasSuffix(".scales") || $0.key.hasSuffix(".biases")) }
        }

        do {
            let params = ModuleParameters.unflattened(newWeights)
            try self.update(parameters: params, verify: .none)

            // Manually load code predictor weights (arrays need explicit handling)
            @discardableResult
            func loadQuantizedLinear(_ module: Linear, prefix: String) throws -> Bool {
                guard let w = newWeights["\(prefix).weight"] else { return false }
                var params: [String: MLXArray] = ["weight": w]
                if let s = newWeights["\(prefix).scales"] {
                    params["scales"] = s
                }
                if let b = newWeights["\(prefix).biases"] {
                    params["biases"] = b
                }
                let p = ModuleParameters.unflattened(params)
                try module.update(parameters: p, verify: .none)
                return true
            }

            let numCodeEmbeddings = code_predictor.codec_embedding.count
            for i in 0..<numCodeEmbeddings {
                if let w = newWeights["code_predictor.codec_embedding.\(i).weight"] {
                    let moduleParams = ModuleParameters.unflattened(["weight": w])
                    try code_predictor.codec_embedding[i].update(parameters: moduleParams, verify: .none)
                }
            }

            let numLmHeads = code_predictor.lm_head.count
            for i in 0..<numLmHeads {
                try loadQuantizedLinear(code_predictor.lm_head[i], prefix: "code_predictor.lm_head.\(i)")
            }

            if let w = newWeights["code_predictor.norm.weight"] {
                let moduleParams = ModuleParameters.unflattened(["weight": w])
                try code_predictor.norm.update(parameters: moduleParams, verify: .none)
            }

            if let proj = code_predictor.small_to_mtp_projection {
                var params: [String: MLXArray] = [:]
                if let w = newWeights["code_predictor.small_to_mtp_projection.weight"] {
                    params["weight"] = w
                }
                if let b = newWeights["code_predictor.small_to_mtp_projection.bias"] {
                    params["bias"] = b
                }
                if let s = newWeights["code_predictor.small_to_mtp_projection.scales"] {
                    params["scales"] = s
                }
                if let bi = newWeights["code_predictor.small_to_mtp_projection.biases"] {
                    params["biases"] = bi
                }
                if !params.isEmpty {
                    let moduleParams = ModuleParameters.unflattened(params)
                    try proj.update(parameters: moduleParams, verify: .none)
                }
            }

            let numCpLayers = code_predictor.layers.count
            for i in 0..<numCpLayers {
                let prefix = "code_predictor.layers.\(i)"
                let layer = code_predictor.layers[i]

                if let w = newWeights["\(prefix).input_layernorm.weight"] {
                    let p = ModuleParameters.unflattened(["weight": w])
                    try layer.input_layernorm.update(parameters: p, verify: .none)
                }

                if let w = newWeights["\(prefix).post_attention_layernorm.weight"] {
                    let p = ModuleParameters.unflattened(["weight": w])
                    try layer.post_attention_layernorm.update(parameters: p, verify: .none)
                }

                try loadQuantizedLinear(layer.self_attn.q_proj, prefix: "\(prefix).self_attn.q_proj")
                try loadQuantizedLinear(layer.self_attn.k_proj, prefix: "\(prefix).self_attn.k_proj")
                try loadQuantizedLinear(layer.self_attn.v_proj, prefix: "\(prefix).self_attn.v_proj")
                try loadQuantizedLinear(layer.self_attn.o_proj, prefix: "\(prefix).self_attn.o_proj")

                if let w = newWeights["\(prefix).self_attn.q_norm.weight"] {
                    let p = ModuleParameters.unflattened(["weight": w])
                    try layer.self_attn.q_norm.update(parameters: p, verify: .none)
                }
                if let w = newWeights["\(prefix).self_attn.k_norm.weight"] {
                    let p = ModuleParameters.unflattened(["weight": w])
                    try layer.self_attn.k_norm.update(parameters: p, verify: .none)
                }

                try loadQuantizedLinear(layer.mlp.gate_proj, prefix: "\(prefix).mlp.gate_proj")
                try loadQuantizedLinear(layer.mlp.up_proj, prefix: "\(prefix).mlp.up_proj")
                try loadQuantizedLinear(layer.mlp.down_proj, prefix: "\(prefix).mlp.down_proj")
            }
        } catch {
            // Weight update failed
        }
    }

    // MARK: - Token Sampling

    func sampleToken(
        logits: MLXArray,
        temperature: Float = 0.9,
        topK: Int = 0,
        eosTokenId: Int32 = 2150,
        repetitionPenalty: Float = 1.05,
        generatedTokenSet: Set<Int32>? = nil
    ) -> MLXArray {
        var logits = logits

        if logits.shape.count == 3 {
            logits = logits[0..., (logits.shape[1] - 1)..<logits.shape[1], 0...].squeezed(axis: 1)
        }

        if let uniqueTokens = generatedTokenSet, !uniqueTokens.isEmpty, repetitionPenalty != 1.0 {
            let vocabSize = logits.shape.last!
            var penaltyValues = [Float](repeating: 1.0, count: vocabSize)
            for token in uniqueTokens {
                let idx = Int(token)
                if idx < vocabSize {
                    penaltyValues[idx] = repetitionPenalty
                }
            }
            let penaltyArray = MLXArray(penaltyValues).expandedDimensions(axis: 0)
            logits = logits / penaltyArray
        }

        if temperature > 0 {
            logits = logits / temperature
        } else {
            return argMax(logits, axis: -1)
        }

        if topK > 0 && topK < logits.shape.last! {
            let vocabSize = logits.shape.last!
            let k = min(topK, vocabSize)
            let topValues = top(logits, k: k, axis: -1)
            let threshold = topValues.min(axis: -1, keepDims: true)
            let mask = logits .< threshold
            logits = which(mask, MLXArray(-Float.infinity), logits)
        }

        let vocabSize = logits.shape.last!
        if vocabSize == config.vocab_size {
            logits = which(cachedValidMask, logits, MLXArray(-Float.infinity))
        }

        return MLXRandom.categorical(logits, axis: -1)
    }

    // MARK: - Generation

    /// True when every codebook row has the same non-zero length. The ICL paths
    /// index `refCodes[q][t]` and sum per-codebook `MLXArray`s, so a jagged array
    /// would trap on index-out-of-range or abort on an MLX broadcast mismatch.
    public static func referenceCodesAreRectangular(_ codes: [[Int32]]) -> Bool {
        guard let width = codes.first?.count, width > 0 else { return false }
        return codes.allSatisfy { $0.count == width }
    }

    /// True when the inputs constitute a valid ICL voice-cloning request: a
    /// non-empty reference transcript plus rectangular reference audio codes with
    /// at least one frame. Mirrors the guard on the ICL prefill branch so the
    /// repetition-penalty floor and KV-cache trimming policy never diverge from
    /// the path that actually runs.
    public static func isICLInput(referenceAudioCodes: [[Int32]]?, referenceTranscript: String?) -> Bool {
        guard let transcript = referenceTranscript, !transcript.isEmpty else { return false }
        guard let refCodes = referenceAudioCodes, referenceCodesAreRectangular(refCodes) else { return false }
        return true
    }

    /// Generate audio codes without decoding (for batch decoding later).
    public func generateCodes(
        prompt: String,
        text: String,
        instruct: String? = nil,
        speakerEmbedding: MLXArray? = nil,
        referenceTranscript: String? = nil,
        referenceAudioCodes: [[Int32]]? = nil,
        tokenizer: Qwen3Tokenizer,
        temperature: Float = 0.9,
        detailTemperature: Float? = nil,
        code0TopK: Int = 80,
        code0RepetitionPenalty: Float = 1.15,
        maxTokens: Int = 1200,
        iclMaxTokensCeiling: Int? = nil,
        iclTokensPerTextToken: Int? = nil
    ) -> [[Int32]] {
        // Detail codes (1-15) use lower temperature for acoustic fidelity;
        // code0 (semantic/prosodic) keeps the user-specified temperature for natural variation.
        let resolvedDetailTemp = detailTemperature ?? max(0.3, temperature * 0.65)
        // useICL must match the conditions of the ICL prefill branch below exactly.
        // A non-nil-but-empty (or zero-frame) referenceAudioCodes is NOT ICL: it must
        // not floor the repetition penalty or disable KV-cache trimming for what is
        // really a normal generation.
        let useICL = Qwen3Talker.isICLInput(referenceAudioCodes: referenceAudioCodes, referenceTranscript: referenceTranscript)
        let speakerName = prompt.lowercased()
        let speakerId = config.spk_id[speakerName]
        let debugGenEntry = ProcessInfo.processInfo.environment["DUPER_DEBUG_GENERATION"] == "1"
        if debugGenEntry { print("DEBUG [generateCodes]: entry prompt='\(prompt.prefix(30))' text='\(text.prefix(30))' speakerId=\(speakerId as Any) spkEmbed=\(speakerEmbedding?.shape ?? []) useICL=\(useICL) temp=\(temperature) detailTemp=\(resolvedDetailTemp) code0TopK=\(code0TopK) code0RepPen=\(code0RepetitionPenalty)"); fflush(stdout) }

        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let chatIdsRaw: [Int32] = tokenizer.encode(text: chatText)
        let inputIds = MLXArray(chatIdsRaw).expandedDimensions(axis: 0)
        if debugGenEntry { print("DEBUG [generateCodes]: inputIds shape=\(inputIds.shape)"); fflush(stdout) }

        let minTokens = 9
        guard inputIds.shape[1] >= minTokens else {
            if debugGenEntry { print("DEBUG [generateCodes]: input too short (\(inputIds.shape[1]) < \(minTokens))") }
            return []
        }

        // When ICL sizing hints are supplied, derive the per-call token cap from the
        // already-tokenized input instead of having the caller tokenize a second time.
        let resolvedMaxTokens: Int
        if let ceiling = iclMaxTokensCeiling, let perToken = iclTokensPerTextToken {
            resolvedMaxTokens = min(ceiling, max(75, inputIds.shape[1] * perToken))
        } else {
            resolvedMaxTokens = maxTokens
        }

        let ttsTokens = MLXArray([Int32(config.tts_bos_token_id), Int32(config.tts_eos_token_id), Int32(config.tts_pad_token_id)]).expandedDimensions(axis: 0)
        let ttsEmbeds = text_projection(text_embedding(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        let codecPrefill = MLXArray([
            Int32(config.codec_nothink_id),
            Int32(config.codec_think_bos_id),
            Int32(config.codec_think_eos_id)
        ]).expandedDimensions(axis: 0)
        var codecEmbed = codec_embedding(codecPrefill)

        let codecSuffix = MLXArray([Int32(config.codec_pad_id), Int32(config.codec_bos_id)]).expandedDimensions(axis: 0)
        let codecSuffixEmbed = codec_embedding(codecSuffix)

        if let spkId = speakerId {
            let speakerIds = MLXArray([Int32(spkId)]).expandedDimensions(axis: 0)
            let speakerEmbed = codec_embedding(speakerIds)
            codecEmbed = concatenated([codecEmbed, speakerEmbed, codecSuffixEmbed], axis: 1)
        } else if let spkEmbed = speakerEmbedding {
            let speakerEmbed = spkEmbed.reshaped([1, 1, -1])
            codecEmbed = concatenated([codecEmbed, speakerEmbed, codecSuffixEmbed], axis: 1)
        } else {
            codecEmbed = concatenated([codecEmbed, codecSuffixEmbed], axis: 1)
        }

        let roleEmbed = text_projection(text_embedding(inputIds[0..., 0..<3]))

        let padCount = codecEmbed.shape[1] - 2
        let padEmbeds = tiled(ttsPadEmbed, repetitions: [1, padCount, 1])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., 0..<(codecEmbed.shape[1]-1), 0...]

        let numCodeGroups = config.code_predictor_config.num_code_groups
        let effectiveRepPenalty = useICL ? max(code0RepetitionPenalty, 1.5) : code0RepetitionPenalty

        var inputEmbeds: MLXArray
        var trailingTextHidden: MLXArray

        if useICL, let refCodes = referenceAudioCodes, let refTranscript = referenceTranscript, !refCodes.isEmpty, let refTime = refCodes.first?.count, refTime > 0 {
            // ICL voice cloning: all text (ref + target) goes in prefill with proper
            // codec_pad/tts_pad overlays. During generation, trailing_text_hidden is
            // just tts_pad. Matches mlx-audio's _prepare_icl_generation_inputs
            // non-streaming layout.

            // Tokenize reference text with assistant role (NOT user)
            let refChat = "<|im_start|>assistant\n\(refTranscript)<|im_end|>\n"
            let refIdsArray: [Int32] = tokenizer.encode(text: refChat)
            guard refIdsArray.count > 5 else { return [] }
            let refTextIds = Array(refIdsArray[3..<(refIdsArray.count - 2)])

            // Pure target text tokens (skip role prefix [:3] and trailing template [-5:])
            guard chatIdsRaw.count > 8 else { return [] }
            let targetTextIds = Array(chatIdsRaw[3..<(chatIdsRaw.count - 5)])

            // Combine ref + target text, project through text embeddings, append tts_eos
            let combinedTextIds = refTextIds + targetTextIds
            let combinedTextArray = MLXArray(combinedTextIds).expandedDimensions(axis: 0)
            var iclTextEmbed = text_projection(text_embedding(combinedTextArray))
            iclTextEmbed = concatenated([iclTextEmbed, ttsEosEmbed], axis: 1)
            let textLens = iclTextEmbed.shape[1]

            // Sum embeddings from ALL codebook groups (RVQ is additive)
            // First group via main codec_embedding, groups 1..N-1 via code_predictor
            let firstCbCodes = MLXArray(refCodes[0]).expandedDimensions(axis: 0)
            var refCodecEmbed = codec_embedding(firstCbCodes)
            for i in 0..<(numCodeGroups - 1) {
                guard i + 1 < refCodes.count else { break }
                guard i < code_predictor.codec_embedding.count else { break }
                let cbCodes = MLXArray(refCodes[i + 1]).expandedDimensions(axis: 0)
                refCodecEmbed = refCodecEmbed + code_predictor.codec_embedding[i](cbCodes)
            }

            // Prepend codec_bos to the summed ref codes
            let codecBosEmbed = codec_embedding(MLXArray([Int32(config.codec_bos_id)]).expandedDimensions(axis: 0))
            let codecEmbedIcl = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
            let codecLens = codecEmbedIcl.shape[1]

            // Non-streaming overlay: text positions get codec_pad, codec positions get tts_pad
            let codecPadEmbed = codec_embedding(MLXArray([Int32(config.codec_pad_id)]).expandedDimensions(axis: 0))
            let textWithCodecPad = iclTextEmbed + tiled(codecPadEmbed, repetitions: [1, textLens, 1])
            let codecWithTtsPad = codecEmbedIcl + tiled(ttsPadEmbed, repetitions: [1, codecLens, 1])
            let iclInputEmbed = concatenated([textWithCodecPad, codecWithTtsPad], axis: 1)

            // Full input: role + combined_prefix + ICL block
            inputEmbeds = concatenated([roleEmbed, combinedEmbed, iclInputEmbed], axis: 1)

            // All text consumed in prefill; generation only gets tts_pad
            trailingTextHidden = ttsPadEmbed

            if debugGenEntry {
                print("DEBUG [generateCodes ICL]: refText=\(refTextIds.count) targetText=\(targetTextIds.count) textLens=\(textLens) codecLens=\(codecLens) refFrames=\(refTime) repPen=\(effectiveRepPenalty)")
                fflush(stdout)
            }
        } else {
            var instructEmbed: MLXArray? = nil
            if let instructText = instruct, !instructText.isEmpty {
                // Explicit instruct text (VoiceDesign or CustomVoice mode)
                let formatted = "<|im_start|>user\n\(instructText)<|im_end|>\n"
                let instructIdsArray: [Int32] = tokenizer.encode(text: formatted)
                let instructIds = MLXArray(instructIdsArray).expandedDimensions(axis: 0)
                instructEmbed = text_projection(text_embedding(instructIds))
            } else if !prompt.isEmpty && speakerId == nil && speakerEmbedding == nil {
                // Backward compat: treat prompt as instruct when no speaker resolved
                let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n"
                let instructIdsArray: [Int32] = tokenizer.encode(text: formatted)
                let instructIds = MLXArray(instructIdsArray).expandedDimensions(axis: 0)
                instructEmbed = text_projection(text_embedding(instructIds))
            }

            if let instructEmbedding = instructEmbed {
                inputEmbeds = concatenated([instructEmbedding, roleEmbed, combinedEmbed], axis: 1)
            } else {
                inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
            }

            let firstTextEmbed = text_projection(text_embedding(inputIds[0..., 3..<4])) + codecEmbed[0..., (codecEmbed.shape[1]-1)..., 0...]
            inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

            let trailingLen = inputIds.shape[1] - 4 - 5
            if trailingLen > 0 {
                trailingTextHidden = text_projection(text_embedding(inputIds[0..., 4..<(inputIds.shape[1]-5)]))
                trailingTextHidden = concatenated([trailingTextHidden, ttsEosEmbed], axis: 1)
            } else {
                trailingTextHidden = ttsEosEmbed
            }
        }

        let debugGen = ProcessInfo.processInfo.environment["DUPER_DEBUG_GENERATION"] == "1"
        if debugGen { print("DEBUG [generateCodes]: inputEmbeds shape=\(inputEmbeds.shape) trailingShape=\(trailingTextHidden.shape) useICL=\(useICL)") }
        var (h, cache) = self.callAsFunction(inputEmbeds, cache: nil, positionOffset: nil)
        var positionOffset = inputEmbeds.shape[1]
        if debugGen { print("DEBUG [generateCodes]: prefill done, h shape=\(h.shape), positionOffset=\(positionOffset)") }

        var generatedCodes: [[Int32]] = []
        let eosTokenId: Int32 = Int32(config.codec_eos_token_id)
        let padTokenId: Int32 = Int32(config.codec_pad_id)
        var trailingIdx = 0
        var consecutivePad = 0
        if debugGen { print("DEBUG [generateCodes]: numCodeGroups=\(numCodeGroups), code_predictor embeddings=\(code_predictor.codec_embedding.count), lm_heads=\(code_predictor.lm_head.count)") }

        var logits = codec_head(h)

        var generatedCode0Tokens: [Int32] = []
        var generatedCode0TokensSet: Set<Int32> = []
        var generatedCodePredictorSets: [Set<Int32>] = Array(repeating: Set(), count: numCodeGroups - 1)

        // Pre-compute EOS/pad mask once — vocabSize is constant across steps
        let vocabSizeForMask = logits.shape.last!
        var eosMaskValues = [Float](repeating: 0.0, count: vocabSizeForMask)
        if Int(eosTokenId) < vocabSizeForMask { eosMaskValues[Int(eosTokenId)] = -Float.infinity }
        if Int(padTokenId) < vocabSizeForMask { eosMaskValues[Int(padTokenId)] = -Float.infinity }
        let eosMaskArray = MLXArray(eosMaskValues).expandedDimensions(axis: 0)

        let totalTextTokens = trailingTextHidden.shape[1]

        for step in 0..<resolvedMaxTokens {
            if Task.isCancelled { break }
            if debugGen && (step < 3 || step % 50 == 0) {
                print("DEBUG [generateCodes]: step \(step), logits shape=\(logits.shape), trailingIdx=\(trailingIdx)/\(totalTextTokens)")
            }

            let hasRemainingText = trailingIdx < totalTextTokens

            var samplingLogits = logits
            if hasRemainingText {
                samplingLogits = logits + eosMaskArray
            }

            let nextToken = sampleToken(
                logits: samplingLogits,
                temperature: temperature,
                topK: code0TopK,
                repetitionPenalty: effectiveRepPenalty,
                generatedTokenSet: generatedCode0TokensSet.isEmpty ? nil : generatedCode0TokensSet
            )
            let code0Value = nextToken[0].item(Int32.self)
            if debugGen && step < 3 { print("DEBUG [generateCodes]: step \(step) code0=\(code0Value)") }

            if code0Value == eosTokenId {
                break
            } else if code0Value == padTokenId {
                consecutivePad += 1
                if consecutivePad > 6 {
                    break
                }
            } else {
                consecutivePad = 0
            }

            var codeTokens: [Int32] = [code0Value]
            let nextTokenArray = nextToken.expandedDimensions(axis: 0)
            let codeHidden = h[0..., (h.shape[1]-1)..<h.shape[1], 0...]
            var codePredictorCache: [KVCache]? = nil

            for codeIdx in 0..<(numCodeGroups - 1) {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = codec_embedding(nextTokenArray)
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    guard codeIdx < codeTokens.count else { break }
                    guard codeIdx - 1 < code_predictor.codec_embedding.count else { break }
                    let prevCode = MLXArray([codeTokens[codeIdx]]).expandedDimensions(axis: 0)
                    codeInput = code_predictor.codec_embedding[codeIdx - 1](prevCode)
                }
                let (codeLogits, newCodeCache) = code_predictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                codePredictorCache = newCodeCache

                let codeToken = sampleToken(
                    logits: codeLogits,
                    temperature: resolvedDetailTemp,
                    generatedTokenSet: generatedCodePredictorSets[codeIdx].isEmpty ? nil : generatedCodePredictorSets[codeIdx]
                )
                let codeValue = codeToken[0].item(Int32.self)
                codeTokens.append(codeValue)
                generatedCodePredictorSets[codeIdx].insert(codeValue)
            }

            if debugGen && step < 3 { print("DEBUG [generateCodes]: step \(step) codeTokens=\(codeTokens.count) values=\(codeTokens.prefix(4))") }
            generatedCodes.append(codeTokens)
            generatedCode0Tokens.append(code0Value)
            generatedCode0TokensSet.insert(code0Value)
            codePredictorCache = nil

            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.shape[1] {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            var codecEmbedSum = codec_embedding(nextTokenArray)
            for i in 0..<(numCodeGroups - 1) {
                guard i + 1 < codeTokens.count else { break }
                guard i < code_predictor.codec_embedding.count else { break }
                let codeVal = MLXArray([codeTokens[i + 1]]).expandedDimensions(axis: 0)
                codecEmbedSum = codecEmbedSum + code_predictor.codec_embedding[i](codeVal)
            }
            eval(codecEmbedSum)

            inputEmbeds = textEmbed + codecEmbedSum
            eval(inputEmbeds)
            let (hStep, cStep) = self.callAsFunction(inputEmbeds, cache: cache, positionOffset: positionOffset)
            h = hStep
            cache = cStep
            logits = codec_head(h)
            positionOffset += 1

            if (step + 1) % 15 == 0 {
                // In ICL mode, trimming evicts the reference conditioning from KV cache,
                // causing the model to lose voice identity after ~15 steps. The Python
                // mlx-audio implementation does not trim during ICL generation.
                //
                // Memory note: with trimming disabled, peak KV memory for an ICL call is
                // bounded by prefill(ref codes + ref text + target text) + resolvedMaxTokens
                // rather than maxKVCacheWindow. The pipeline resets the cache per chunk, so
                // this peak is per-chunk, not cumulative across the whole utterance. A long
                // reference combined with a raised iclMaxTokensCeiling raises that peak.
                if !useICL {
                    cache = trimKVCache(cache, maxWindow: maxKVCacheWindow)
                }
                eval(h, logits)
                Stream.defaultStream(.gpu).synchronize()
                Memory.clearCache()
            }
        }

        cache = []
        h = MLXArray([])
        logits = MLXArray([])
        inputEmbeds = MLXArray([])
        Stream.defaultStream(.gpu).synchronize()
        Memory.clearCache()

        let validCodes = generatedCodes.filter { frame in
            guard let firstCode = frame.first else { return false }
            return firstCode >= 0 && firstCode < 2048
        }

        return validCodes
    }

    /// Generate audio codes and decode to audio samples.
    public func generate(
        prompt: String,
        text: String,
        instruct: String? = nil,
        speakerEmbedding: MLXArray? = nil,
        referenceTranscript: String? = nil,
        tokenizer: Qwen3Tokenizer,
        decoder: AudioDecoder,
        temperature: Float = 0.9,
        detailTemperature: Float? = nil,
        code0TopK: Int = 80,
        code0RepetitionPenalty: Float = 1.15,
        maxTokens: Int = 1200
    ) -> [Float] {
        let generatedCodes = generateCodes(
            prompt: prompt,
            text: text,
            instruct: instruct,
            speakerEmbedding: speakerEmbedding,
            referenceTranscript: referenceTranscript,
            tokenizer: tokenizer,
            temperature: temperature,
            detailTemperature: detailTemperature,
            code0TopK: code0TopK,
            code0RepetitionPenalty: code0RepetitionPenalty,
            maxTokens: maxTokens
        )

        guard !generatedCodes.isEmpty else {
            return []
        }

        let numCodeGroups = config.code_predictor_config.num_code_groups
        let flatCodes: [Int32] = generatedCodes.flatMap { $0 }
        let codesArray = MLXArray(flatCodes).reshaped([1, generatedCodes.count, numCodeGroups])

        let audio = decoder.decode(codes: codesArray)
        let flatAudio = audio.reshaped([-1])
        eval(flatAudio)
        let samples = flatAudio.asArray(Float.self)

        Memory.clearCache()

        guard !samples.isEmpty else {
            return []
        }

        let hasInvalid = samples.contains { $0.isNaN || $0.isInfinite }
        if hasInvalid {
            return samples.map { val in
                if val.isNaN || val.isInfinite { return 0.0 }
                return max(-1.0, min(1.0, val))
            }
        }

        return samples
    }

    /// Stream audio generation, yielding code chunks for incremental decoding.
    public func generateStream(
        prompt: String,
        text: String,
        instruct: String? = nil,
        speakerEmbedding: MLXArray? = nil,
        referenceTranscript: String? = nil,
        referenceAudioCodes: [[Int32]]? = nil,
        tokenizer: Qwen3Tokenizer,
        temperature: Float = 0.9,
        detailTemperature: Float? = nil,
        code0TopK: Int = 80,
        code0RepetitionPenalty: Float = 1.15,
        maxTokens: Int = 1200,
        chunkSize: Int = 12
    ) -> AsyncThrowingStream<[[Int32]], Error> {
        let model = self
        let config = self.config
        let resolvedDetailTemp = detailTemperature ?? max(0.3, temperature * 0.65)

        // Match the ICL prefill branch exactly (see isICLInput / generateCodes).
        let useICL = Qwen3Talker.isICLInput(referenceAudioCodes: referenceAudioCodes, referenceTranscript: referenceTranscript)

        return AsyncThrowingStream<[[Int32]], Error> { (continuation: AsyncThrowingStream<[[Int32]], Error>.Continuation) in
            Task {
                let speakerName = prompt.lowercased()
                let speakerId = config.spk_id[speakerName]

                let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
                let chatIdsRaw: [Int32] = tokenizer.encode(text: chatText)
                let inputIds = MLXArray(chatIdsRaw).expandedDimensions(axis: 0)

                let minTokens = 9
                guard inputIds.shape[1] >= minTokens else {
                    continuation.finish()
                    return
                }

                let ttsTokens = MLXArray([Int32(config.tts_bos_token_id), Int32(config.tts_eos_token_id), Int32(config.tts_pad_token_id)]).expandedDimensions(axis: 0)
                let ttsEmbeds = model.text_projection(model.text_embedding(ttsTokens))
                let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
                let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
                let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

                let codecPrefill = MLXArray([
                    Int32(config.codec_nothink_id),
                    Int32(config.codec_think_bos_id),
                    Int32(config.codec_think_eos_id)
                ]).expandedDimensions(axis: 0)
                var codecEmbed = model.codec_embedding(codecPrefill)

                let codecSuffix = MLXArray([Int32(config.codec_pad_id), Int32(config.codec_bos_id)]).expandedDimensions(axis: 0)
                let codecSuffixEmbed = model.codec_embedding(codecSuffix)

                if let spkId = speakerId {
                    let speakerIds = MLXArray([Int32(spkId)]).expandedDimensions(axis: 0)
                    let speakerEmbed = model.codec_embedding(speakerIds)
                    codecEmbed = concatenated([codecEmbed, speakerEmbed, codecSuffixEmbed], axis: 1)
                } else if let spkEmbed = speakerEmbedding {
                    let speakerEmbed = spkEmbed.reshaped([1, 1, -1])
                    codecEmbed = concatenated([codecEmbed, speakerEmbed, codecSuffixEmbed], axis: 1)
                } else {
                    codecEmbed = concatenated([codecEmbed, codecSuffixEmbed], axis: 1)
                }

                let roleEmbed = model.text_projection(model.text_embedding(inputIds[0..., 0..<3]))

                let padCount = codecEmbed.shape[1] - 2
                let padEmbeds = tiled(ttsPadEmbed, repetitions: [1, padCount, 1])
                var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
                combinedEmbed = combinedEmbed + codecEmbed[0..., 0..<(codecEmbed.shape[1]-1), 0...]

                let numCodeGroups = config.code_predictor_config.num_code_groups
                let effectiveRepPenalty = useICL ? max(code0RepetitionPenalty, 1.5) : code0RepetitionPenalty

                var inputEmbeds: MLXArray
                var trailingTextHidden: MLXArray

                if useICL, let refCodes = referenceAudioCodes, let refTranscript = referenceTranscript, !refCodes.isEmpty, let refTime = refCodes.first?.count, refTime > 0 {
                    // ICL voice cloning: all text (ref + target) goes in prefill with proper
                    // codec_pad/tts_pad overlays. During generation, trailing_text_hidden is
                    // just tts_pad. Matches mlx-audio's _prepare_icl_generation_inputs
                    // non-streaming layout.

                    let refChat = "<|im_start|>assistant\n\(refTranscript)<|im_end|>\n"
                    let refIdsArray: [Int32] = tokenizer.encode(text: refChat)
                    guard refIdsArray.count > 5 else {
                        continuation.finish()
                        return
                    }
                    let refTextIds = Array(refIdsArray[3..<(refIdsArray.count - 2)])

                    guard chatIdsRaw.count > 8 else {
                        continuation.finish()
                        return
                    }
                    let targetTextIds = Array(chatIdsRaw[3..<(chatIdsRaw.count - 5)])

                    let combinedTextIds = refTextIds + targetTextIds
                    let combinedTextArray = MLXArray(combinedTextIds).expandedDimensions(axis: 0)
                    var iclTextEmbed = model.text_projection(model.text_embedding(combinedTextArray))
                    iclTextEmbed = concatenated([iclTextEmbed, ttsEosEmbed], axis: 1)
                    let textLens = iclTextEmbed.shape[1]

                    let firstCbCodes = MLXArray(refCodes[0]).expandedDimensions(axis: 0)
                    var refCodecEmbed = model.codec_embedding(firstCbCodes)
                    for i in 0..<(numCodeGroups - 1) {
                        guard i + 1 < refCodes.count else { break }
                        guard i < model.code_predictor.codec_embedding.count else { break }
                        let cbCodes = MLXArray(refCodes[i + 1]).expandedDimensions(axis: 0)
                        refCodecEmbed = refCodecEmbed + model.code_predictor.codec_embedding[i](cbCodes)
                    }

                    let codecBosEmbed = model.codec_embedding(MLXArray([Int32(config.codec_bos_id)]).expandedDimensions(axis: 0))
                    let codecEmbedIcl = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
                    let codecLens = codecEmbedIcl.shape[1]

                    let codecPadEmbed = model.codec_embedding(MLXArray([Int32(config.codec_pad_id)]).expandedDimensions(axis: 0))
                    let textWithCodecPad = iclTextEmbed + tiled(codecPadEmbed, repetitions: [1, textLens, 1])
                    let codecWithTtsPad = codecEmbedIcl + tiled(ttsPadEmbed, repetitions: [1, codecLens, 1])
                    let iclInputEmbed = concatenated([textWithCodecPad, codecWithTtsPad], axis: 1)

                    inputEmbeds = concatenated([roleEmbed, combinedEmbed, iclInputEmbed], axis: 1)
                    trailingTextHidden = ttsPadEmbed
                } else {
                    var instructEmbed: MLXArray? = nil
                    if let instructText = instruct, !instructText.isEmpty {
                        // Explicit instruct text (VoiceDesign or CustomVoice mode)
                        let formatted = "<|im_start|>user\n\(instructText)<|im_end|>\n"
                        let instructIdsArray: [Int32] = tokenizer.encode(text: formatted)
                        let instructIds = MLXArray(instructIdsArray).expandedDimensions(axis: 0)
                        instructEmbed = model.text_projection(model.text_embedding(instructIds))
                    } else if !prompt.isEmpty && speakerId == nil && speakerEmbedding == nil {
                        // Backward compat: treat prompt as instruct when no speaker resolved
                        let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n"
                        let instructIdsArray: [Int32] = tokenizer.encode(text: formatted)
                        let instructIds = MLXArray(instructIdsArray).expandedDimensions(axis: 0)
                        instructEmbed = model.text_projection(model.text_embedding(instructIds))
                    }

                    if let instructEmbedding = instructEmbed {
                        inputEmbeds = concatenated([instructEmbedding, roleEmbed, combinedEmbed], axis: 1)
                    } else {
                        inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
                    }

                    let firstTextEmbed = model.text_projection(model.text_embedding(inputIds[0..., 3..<4])) + codecEmbed[0..., (codecEmbed.shape[1]-1)..., 0...]
                    inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

                    let trailingLen = inputIds.shape[1] - 4 - 5
                    if trailingLen > 0 {
                        trailingTextHidden = model.text_projection(model.text_embedding(inputIds[0..., 4..<(inputIds.shape[1]-5)]))
                        trailingTextHidden = concatenated([trailingTextHidden, ttsEosEmbed], axis: 1)
                    } else {
                        trailingTextHidden = ttsEosEmbed
                    }
                }

                var (h, cache) = model.callAsFunction(inputEmbeds, cache: nil, positionOffset: nil)
                var positionOffset = inputEmbeds.shape[1]

                var chunkCodes: [[Int32]] = []
                let eosTokenId: Int32 = Int32(config.codec_eos_token_id)
                let padTokenId: Int32 = Int32(config.codec_pad_id)
                var trailingIdx = 0
                var consecutivePad = 0

                var logits = model.codec_head(h)

                var generatedCode0Tokens: [Int32] = []
                var generatedCode0TokensSet: Set<Int32> = []

                // Pre-compute EOS/pad mask once — vocabSize is constant across steps
                let streamVocabSize = logits.shape.last!
                var streamEosMaskValues = [Float](repeating: 0.0, count: streamVocabSize)
                if Int(eosTokenId) < streamVocabSize { streamEosMaskValues[Int(eosTokenId)] = -Float.infinity }
                if Int(padTokenId) < streamVocabSize { streamEosMaskValues[Int(padTokenId)] = -Float.infinity }
                let streamEosMaskArray = MLXArray(streamEosMaskValues).expandedDimensions(axis: 0)

                let totalTextTokens = trailingTextHidden.shape[1]

                for step in 0..<maxTokens {
                    if Task.isCancelled {
                        break
                    }

                    let hasRemainingText = trailingIdx < totalTextTokens

                    var samplingLogits = logits
                    if hasRemainingText {
                        samplingLogits = logits + streamEosMaskArray
                    }

                    let nextToken = model.sampleToken(
                        logits: samplingLogits,
                        temperature: temperature,
                        topK: code0TopK,
                        repetitionPenalty: effectiveRepPenalty,
                        generatedTokenSet: generatedCode0TokensSet.isEmpty ? nil : generatedCode0TokensSet
                    )
                    let code0Value = nextToken[0].item(Int32.self)

                    if code0Value == eosTokenId {
                        break
                    } else if code0Value == padTokenId {
                        consecutivePad += 1
                        if consecutivePad > 6 {
                            break
                        }
                    } else {
                        consecutivePad = 0
                    }

                    var codeTokens: [Int32] = [code0Value]
                    let nextTokenArray = nextToken.expandedDimensions(axis: 0)
                    let codeHidden = h[0..., (h.shape[1]-1)..<h.shape[1], 0...]
                    var codePredictorCache: [KVCache]? = nil

                    for codeIdx in 0..<(numCodeGroups - 1) {
                        let codeInput: MLXArray

                        if codeIdx == 0 {
                            let code0Embed = model.codec_embedding(nextTokenArray)
                            codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                        } else {
                            guard codeIdx < codeTokens.count else { break }
                            guard codeIdx - 1 < model.code_predictor.codec_embedding.count else { break }
                            let prevCode = MLXArray([codeTokens[codeIdx]]).expandedDimensions(axis: 0)
                            codeInput = model.code_predictor.codec_embedding[codeIdx - 1](prevCode)
                        }

                        let (codeLogits, newCodeCache) = model.code_predictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                        codePredictorCache = newCodeCache

                        let codeToken = model.sampleToken(logits: codeLogits, temperature: resolvedDetailTemp)
                        let codeValue = codeToken[0].item(Int32.self)
                        codeTokens.append(codeValue)
                    }

                    chunkCodes.append(codeTokens)
                    generatedCode0Tokens.append(code0Value)
                    generatedCode0TokensSet.insert(code0Value)
                    codePredictorCache = nil

                    if chunkCodes.count >= chunkSize {
                        continuation.yield(chunkCodes)
                        chunkCodes = []
                        Memory.clearCache()
                    }

                    let textEmbed: MLXArray
                    if trailingIdx < trailingTextHidden.shape[1] {
                        textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                        trailingIdx += 1
                    } else {
                        textEmbed = ttsPadEmbed
                    }

                    var codecEmbedSum = model.codec_embedding(nextTokenArray)
                    for i in 0..<(numCodeGroups - 1) {
                        guard i + 1 < codeTokens.count else { break }
                        guard i < model.code_predictor.codec_embedding.count else { break }
                        let codeVal = MLXArray([codeTokens[i + 1]]).expandedDimensions(axis: 0)
                        codecEmbedSum = codecEmbedSum + model.code_predictor.codec_embedding[i](codeVal)
                    }
                    eval(codecEmbedSum)

                    inputEmbeds = textEmbed + codecEmbedSum
                    eval(inputEmbeds)

                    let (hStep, cStep) = model.callAsFunction(inputEmbeds, cache: cache, positionOffset: positionOffset)
                    h = hStep
                    cache = cStep
                    logits = model.codec_head(h)
                    positionOffset += 1

                    if (step + 1) % 15 == 0 {
                        // ICL prefill is much longer than maxKVCacheWindow; trimming
                        // evicts the voice conditioning. Match the Python mlx-audio
                        // streaming path which does not trim.
                        if !useICL {
                            cache = trimKVCache(cache, maxWindow: maxKVCacheWindow)
                        }
                        eval(h, logits)
                        Stream.defaultStream(.gpu).synchronize()
                        Memory.clearCache()
                    }
                }

                if !chunkCodes.isEmpty {
                    continuation.yield(chunkCodes)
                }

                cache = []
                h = MLXArray([])
                logits = MLXArray([])
                Stream.defaultStream(.gpu).synchronize()
                inputEmbeds = MLXArray([])
                Memory.clearCache()

                continuation.finish()
            }
        }
    }
}
