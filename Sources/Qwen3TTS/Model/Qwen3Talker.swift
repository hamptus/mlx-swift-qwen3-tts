import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Qwen3 Talker Model

public class Qwen3Talker: Module {
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
        generatedTokens: [Int32]? = nil
    ) -> MLXArray {
        var logits = logits

        if logits.shape.count == 3 {
            logits = logits[0..., (logits.shape[1] - 1)..<logits.shape[1], 0...].squeezed(axis: 1)
        }

        if let tokens = generatedTokens, !tokens.isEmpty, repetitionPenalty != 1.0 {
            let vocabSize = logits.shape.last!
            var penaltyValues = [Float](repeating: 1.0, count: vocabSize)
            let uniqueTokens = Set(tokens)
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
        maxTokens: Int = 1200
    ) -> [[Int32]] {
        let useICL = referenceAudioCodes != nil && referenceTranscript != nil && !referenceTranscript!.isEmpty
        let speakerName = prompt.lowercased()
        let speakerId = config.spk_id[speakerName]
        print("DEBUG [generateCodes]: entry prompt='\(prompt.prefix(30))' text='\(text.prefix(30))' speakerId=\(speakerId as Any) spkEmbed=\(speakerEmbedding?.shape ?? []) useICL=\(useICL)"); fflush(stdout)

        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = tokenizer.encode(text: chatText).asType(.int32)
        print("DEBUG [generateCodes]: inputIds shape=\(inputIds.shape)"); fflush(stdout)

        let minTokens = 9
        guard inputIds.shape[1] >= minTokens else {
            print("DEBUG [generateCodes]: input too short (\(inputIds.shape[1]) < \(minTokens))")
            return []
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

        var instructEmbed: MLXArray? = nil
        if let instructText = instruct, !instructText.isEmpty {
            // Explicit instruct text (VoiceDesign or CustomVoice mode)
            let formatted = "<|im_start|>user\n\(instructText)<|im_end|>\n"
            let instructIdsArray: [Int32] = tokenizer.encode(text: formatted)
            let instructIds = MLXArray(instructIdsArray).expandedDimensions(axis: 0)
            instructEmbed = text_projection(text_embedding(instructIds))
        } else if useICL, let refCodes = referenceAudioCodes, let refTranscript = referenceTranscript {
            let refText = "<|im_start|>user\n\(refTranscript)<|im_end|>\n"
            let refTextIds: [Int32] = tokenizer.encode(text: refText)
            let refTextEmbed = text_projection(text_embedding(MLXArray(refTextIds).expandedDimensions(axis: 0)))

            let numFrames = refCodes.first?.count ?? 0
            if numFrames > 0 && !refCodes.isEmpty {
                let semanticCodes = MLXArray(refCodes[0]).expandedDimensions(axis: 0)
                let refAudioEmbed = codec_embedding(semanticCodes)
                instructEmbed = concatenated([refTextEmbed, refAudioEmbed], axis: 1)
            } else {
                instructEmbed = refTextEmbed
            }
        } else if !prompt.isEmpty && speakerId == nil && speakerEmbedding == nil {
            // Backward compat: treat prompt as instruct when no speaker resolved
            let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n"
            let instructIdsArray: [Int32] = tokenizer.encode(text: formatted)
            let instructIds = MLXArray(instructIdsArray).expandedDimensions(axis: 0)
            instructEmbed = text_projection(text_embedding(instructIds))
        }

        var inputEmbeds: MLXArray
        if let instructEmbedding = instructEmbed {
            inputEmbeds = concatenated([instructEmbedding, roleEmbed, combinedEmbed], axis: 1)
        } else {
            inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        let firstTextEmbed = text_projection(text_embedding(inputIds[0..., 3..<4])) + codecEmbed[0..., (codecEmbed.shape[1]-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        let trailingLen = inputIds.shape[1] - 4 - 5
        var trailingTextHidden: MLXArray
        if trailingLen > 0 {
            trailingTextHidden = text_projection(text_embedding(inputIds[0..., 4..<(inputIds.shape[1]-5)]))
            trailingTextHidden = concatenated([trailingTextHidden, ttsEosEmbed], axis: 1)
        } else {
            trailingTextHidden = ttsEosEmbed
        }

        print("DEBUG [generateCodes]: inputEmbeds shape=\(inputEmbeds.shape) trailingLen=\(trailingLen)")
        var (h, cache) = self.callAsFunction(inputEmbeds, cache: nil, positionOffset: nil)
        var positionOffset = inputEmbeds.shape[1]
        print("DEBUG [generateCodes]: prefill done, h shape=\(h.shape), positionOffset=\(positionOffset)")

        var generatedCodes: [[Int32]] = []
        let eosTokenId: Int32 = Int32(config.codec_eos_token_id)
        let padTokenId: Int32 = Int32(config.codec_pad_id)
        var trailingIdx = 0
        var consecutivePad = 0
        let numCodeGroups = config.code_predictor_config.num_code_groups
        print("DEBUG [generateCodes]: numCodeGroups=\(numCodeGroups), code_predictor embeddings=\(code_predictor.codec_embedding.count), lm_heads=\(code_predictor.lm_head.count)")

        var logits = codec_head(h)

        var generatedCode0Tokens: [Int32] = []
        var generatedCodePredictorTokens: [[Int32]] = Array(repeating: [], count: numCodeGroups - 1)

        let totalTextTokens = trailingTextHidden.shape[1]

        for step in 0..<maxTokens {
            if Task.isCancelled { break }
            if step < 3 || step % 50 == 0 {
                print("DEBUG [generateCodes]: step \(step), logits shape=\(logits.shape), trailingIdx=\(trailingIdx)/\(totalTextTokens)")
            }

            let hasRemainingText = trailingIdx < totalTextTokens

            var samplingLogits = logits
            if hasRemainingText {
                let vocabSize = logits.shape.last!
                if vocabSize > Int(eosTokenId) {
                    var maskValues = [Float](repeating: 0.0, count: vocabSize)
                    maskValues[Int(eosTokenId)] = -Float.infinity
                    if Int(padTokenId) < vocabSize {
                        maskValues[Int(padTokenId)] = -Float.infinity
                    }
                    let maskArray = MLXArray(maskValues).expandedDimensions(axis: 0)
                    samplingLogits = logits + maskArray
                }
            }

            let nextToken = sampleToken(
                logits: samplingLogits,
                temperature: temperature,
                generatedTokens: generatedCode0Tokens.isEmpty ? nil : generatedCode0Tokens
            )
            let code0Value = nextToken[0].item(Int32.self)
            if step < 3 { print("DEBUG [generateCodes]: step \(step) code0=\(code0Value)") }

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
                    guard codeIdx < codeTokens.count else {
                        print("CRASH AVOIDED [generateCodes]: codeIdx=\(codeIdx) >= codeTokens.count=\(codeTokens.count)")
                        break
                    }
                    guard codeIdx - 1 < code_predictor.codec_embedding.count else {
                        print("CRASH AVOIDED [generateCodes]: codeIdx-1=\(codeIdx-1) >= codec_embedding.count=\(code_predictor.codec_embedding.count)")
                        break
                    }
                    let prevCode = MLXArray([codeTokens[codeIdx]]).expandedDimensions(axis: 0)
                    codeInput = code_predictor.codec_embedding[codeIdx - 1](prevCode)
                }
                let (codeLogits, newCodeCache) = code_predictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                codePredictorCache = newCodeCache

                let codebookHistory = generatedCodePredictorTokens[codeIdx]
                let codeToken = sampleToken(
                    logits: codeLogits,
                    temperature: temperature,
                    generatedTokens: codebookHistory.isEmpty ? nil : codebookHistory
                )
                let codeValue = codeToken[0].item(Int32.self)
                codeTokens.append(codeValue)
                generatedCodePredictorTokens[codeIdx].append(codeValue)
            }

            if step < 3 { print("DEBUG [generateCodes]: step \(step) codeTokens=\(codeTokens.count) values=\(codeTokens.prefix(4))") }
            generatedCodes.append(codeTokens)
            generatedCode0Tokens.append(code0Value)
            codePredictorCache = nil

            if (step + 1) % 10 == 0 {
                Memory.clearCache()
            }

            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.shape[1] {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            var codecEmbedSum = codec_embedding(nextTokenArray)
            for i in 0..<(numCodeGroups - 1) {
                guard i + 1 < codeTokens.count else {
                    print("CRASH AVOIDED [generateCodes embed]: i+1=\(i+1) >= codeTokens.count=\(codeTokens.count)")
                    break
                }
                guard i < code_predictor.codec_embedding.count else {
                    print("CRASH AVOIDED [generateCodes embed]: i=\(i) >= codec_embedding.count=\(code_predictor.codec_embedding.count)")
                    break
                }
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
                cache = trimKVCache(cache, maxWindow: maxKVCacheWindow)
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
        tokenizer: Qwen3Tokenizer,
        temperature: Float = 0.9,
        maxTokens: Int = 1200,
        chunkSize: Int = 12
    ) -> AsyncThrowingStream<[[Int32]], Error> {
        let model = self
        let config = self.config

        return AsyncThrowingStream<[[Int32]], Error> { (continuation: AsyncThrowingStream<[[Int32]], Error>.Continuation) in
            Task {
                let speakerName = prompt.lowercased()
                let speakerId = config.spk_id[speakerName]

                let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
                let inputIds = tokenizer.encode(text: chatText).asType(.int32)

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

                var inputEmbeds: MLXArray
                if let instructEmbedding = instructEmbed {
                    inputEmbeds = concatenated([instructEmbedding, roleEmbed, combinedEmbed], axis: 1)
                } else {
                    inputEmbeds = concatenated([roleEmbed, combinedEmbed], axis: 1)
                }

                let firstTextEmbed = model.text_projection(model.text_embedding(inputIds[0..., 3..<4])) + codecEmbed[0..., (codecEmbed.shape[1]-1)..., 0...]
                inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

                let trailingLen = inputIds.shape[1] - 4 - 5
                var trailingTextHidden: MLXArray
                if trailingLen > 0 {
                    trailingTextHidden = model.text_projection(model.text_embedding(inputIds[0..., 4..<(inputIds.shape[1]-5)]))
                    trailingTextHidden = concatenated([trailingTextHidden, ttsEosEmbed], axis: 1)
                } else {
                    trailingTextHidden = ttsEosEmbed
                }

                var (h, cache) = model.callAsFunction(inputEmbeds, cache: nil, positionOffset: nil)
                var positionOffset = inputEmbeds.shape[1]

                var chunkCodes: [[Int32]] = []
                let eosTokenId: Int32 = Int32(config.codec_eos_token_id)
                let padTokenId: Int32 = Int32(config.codec_pad_id)
                var trailingIdx = 0
                var consecutivePad = 0
                let numCodeGroups = config.code_predictor_config.num_code_groups

                var logits = model.codec_head(h)

                var generatedCode0Tokens: [Int32] = []
                let totalTextTokens = trailingTextHidden.shape[1]

                for step in 0..<maxTokens {
                    if Task.isCancelled {
                        break
                    }

                    let hasRemainingText = trailingIdx < totalTextTokens

                    var samplingLogits = logits
                    if hasRemainingText {
                        let vocabSize = logits.shape.last!
                        if vocabSize > Int(eosTokenId) {
                            var maskValues = [Float](repeating: 0.0, count: vocabSize)
                            maskValues[Int(eosTokenId)] = -Float.infinity
                            maskValues[Int(padTokenId)] = -Float.infinity
                            let maskArray = MLXArray(maskValues).expandedDimensions(axis: 0)
                            samplingLogits = logits + maskArray
                        }
                    }

                    let nextToken = model.sampleToken(
                        logits: samplingLogits,
                        temperature: temperature,
                        generatedTokens: generatedCode0Tokens.isEmpty ? nil : generatedCode0Tokens
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
                            guard codeIdx < codeTokens.count else {
                                print("CRASH AVOIDED [streaming]: codeIdx=\(codeIdx) >= codeTokens.count=\(codeTokens.count)")
                                break
                            }
                            guard codeIdx - 1 < model.code_predictor.codec_embedding.count else {
                                print("CRASH AVOIDED [streaming]: codeIdx-1=\(codeIdx-1) >= codec_embedding.count=\(model.code_predictor.codec_embedding.count)")
                                break
                            }
                            let prevCode = MLXArray([codeTokens[codeIdx]]).expandedDimensions(axis: 0)
                            codeInput = model.code_predictor.codec_embedding[codeIdx - 1](prevCode)
                        }

                        let (codeLogits, newCodeCache) = model.code_predictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                        codePredictorCache = newCodeCache

                        let codeToken = model.sampleToken(logits: codeLogits, temperature: temperature)
                        let codeValue = codeToken[0].item(Int32.self)
                        codeTokens.append(codeValue)
                    }

                    chunkCodes.append(codeTokens)
                    generatedCode0Tokens.append(code0Value)
                    codePredictorCache = nil
                    if (step + 1) % 10 == 0 {
                        Memory.clearCache()
                    }

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
                        guard i + 1 < codeTokens.count else {
                            print("CRASH AVOIDED [streaming embed]: i+1=\(i+1) >= codeTokens.count=\(codeTokens.count)")
                            break
                        }
                        guard i < model.code_predictor.codec_embedding.count else {
                            print("CRASH AVOIDED [streaming embed]: i=\(i) >= codec_embedding.count=\(model.code_predictor.codec_embedding.count)")
                            break
                        }
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
                        cache = trimKVCache(cache, maxWindow: maxKVCacheWindow)
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
