import Foundation
import MLX
import MLXNN

/// A chunk of generated audio for streaming playback.
public struct AudioChunk: Sendable {
    /// Audio samples (Float, 24kHz)
    public let samples: [Float]
    /// Range of tokens in this chunk
    public let tokenRange: Range<Int>
    /// Whether this is the final chunk
    public let isFinal: Bool

    public init(samples: [Float], tokenRange: Range<Int>, isFinal: Bool) {
        self.samples = samples
        self.tokenRange = tokenRange
        self.isFinal = isFinal
    }
}

/// Configuration for the Qwen3 TTS pipeline.
public struct Qwen3TTSPipelineConfiguration: Sendable {
    /// Whether to apply mixed quantization at runtime (for non-pre-quantized models).
    /// Pre-quantized models (those with `quantization` in config.json) ignore this.
    public var applyRuntimeQuantization: Bool

    /// Default generation temperature (0.0-1.0). Higher = more varied.
    public var defaultTemperature: Float

    /// Maximum tokens per generation chunk (~12 tokens = 1 second at 12Hz).
    public var defaultMaxTokens: Int

    /// Default chunk size for streaming (frames per yield).
    public var defaultStreamingChunkSize: Int

    /// Number of samples to crossfade between text chunks (default 480 = 20ms at 24kHz).
    public var crossfadeSamples: Int

    public init(
        applyRuntimeQuantization: Bool = true,
        defaultTemperature: Float = 0.85,
        defaultMaxTokens: Int = 2400,
        defaultStreamingChunkSize: Int = 12,
        crossfadeSamples: Int = 480
    ) {
        self.applyRuntimeQuantization = applyRuntimeQuantization
        self.defaultTemperature = defaultTemperature
        self.defaultMaxTokens = defaultMaxTokens
        self.defaultStreamingChunkSize = defaultStreamingChunkSize
        self.crossfadeSamples = crossfadeSamples
    }

    public static let `default` = Qwen3TTSPipelineConfiguration()
}

/// High-level pipeline for Qwen3 text-to-speech generation.
///
/// Usage:
/// ```swift
/// let pipeline = try Qwen3TTSPipeline(modelPath: modelURL)
/// let samples = pipeline.generate(text: "Hello world", speaker: "Aiden")
/// ```
public final class Qwen3TTSPipeline: @unchecked Sendable {
    /// Audio sample rate in Hz.
    public static let sampleRate: Int = 24000

    private let model: Qwen3Talker
    private let tokenizer: Qwen3Tokenizer
    private let decoder: AudioDecoder
    private let speakerEncoder: SpeakerEncoder?
    private let audioEncoder: Qwen3TTSAudioEncoder?
    private let config: Qwen3TTSConfig
    private let device: Device
    private let pipelineConfig: Qwen3TTSPipelineConfiguration

    /// Available built-in speaker names (from the model's spk_id map).
    public var availableSpeakers: [String] {
        config.spk_id.keys.sorted()
    }

    /// Whether voice cloning via speaker embeddings is supported.
    public var supportsVoiceCloning: Bool {
        speakerEncoder?.isWeightsLoaded ?? false
    }

    /// Whether ICL (in-context learning) audio encoding is available.
    public var supportsICL: Bool {
        audioEncoder != nil
    }

    /// Load a Qwen3 TTS model from a local directory.
    ///
    /// The directory must contain:
    /// - `config.json` — model configuration
    /// - `model.safetensors` — model weights
    /// - `tokenizer.json` — BPE tokenizer
    /// - `speech_tokenizer/` — vocoder subdirectory with its own config.json and model.safetensors
    ///
    /// - Parameters:
    ///   - modelPath: Path to the model directory
    ///   - configuration: Pipeline configuration options
    /// - Throws: If any required file is missing or loading fails
    public init(modelPath: URL, configuration: Qwen3TTSPipelineConfiguration = .default) throws {
        self.pipelineConfig = configuration

        let device = DeviceSelector.resolveDevice()
        self.device = device

        // Load in the resolved device context
        let result: (Qwen3Talker, Qwen3Tokenizer, AudioDecoder, SpeakerEncoder?, Qwen3TTSAudioEncoder?, Qwen3TTSConfig) = try Device.withDefaultDevice(device) {
            // Parse config
            let configURL = modelPath.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw Qwen3TTSError.fileNotFound("config.json")
            }
            let configData = try Data(contentsOf: configURL)
            let modelConfig = try JSONDecoder().decode(Qwen3TTSConfig.self, from: configData)

            // Load tokenizer
            let tokenizer = Qwen3Tokenizer(modelPath: modelPath)

            // Load model weights (always on CPU first)
            let weightsURL = modelPath.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: weightsURL.path) else {
                throw Qwen3TTSError.fileNotFound("model.safetensors")
            }
            var weights = try MLX.loadArrays(url: weightsURL, stream: .cpu)

            let model = Qwen3Talker(config: modelConfig)

            // Evaluate random init weights, then load actual weights
            eval(model.parameters())
            DeviceSelector.synchronizeIfNeeded(device: device)
            Memory.clearCache()

            model.load(weights: weights)
            DeviceSelector.synchronizeIfNeeded(device: device)
            Memory.clearCache()

            // Load speaker encoder if weights are present
            var speakerEncoder: SpeakerEncoder? = nil
            let hasSpeakerEncoderWeights = weights.keys.contains { $0.hasPrefix("speaker_encoder.") }
            if hasSpeakerEncoderWeights {
                let spkEncoder = SpeakerEncoder()
                eval(spkEncoder.parameters())
                Memory.clearCache()

                spkEncoder.load(weights: weights)
                DeviceSelector.synchronizeIfNeeded(device: device)
                Memory.clearCache()

                if spkEncoder.isWeightsLoaded {
                    speakerEncoder = spkEncoder
                }
            }

            // Materialize all weights in GPU memory before freeing source
            eval(model.parameters())
            if let spkEnc = speakerEncoder {
                eval(spkEnc.parameters())
            }
            DeviceSelector.synchronizeIfNeeded(device: device)

            // Free original weights
            weights = [:]
            Memory.clearCache()

            // Apply runtime quantization if needed
            if modelConfig.quantization == nil && configuration.applyRuntimeQuantization {
                Qwen3TTSPipeline.applyMixedQuantization(to: model)
                DeviceSelector.synchronizeIfNeeded(device: device)
                Memory.clearCache()
            }

            // Load speech tokenizer (vocoder)
            let speechTokenizerURL = modelPath.appendingPathComponent("speech_tokenizer")
            let speechConfigCandidates = [
                speechTokenizerURL.appendingPathComponent("config.json"),
                speechTokenizerURL.appendingPathComponent("configuration.json"),
                speechTokenizerURL.appendingPathComponent("speech_tokenizer_config.json")
            ]
            guard let speechConfigURL = speechConfigCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
                throw Qwen3TTSError.fileNotFound("speech_tokenizer/config.json")
            }
            let audioConfigData = try Data(contentsOf: speechConfigURL)
            let audioConfig = try JSONDecoder().decode(AudioDecoderConfig.self, from: audioConfigData)

            let speechWeightsURL = speechTokenizerURL.appendingPathComponent("model.safetensors")
            let decoder = AudioDecoder(config: audioConfig)
            let mlxLoaded = decoder.loadMLXDecoder(configURL: speechConfigURL, weightsURL: speechWeightsURL)
            guard mlxLoaded else {
                throw Qwen3TTSError.decoderLoadFailed
            }

            // Load audio encoder for ICL (optional)
            var audioEncoder: Qwen3TTSAudioEncoder? = nil
            do {
                let encoder = Qwen3TTSAudioEncoder()
                try encoder.loadWeights(from: speechWeightsURL)
                audioEncoder = encoder
            } catch {
                // ICL mode unavailable
            }

            DeviceSelector.synchronizeIfNeeded(device: device)
            Memory.clearCache()

            return (model, tokenizer, decoder, speakerEncoder, audioEncoder, modelConfig)
        }

        self.model = result.0
        self.tokenizer = result.1
        self.decoder = result.2
        self.speakerEncoder = result.3
        self.audioEncoder = result.4
        self.config = result.5
    }

    // MARK: - Simple Generation

    /// Generate speech from text using a built-in speaker name.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - speaker: Speaker name (e.g., "Aiden", "Serena"). Case-insensitive.
    ///   - temperature: Generation temperature (default from configuration)
    ///   - maxTokens: Maximum tokens to generate (default from configuration)
    /// - Returns: Audio samples at 24kHz
    public func generate(
        text: String,
        speaker: String,
        temperature: Float? = nil,
        maxTokens: Int? = nil
    ) -> [Float] {
        let temp = temperature ?? pipelineConfig.defaultTemperature
        let tokens = maxTokens ?? pipelineConfig.defaultMaxTokens

        return Device.withDefaultDevice(device) {
            defer {
                model.clearGenerationCache()
                decoder.clearCompiledCache()
                DeviceSelector.synchronizeIfNeeded(device: device)
                Memory.clearCache()
            }
            return model.generate(
                prompt: speaker,
                text: text,
                tokenizer: tokenizer,
                decoder: decoder,
                temperature: temp,
                maxTokens: tokens
            )
        }
    }

    /// Generate speech from text using a speaker embedding (voice cloning).
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - speakerEmbedding: 1024-dimensional speaker embedding
    ///   - temperature: Generation temperature (default from configuration)
    ///   - maxTokens: Maximum tokens to generate (default from configuration)
    /// - Returns: Audio samples at 24kHz
    public func generate(
        text: String,
        speakerEmbedding: [Float],
        temperature: Float? = nil,
        maxTokens: Int? = nil
    ) -> [Float] {
        let temp = temperature ?? pipelineConfig.defaultTemperature
        let tokens = maxTokens ?? pipelineConfig.defaultMaxTokens

        return Device.withDefaultDevice(device) {
            defer {
                model.clearGenerationCache()
                decoder.clearCompiledCache()
                DeviceSelector.synchronizeIfNeeded(device: device)
                Memory.clearCache()
            }
            let embed = MLXArray(speakerEmbedding)
            return model.generate(
                prompt: "",
                text: text,
                speakerEmbedding: embed,
                tokenizer: tokenizer,
                decoder: decoder,
                temperature: temp,
                maxTokens: tokens
            )
        }
    }

    // MARK: - Streaming Generation

    /// Stream audio chunks for low-latency playback.
    ///
    /// Uses buffer-and-batch approach: accumulates codes until a decode chunk is ready,
    /// then decodes and yields audio samples.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - speaker: Speaker name (case-insensitive)
    ///   - speakerEmbedding: Optional speaker embedding for voice cloning (overrides speaker name)
    ///   - temperature: Generation temperature
    ///   - maxTokens: Maximum tokens to generate
    ///   - chunkSize: Frames per streaming yield
    /// - Returns: AsyncThrowingStream of AudioChunk
    public func generateStream(
        text: String,
        speaker: String = "",
        speakerEmbedding: [Float]? = nil,
        temperature: Float? = nil,
        maxTokens: Int? = nil,
        chunkSize: Int? = nil
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let temp = temperature ?? pipelineConfig.defaultTemperature
        let tokens = maxTokens ?? pipelineConfig.defaultMaxTokens
        let chunk = chunkSize ?? pipelineConfig.defaultStreamingChunkSize
        let numCodeGroups = config.code_predictor_config.num_code_groups

        let capturedModel = model
        let capturedTokenizer = tokenizer
        let capturedDecoder = decoder
        let capturedDevice = device

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let speakerEmbed = speakerEmbedding.map { MLXArray($0) }
                    let codeStream = Device.withDefaultDevice(capturedDevice) {
                        capturedModel.generateStream(
                            prompt: speaker,
                            text: text,
                            speakerEmbedding: speakerEmbed,
                            tokenizer: capturedTokenizer,
                            temperature: temp,
                            maxTokens: tokens,
                            chunkSize: chunk
                        )
                    }

                    let DECODE_CHUNK_SIZE = 18
                    let LEFT_CONTEXT_SIZE = 8
                    let SAMPLES_PER_FRAME = 1920

                    var codeBuffer: [[Int32]] = []
                    var leftContext: [[Int32]] = []
                    var isFirstDecode = true
                    var totalCodesProcessed = 0

                    func decodeBatch(codes: [[Int32]], isFinal: Bool) -> [Float] {
                        guard !codes.isEmpty else { return [] }

                        var decodeInput: [[Int32]]
                        if isFirstDecode {
                            decodeInput = codes
                            isFirstDecode = false
                        } else {
                            decodeInput = leftContext + codes
                        }

                        let allSamples = Device.withDefaultDevice(capturedDevice) { () -> [Float] in
                            defer {
                                DeviceSelector.synchronizeIfNeeded(device: capturedDevice)
                                Memory.clearCache()
                            }
                            let flatCodes: [Int32] = decodeInput.flatMap { $0 }
                            let codesArray = MLXArray(flatCodes).reshaped([1, decodeInput.count, numCodeGroups])
                            let audio = capturedDecoder.mlxDecode(codes: codesArray)
                            let flatAudio = audio.reshaped([-1])
                            eval(flatAudio)
                            return flatAudio.asArray(Float.self)
                        }

                        let contextSamplesToDrop = leftContext.count * SAMPLES_PER_FRAME
                        var newSamples: [Float]
                        if contextSamplesToDrop > 0 && allSamples.count > contextSamplesToDrop {
                            newSamples = Array(allSamples.dropFirst(contextSamplesToDrop))
                        } else {
                            newSamples = allSamples
                        }

                        leftContext = Array(codes.suffix(LEFT_CONTEXT_SIZE))
                        return newSamples
                    }

                    func cleanSamples(_ samples: [Float]) -> [Float] {
                        samples.map { val -> Float in
                            if val.isNaN || val.isInfinite { return 0.0 }
                            return max(-1.0, min(1.0, val))
                        }
                    }

                    for try await codes in codeStream {
                        if Task.isCancelled { break }
                        guard !codes.isEmpty else { continue }

                        let validCodes = codes.filter { frame in
                            guard let firstCode = frame.first else { return false }
                            return firstCode >= 0 && firstCode < 2048
                        }
                        guard !validCodes.isEmpty else { continue }

                        codeBuffer.append(contentsOf: validCodes)

                        while codeBuffer.count >= DECODE_CHUNK_SIZE {
                            let batchChunk = Array(codeBuffer.prefix(DECODE_CHUNK_SIZE))
                            codeBuffer = Array(codeBuffer.dropFirst(DECODE_CHUNK_SIZE))

                            let samples = decodeBatch(codes: batchChunk, isFinal: false)
                            totalCodesProcessed += batchChunk.count

                            guard !samples.isEmpty else { continue }
                            let tokenRange = (totalCodesProcessed - batchChunk.count)..<totalCodesProcessed
                            continuation.yield(AudioChunk(samples: cleanSamples(samples), tokenRange: tokenRange, isFinal: false))
                        }
                    }

                    // Flush remaining codes
                    if !codeBuffer.isEmpty {
                        let samples = decodeBatch(codes: codeBuffer, isFinal: true)
                        totalCodesProcessed += codeBuffer.count
                        if !samples.isEmpty {
                            let tokenRange = (totalCodesProcessed - codeBuffer.count)..<totalCodesProcessed
                            continuation.yield(AudioChunk(samples: cleanSamples(samples), tokenRange: tokenRange, isFinal: true))
                        }
                    }

                    continuation.yield(AudioChunk(samples: [], tokenRange: totalCodesProcessed..<totalCodesProcessed, isFinal: true))

                    capturedModel.clearGenerationCache()
                    capturedDecoder.clearCompiledCache()
                    Stream.defaultStream(.gpu).synchronize()
                    Memory.clearCache()

                    continuation.finish()
                } catch {
                    capturedModel.clearGenerationCache()
                    capturedDecoder.clearCompiledCache()
                    Stream.defaultStream(.gpu).synchronize()
                    Memory.clearCache()
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - File Output (Memory-Efficient)

    /// Generate speech and write directly to a WAV file.
    ///
    /// This method is memory-efficient for long text: it chunks the text at natural boundaries,
    /// generates each chunk independently, and writes incrementally to disk.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize (any length)
    ///   - speaker: Speaker name
    ///   - speakerEmbedding: Optional speaker embedding for voice cloning
    ///   - referenceTranscript: Optional transcript for ICL voice cloning
    ///   - referenceAudioCodes: Optional pre-encoded reference audio for ICL
    ///   - outputURL: File URL to write the WAV to
    ///   - temperature: Generation temperature
    ///   - onProgress: Optional progress callback (0.0 to 1.0)
    /// - Returns: Total number of samples written
    /// - Throws: If writing fails
    public func generateToFile(
        text: String,
        speaker: String = "",
        speakerEmbedding: [Float]? = nil,
        referenceTranscript: String? = nil,
        referenceAudioCodes: [[Int32]]? = nil,
        outputURL: URL,
        temperature: Float? = nil,
        onProgress: ((Float) -> Void)? = nil
    ) async throws -> Int {
        let temp = temperature ?? pipelineConfig.defaultTemperature
        let numCodeGroups = config.code_predictor_config.num_code_groups

        let textChunks = TextChunker.chunk(text, maxWords: TextChunker.defaultMaxWords)
        guard !textChunks.isEmpty else { return 0 }

        let embeddingData = speakerEmbedding

        return try await Task.detached(priority: .userInitiated) { [self] in
            try Device.withDefaultDevice(self.device) {
                let writer = try StreamingWAVWriter(to: outputURL)
                let speakerEmbed: MLXArray? = embeddingData.map { MLXArray($0) }

                for (chunkIndex, textChunk) in textChunks.enumerated() {
                    if Task.isCancelled {
                        _ = writer.finalize()
                        return writer.sampleCount
                    }

                    let progress = Float(chunkIndex) / Float(textChunks.count)
                    onProgress?(progress)

                    self.model.clearGenerationCache()

                    var codes: [[Int32]] = []
                    autoreleasepool {
                        codes = self.model.generateCodes(
                            prompt: speaker,
                            text: textChunk,
                            speakerEmbedding: speakerEmbed,
                            referenceTranscript: referenceTranscript,
                            referenceAudioCodes: referenceAudioCodes,
                            tokenizer: self.tokenizer,
                            temperature: temp,
                            maxTokens: 600
                        )
                    }

                    Stream.defaultStream(.gpu).synchronize()
                    Memory.clearCache()

                    guard !codes.isEmpty else { continue }

                    // Decode in small batches
                    let samplesPerFrame = 1920
                    let decodeChunkSize = 16
                    let leftContextSize = 8

                    var chunkSamples: [Float] = []
                    chunkSamples.reserveCapacity(codes.count * samplesPerFrame)
                    var decodeLeftContext: [[Int32]] = []
                    var pos = 0

                    while pos < codes.count {
                        autoreleasepool {
                            let endPos = min(pos + decodeChunkSize, codes.count)
                            let batchCodes = decodeLeftContext + Array(codes[pos..<endPos])

                            let flatCodes: [Int32] = batchCodes.flatMap { $0 }
                            let codesArray = MLXArray(flatCodes).reshaped([1, batchCodes.count, numCodeGroups])
                            let audio = self.decoder.mlxDecode(codes: codesArray)
                            let flatAudio = audio.reshaped([-1])
                            eval(flatAudio)
                            var batchSamples = flatAudio.asArray(Float.self)

                            let contextSamples = decodeLeftContext.count * samplesPerFrame
                            if contextSamples > 0 && batchSamples.count > contextSamples {
                                batchSamples = Array(batchSamples.dropFirst(contextSamples))
                            }

                            for sample in batchSamples {
                                if sample.isNaN || sample.isInfinite {
                                    chunkSamples.append(0.0)
                                } else {
                                    chunkSamples.append(max(-1.0, min(1.0, sample)))
                                }
                            }

                            decodeLeftContext = Array(codes[max(0, endPos - leftContextSize)..<endPos])
                            pos = endPos
                        }

                        Stream.defaultStream(.gpu).synchronize()
                        Memory.clearCache()
                    }

                    guard !chunkSamples.isEmpty else { continue }
                    try writer.write(samples: chunkSamples)
                    chunkSamples = []

                    self.model.clearGenerationCache()
                    self.decoder.clearCompiledCache()
                    Stream.defaultStream(.gpu).synchronize()
                    Memory.clearCache()
                }

                onProgress?(1.0)
                let result = writer.finalize()
                return result.sampleCount
            }
        }.value
    }

    // MARK: - Batch Generation (Silent)

    /// Generate speech for text of any length, returning all samples at once.
    ///
    /// Uses text chunking and crossfading for seamless output.
    /// For very long text, prefer `generateToFile` for lower memory usage.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize (any length)
    ///   - speaker: Speaker name
    ///   - speakerEmbedding: Optional speaker embedding for voice cloning
    ///   - referenceTranscript: Optional transcript for ICL
    ///   - temperature: Generation temperature
    ///   - onProgress: Optional progress callback (0.0 to 1.0)
    /// - Returns: Audio samples at 24kHz
    public func generateBatch(
        text: String,
        speaker: String = "",
        speakerEmbedding: [Float]? = nil,
        referenceTranscript: String? = nil,
        temperature: Float? = nil,
        onProgress: ((Float) -> Void)? = nil
    ) async -> [Float] {
        let temp = temperature ?? pipelineConfig.defaultTemperature
        let crossfade = pipelineConfig.crossfadeSamples
        let numCodeGroups = config.code_predictor_config.num_code_groups

        let textChunks = TextChunker.chunk(text, maxWords: TextChunker.defaultMaxWords)
        guard !textChunks.isEmpty else { return [] }

        // Short text: single generation
        if textChunks.count == 1 {
            onProgress?(0.0)
            let samples = generate(text: textChunks[0], speaker: speaker, temperature: temp)
            onProgress?(1.0)
            return samples
        }

        let embeddingData = speakerEmbedding

        return await Task.detached(priority: .userInitiated) { [self] () -> [Float] in
            Device.withDefaultDevice(self.device) {
                var allSamples: [Float] = []
                var previousTail: [Float] = []
                let speakerEmbed: MLXArray? = embeddingData.map { MLXArray($0) }

                for (chunkIndex, textChunk) in textChunks.enumerated() {
                    if Task.isCancelled { return allSamples }

                    let isLastChunk = chunkIndex == textChunks.count - 1
                    let progress = Float(chunkIndex) / Float(textChunks.count)
                    onProgress?(progress)

                    let codes = self.model.generateCodes(
                        prompt: speaker,
                        text: textChunk,
                        speakerEmbedding: speakerEmbed,
                        referenceTranscript: referenceTranscript,
                        tokenizer: self.tokenizer,
                        temperature: temp,
                        maxTokens: 600
                    )

                    guard !codes.isEmpty else { continue }

                    // Decode in batches
                    let samplesPerFrame = 1920
                    let decodeChunkSize = 24
                    let leftContextSize = 8

                    var chunkSamples: [Float] = []
                    var decodeLeftContext: [[Int32]] = []
                    var pos = 0

                    while pos < codes.count {
                        let endPos = min(pos + decodeChunkSize, codes.count)
                        let batchCodes = decodeLeftContext + Array(codes[pos..<endPos])

                        let flatCodes: [Int32] = batchCodes.flatMap { $0 }
                        let codesArray = MLXArray(flatCodes).reshaped([1, batchCodes.count, numCodeGroups])
                        let audio = self.decoder.mlxDecode(codes: codesArray)
                        let flatAudio = audio.reshaped([-1])
                        eval(flatAudio)
                        var batchSamples = flatAudio.asArray(Float.self)

                        let contextSamples = decodeLeftContext.count * samplesPerFrame
                        if contextSamples > 0 && batchSamples.count > contextSamples {
                            batchSamples = Array(batchSamples.dropFirst(contextSamples))
                        }

                        for sample in batchSamples {
                            if sample.isNaN || sample.isInfinite {
                                chunkSamples.append(0.0)
                            } else {
                                chunkSamples.append(max(-1.0, min(1.0, sample)))
                            }
                        }

                        decodeLeftContext = Array(codes[max(0, endPos - leftContextSize)..<endPos])
                        pos = endPos

                        DeviceSelector.synchronizeIfNeeded(device: self.device)
                        Memory.clearCache()
                    }

                    guard !chunkSamples.isEmpty else { continue }

                    // Crossfade with previous chunk
                    if !previousTail.isEmpty && crossfade > 0 {
                        let fadeLength = min(crossfade, previousTail.count, chunkSamples.count)
                        var crossfaded: [Float] = []
                        for i in 0..<fadeLength {
                            let fadeOut = Float(fadeLength - i) / Float(fadeLength)
                            let fadeIn = Float(i) / Float(fadeLength)
                            crossfaded.append(previousTail[i] * fadeOut + chunkSamples[i] * fadeIn)
                        }
                        allSamples.append(contentsOf: crossfaded)
                        chunkSamples = Array(chunkSamples.dropFirst(fadeLength))
                    }

                    if isLastChunk {
                        allSamples.append(contentsOf: chunkSamples)
                    } else if chunkSamples.count > crossfade {
                        allSamples.append(contentsOf: chunkSamples.dropLast(crossfade))
                        previousTail = Array(chunkSamples.suffix(crossfade))
                    } else {
                        previousTail = chunkSamples
                    }

                    DeviceSelector.synchronizeIfNeeded(device: self.device)
                    Memory.clearCache()
                }

                onProgress?(1.0)
                return allSamples
            }
        }.value
    }

    // MARK: - Voice Cloning

    /// Extract a speaker embedding from audio samples.
    ///
    /// - Parameter audioSamples: Raw audio samples (any sample rate, but 16kHz preferred)
    /// - Returns: 1024-dimensional speaker embedding, or nil if speaker encoder is not loaded
    public func extractSpeakerEmbedding(audioSamples: [Float]) -> [Float]? {
        guard let spkEncoder = speakerEncoder, spkEncoder.isWeightsLoaded else {
            return nil
        }

        return Device.withDefaultDevice(device) {
            defer { Memory.clearCache() }
            let audioArray = MLXArray(audioSamples)
            let embedding = spkEncoder.extractEmbedding(audio: audioArray)
            eval(embedding)
            return embedding.asArray(Float.self)
        }
    }

    /// Encode reference audio for ICL (in-context learning) voice cloning.
    ///
    /// - Parameter audioSamples: Audio samples at 24kHz
    /// - Returns: Audio codes as [[Int32]] with shape [num_quantizers, time], or nil
    public func encodeReferenceAudio(audioSamples: [Float]) -> [[Int32]]? {
        guard let encoder = audioEncoder else { return nil }

        return Device.withDefaultDevice(device) {
            defer { Memory.clearCache() }

            let audioArray = MLXArray(audioSamples).expandedDimensions(axis: 0)
            let codes = encoder.encode(audioArray)
            eval(codes)

            let numQuantizers = codes.shape[1]
            let timeFrames = codes.shape[2]

            var result: [[Int32]] = []
            for q in 0..<numQuantizers {
                let quantizerCodes = codes[0, q, 0..<timeFrames]
                eval(quantizerCodes)
                result.append(quantizerCodes.asArray(Int32.self))
            }
            return result
        }
    }

    // MARK: - Memory Management

    /// Clear cached state from the model and decoder.
    /// Call this between generations if memory accumulates.
    public func clearCache() {
        model.clearGenerationCache()
        decoder.clearCompiledCache()
        DeviceSelector.synchronizeIfNeeded(device: device)
        Memory.clearCache()
    }

    // MARK: - Private

    /// Apply mixed 4-6 bit quantization for non-pre-quantized models.
    private static func applyMixedQuantization(to model: Module) {
        quantize(model: model) { path, module in
            guard module is Quantizable else { return nil }

            let pathLower = path.lowercased()
            let use6Bit = pathLower.contains("embed") ||
                          pathLower.contains("qproj") ||
                          pathLower.contains("kproj") ||
                          pathLower.contains("vproj") ||
                          pathLower.contains("q_proj") ||
                          pathLower.contains("k_proj") ||
                          pathLower.contains("v_proj") ||
                          pathLower.contains("lm_head") ||
                          pathLower.contains("codec_head")

            return use6Bit
                ? (groupSize: 64, bits: 6, mode: .affine)
                : (groupSize: 64, bits: 4, mode: .affine)
        }
    }
}

// MARK: - Errors

public enum Qwen3TTSError: LocalizedError {
    case fileNotFound(String)
    case decoderLoadFailed
    case modelNotLoaded

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let file):
            return "Required file not found: \(file)"
        case .decoderLoadFailed:
            return "Failed to load MLX audio decoder"
        case .modelNotLoaded:
            return "Model is not loaded"
        }
    }
}

// MARK: - WAV Utility

extension Qwen3TTSPipeline {
    /// Convert raw WAV data to float samples (assumes 16-bit PCM).
    public static func wavToFloatSamples(data: Data) -> [Float] {
        guard data.count > 44 else { return [] }
        let sampleData = data.dropFirst(44)
        var samples: [Float] = []
        samples.reserveCapacity(sampleData.count / 2)

        for i in stride(from: 0, to: sampleData.count - 1, by: 2) {
            let low = UInt16(sampleData[sampleData.startIndex + i])
            let high = UInt16(sampleData[sampleData.startIndex + i + 1])
            let int16 = Int16(bitPattern: low | (high << 8))
            samples.append(Float(int16) / 32767.0)
        }

        return samples
    }
}
