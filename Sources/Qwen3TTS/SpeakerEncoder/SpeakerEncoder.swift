import Foundation
import MLX
import MLXNN
import Accelerate

// MARK: - Mel Filterbank Cache

private final class MelFilterbankCache {
    static let shared = MelFilterbankCache()

    private var cache: [String: MLXArray] = [:]
    private let lock = NSLock()

    func get(sampleRate: Int, nFFT: Int, numMels: Int, fmin: Float, fmax: Float) -> MLXArray {
        let key = "\(sampleRate)_\(nFFT)_\(numMels)_\(fmin)_\(fmax)"
        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[key] {
            return cached
        }

        let filterbank = createMelFilterbankImpl(
            sampleRate: sampleRate,
            nFFT: nFFT,
            numMels: numMels,
            fmin: fmin,
            fmax: fmax
        )
        cache[key] = filterbank
        return filterbank
    }
}

// MARK: - Mel Spectrogram

public func melSpectrogram(
    audio: MLXArray,
    nFFT: Int = 1024,
    numMels: Int = 128,
    sampleRate: Int = 24000,
    hopSize: Int = 256,
    winSize: Int = 1024,
    fmin: Float = 0.0,
    fmax: Float = 12000.0
) -> MLXArray {
    var x = audio
    if x.ndim == 1 {
        x = x.expandedDimensions(axis: 0)
    }

    let batchSize = x.shape[0]

    let melFilters = MelFilterbankCache.shared.get(
        sampleRate: sampleRate,
        nFFT: nFFT,
        numMels: numMels,
        fmin: fmin,
        fmax: fmax
    )

    var mels: [MLXArray] = []
    for i in 0..<batchSize {
        let sample = x[i]
        let spec = speakerEncoderSTFT(sample, nFFT: nFFT, hopLength: hopSize, winLength: winSize)
        let specMag = abs(spec)
        let mel = matmul(specMag, melFilters)
        let logMel = log(clip(mel, min: Float(1e-5), max: Float.greatestFiniteMagnitude))
        mels.append(logMel)
    }

    return MLX.stacked(mels, axis: 0)
}

private func createMelFilterbankImpl(
    sampleRate: Int,
    nFFT: Int,
    numMels: Int,
    fmin: Float,
    fmax: Float
) -> MLXArray {
    let numFreqs = nFFT / 2 + 1

    let fMin: Float = 0.0
    let fSp: Float = 200.0 / 3.0
    let minLogHz: Float = 1000.0
    let minLogMel = (minLogHz - fMin) / fSp
    let logStep = Float(log(6.4) / 27.0)

    func hzToMel(_ hz: Float) -> Float {
        if hz >= minLogHz {
            return minLogMel + Float(log(Double(hz / minLogHz))) / logStep
        }
        return (hz - fMin) / fSp
    }

    func melToHz(_ mel: Float) -> Float {
        if mel >= minLogMel {
            return minLogHz * Float(exp(Double(logStep * (mel - minLogMel))))
        }
        return fMin + fSp * mel
    }

    var allFreqs: [Float] = []
    for i in 0..<numFreqs {
        allFreqs.append(Float(i) * Float(sampleRate / 2) / Float(numFreqs - 1))
    }

    let mMin = hzToMel(fmin)
    let mMax = hzToMel(fmax)

    var mPts: [Float] = []
    for i in 0..<(numMels + 2) {
        mPts.append(mMin + Float(i) * (mMax - mMin) / Float(numMels + 1))
    }

    let fPts = mPts.map { melToHz($0) }

    var fDiff: [Float] = []
    for i in 0..<(fPts.count - 1) {
        fDiff.append(fPts[i + 1] - fPts[i])
    }

    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: numMels), count: numFreqs)

    for k in 0..<numFreqs {
        let freq = allFreqs[k]
        for m in 0..<numMels {
            let fLeft = fPts[m]
            let fRight = fPts[m + 2]
            let downSlope = (freq - fLeft) / fDiff[m]
            let upSlope = (fRight - freq) / fDiff[m + 1]
            filterbank[k][m] = max(0.0, min(downSlope, upSlope))
        }
    }

    for m in 0..<numMels {
        let enorm = 2.0 / (fPts[m + 2] - fPts[m])
        for k in 0..<numFreqs {
            filterbank[k][m] *= enorm
        }
    }

    let flat = filterbank.flatMap { $0 }
    return MLXArray(flat).reshaped([numFreqs, numMels])
}

private func reflectPadSignal(_ signal: MLXArray, pad: Int) -> MLXArray {
    if pad <= 0 { return signal }
    let n = signal.shape[0]

    var indices: [Int32] = []

    for i in stride(from: pad, through: 1, by: -1) {
        indices.append(Int32(i))
    }

    for i in 0..<n {
        indices.append(Int32(i))
    }

    for i in stride(from: n - 2, through: max(n - pad - 1, 0), by: -1) {
        indices.append(Int32(i))
    }

    return signal[MLXArray(indices)]
}

private func speakerEncoderSTFT(
    _ signal: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int
) -> MLXArray {
    let padLength = nFFT / 2
    let padded = reflectPadSignal(signal, pad: padLength)

    let numSamples = padded.shape[0]
    let numFrames = (numSamples - nFFT) / hopLength + 1

    var window = [Float](repeating: 0, count: winLength)
    for i in 0..<winLength {
        window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(winLength - 1)))
    }

    let paddedArray = padded.asType(DType.float32)
    eval(paddedArray)

    let paddedFlat = paddedArray.asArray(Float.self)
    let maxAccess = numFrames > 0 ? (numFrames - 1) * hopLength + nFFT - 1 : -1
    print("DEBUG [stft]: paddedFlat.count=\(paddedFlat.count) numFrames=\(numFrames) nFFT=\(nFFT) hop=\(hopLength) maxAccess=\(maxAccess)")
    precondition(numFrames == 0 || maxAccess < paddedFlat.count, "stft: would access index \(maxAccess) in array of size \(paddedFlat.count)")

    var framesFlat: [Float] = []
    framesFlat.reserveCapacity(numFrames * nFFT)

    for i in 0..<numFrames {
        let start = i * hopLength
        for j in 0..<nFFT {
            let sample = paddedFlat[start + j]
            let windowVal = window[j]
            framesFlat.append(sample * windowVal)
        }
    }

    let framesStacked = MLXArray(framesFlat).reshaped([numFrames, nFFT])
    eval(framesStacked)

    let fftResult = rfft(framesStacked, axis: 1)

    return fftResult
}

// MARK: - Speaker Encoder Components

private func reflectPad1d(_ x: MLXArray, pad: Int) -> MLXArray {
    if pad <= 0 { return x }
    let (_, time, _) = (x.shape[0], x.shape[1], x.shape[2])

    var indices: [Int32] = []

    for i in stride(from: pad, through: 1, by: -1) {
        indices.append(Int32(i))
    }

    for i in 0..<time {
        indices.append(Int32(i))
    }

    for i in stride(from: time - 2, through: max(time - pad - 1, 0), by: -1) {
        indices.append(Int32(i))
    }

    return x[0..., MLXArray(indices), 0...]
}

public class TimeDelayNetBlock: Module {
    public let conv: Conv1d
    private let padAmount: Int

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1) {
        self.padAmount = (kernelSize - 1) * dilation / 2
        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = reflectPad1d(h, pad: padAmount)
        h = conv(h)
        h = h.transposed(0, 2, 1)
        return relu(h)
    }
}

public class Res2NetBlock: Module {
    public let blocks: [TimeDelayNetBlock]
    private let scale: Int

    public init(inChannels: Int, outChannels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        self.scale = scale
        let inChannel = inChannels / scale
        let hiddenChannel = outChannels / scale

        var blocks: [TimeDelayNetBlock] = []
        for _ in 0..<(scale - 1) {
            blocks.append(TimeDelayNetBlock(
                inChannels: inChannel,
                outChannels: hiddenChannel,
                kernelSize: kernelSize,
                dilation: dilation
            ))
        }
        self.blocks = blocks
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunkSize = x.shape[1] / scale
        var outputs: [MLXArray] = []
        var outputPart: MLXArray? = nil

        for i in 0..<scale {
            let chunk = x[0..., (i * chunkSize)..<((i + 1) * chunkSize), 0...]

            if i == 0 {
                outputPart = chunk
            } else if i == 1 {
                outputPart = blocks[i - 1](chunk)
            } else {
                outputPart = blocks[i - 1](chunk + outputPart!)
            }
            outputs.append(outputPart!)
        }

        return concatenated(outputs, axis: 1)
    }
}

public class SqueezeExcitationBlock: Module {
    public let conv1: Conv1d
    public let conv2: Conv1d

    public init(inChannels: Int, seChannels: Int, outChannels: Int) {
        self.conv1 = Conv1d(inputChannels: inChannels, outputChannels: seChannels, kernelSize: 1)
        self.conv2 = Conv1d(inputChannels: seChannels, outputChannels: outChannels, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xMean = mean(x, axis: 2, keepDims: true)
        var se = xMean.transposed(0, 2, 1)
        se = relu(conv1(se))
        se = sigmoid(conv2(se))
        se = se.transposed(0, 2, 1)
        return x * se
    }
}

public class SqueezeExcitationRes2NetBlock: Module {
    public let tdnn1: TimeDelayNetBlock
    public let res2net_block: Res2NetBlock
    public let tdnn2: TimeDelayNetBlock
    public let se_block: SqueezeExcitationBlock

    public init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        self.tdnn1 = TimeDelayNetBlock(inChannels: inChannels, outChannels: outChannels, kernelSize: 1, dilation: 1)
        self.res2net_block = Res2NetBlock(inChannels: outChannels, outChannels: outChannels, scale: res2netScale, kernelSize: kernelSize, dilation: dilation)
        self.tdnn2 = TimeDelayNetBlock(inChannels: outChannels, outChannels: outChannels, kernelSize: 1, dilation: 1)
        self.se_block = SqueezeExcitationBlock(inChannels: outChannels, seChannels: seChannels, outChannels: outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = tdnn1(x)
        h = res2net_block(h)
        h = tdnn2(h)
        h = se_block(h)
        return h + residual
    }
}

public class AttentiveStatisticsPooling: Module {
    public let tdnn: TimeDelayNetBlock
    public let conv: Conv1d
    private let eps: Float = 1e-12

    public init(channels: Int, attentionChannels: Int = 128) {
        self.tdnn = TimeDelayNetBlock(inChannels: channels * 3, outChannels: attentionChannels, kernelSize: 1, dilation: 1)
        self.conv = Conv1d(inputChannels: attentionChannels, outputChannels: channels, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, channels, seqLength) = (x.shape[0], x.shape[1], x.shape[2])

        let meanVal = mean(x, axis: 2, keepDims: true)
        let varVal = variance(x, axis: 2, keepDims: true)
        let stdVal = sqrt(varVal + eps)

        let meanExpanded = broadcast(meanVal, to: [batch, channels, seqLength])
        let stdExpanded = broadcast(stdVal, to: [batch, channels, seqLength])

        var attention = concatenated([x, meanExpanded, stdExpanded], axis: 1)

        attention = tdnn(attention)
        attention = tanh(attention)

        attention = attention.transposed(0, 2, 1)
        attention = conv(attention)
        attention = attention.transposed(0, 2, 1)

        attention = softmax(attention, axis: 2)

        let weightedMean = sum(attention * x, axis: 2, keepDims: true)
        let diff = x - weightedMean
        let weightedVar = sum(attention * (diff * diff), axis: 2, keepDims: true)
        let weightedStd = sqrt(clip(weightedVar, min: eps, max: Float.greatestFiniteMagnitude))

        let pooled = concatenated([weightedMean, weightedStd], axis: 1)
        return pooled
    }
}

// MARK: - Speaker Encoder Configuration

public struct SpeakerEncoderConfig: Sendable {
    public var encDim: Int = 1024
    public var melDim: Int = 128
    public var encChannels: [Int] = [512, 512, 512, 512, 1536]
    public var encKernelSizes: [Int] = [5, 3, 3, 3, 1]
    public var encDilations: [Int] = [1, 2, 3, 4, 1]
    public var encRes2netScale: Int = 8
    public var encSeChannels: Int = 128
    public var encAttentionChannels: Int = 128
    public var sampleRate: Int = 24000

    public init() {}

    public init(from dict: [String: Any]) {
        if let v = dict["enc_dim"] as? Int { encDim = v }
        if let v = dict["sample_rate"] as? Int { sampleRate = v }
    }
}

// MARK: - Speaker Encoder

public class SpeakerEncoder: Module {
    public let config: SpeakerEncoderConfig

    public let blocks: [Module]

    private let block0: TimeDelayNetBlock
    private let block1: SqueezeExcitationRes2NetBlock
    private let block2: SqueezeExcitationRes2NetBlock
    private let block3: SqueezeExcitationRes2NetBlock

    public let mfa: TimeDelayNetBlock
    public let asp: AttentiveStatisticsPooling
    public let fc: Conv1d

    public private(set) var isWeightsLoaded: Bool = false

    public init(config: SpeakerEncoderConfig = SpeakerEncoderConfig()) {
        self.config = config

        self.block0 = TimeDelayNetBlock(
            inChannels: config.melDim,
            outChannels: config.encChannels[0],
            kernelSize: config.encKernelSizes[0],
            dilation: config.encDilations[0]
        )

        self.block1 = SqueezeExcitationRes2NetBlock(
            inChannels: config.encChannels[0],
            outChannels: config.encChannels[1],
            res2netScale: config.encRes2netScale,
            seChannels: config.encSeChannels,
            kernelSize: config.encKernelSizes[1],
            dilation: config.encDilations[1]
        )

        self.block2 = SqueezeExcitationRes2NetBlock(
            inChannels: config.encChannels[1],
            outChannels: config.encChannels[2],
            res2netScale: config.encRes2netScale,
            seChannels: config.encSeChannels,
            kernelSize: config.encKernelSizes[2],
            dilation: config.encDilations[2]
        )

        self.block3 = SqueezeExcitationRes2NetBlock(
            inChannels: config.encChannels[2],
            outChannels: config.encChannels[3],
            res2netScale: config.encRes2netScale,
            seChannels: config.encSeChannels,
            kernelSize: config.encKernelSizes[3],
            dilation: config.encDilations[3]
        )

        self.blocks = [block0, block1, block2, block3]

        self.mfa = TimeDelayNetBlock(
            inChannels: config.encChannels[4],
            outChannels: config.encChannels[4],
            kernelSize: config.encKernelSizes[4],
            dilation: config.encDilations[4]
        )

        self.asp = AttentiveStatisticsPooling(
            channels: config.encChannels[4],
            attentionChannels: config.encAttentionChannels
        )

        self.fc = Conv1d(
            inputChannels: config.encChannels[4] * 2,
            outputChannels: config.encDim,
            kernelSize: 1
        )

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        var hiddenStatesList: [MLXArray] = []

        h = block0(h)
        hiddenStatesList.append(h)

        h = block1(h)
        hiddenStatesList.append(h)

        h = block2(h)
        hiddenStatesList.append(h)

        h = block3(h)
        hiddenStatesList.append(h)

        h = concatenated(Array(hiddenStatesList[1...]), axis: 1)
        h = mfa(h)

        h = asp(h)

        h = h.transposed(0, 2, 1)
        h = fc(h)
        h = h.transposed(0, 2, 1)

        h = h.squeezed(axis: 2)

        return h
    }

    public func extractEmbedding(audio: MLXArray, sampleRate: Int = 24000) -> MLXArray {
        let mels = melSpectrogram(
            audio: audio,
            nFFT: 1024,
            numMels: 128,
            sampleRate: sampleRate,
            hopSize: 256,
            winSize: 1024,
            fmin: 0,
            fmax: 12000
        )

        let embedding = self(mels)
        eval(embedding)

        return embedding
    }

    private func transposeConv(_ weight: MLXArray) -> MLXArray {
        let t = weight.transposed(0, 2, 1)
        eval(t)
        return t
    }

    public func load(weights: [String: MLXArray]) {
        var w: [String: MLXArray] = [:]
        for (key, val) in weights where key.hasPrefix("speaker_encoder.") {
            let newKey = String(key.dropFirst("speaker_encoder.".count))
            w[newKey] = val
        }

        guard !w.isEmpty else {
            return
        }

        func loadConv(_ conv: Conv1d, weightKey: String, biasKey: String) {
            if let wt = w[weightKey], let b = w[biasKey] {
                let params = ModuleParameters.unflattened([
                    "weight": transposeConv(wt),
                    "bias": b
                ])
                _ = try? conv.update(parameters: params, verify: .none)
            }
        }

        func loadTDNN(_ tdnn: TimeDelayNetBlock, prefix: String) {
            loadConv(tdnn.conv, weightKey: "\(prefix).conv.weight", biasKey: "\(prefix).conv.bias")
        }

        loadTDNN(block0, prefix: "blocks.0")

        let seBlocks = [block1, block2, block3]
        for (i, block) in seBlocks.enumerated() {
            let prefix = "blocks.\(i + 1)"

            loadTDNN(block.tdnn1, prefix: "\(prefix).tdnn1")
            loadTDNN(block.tdnn2, prefix: "\(prefix).tdnn2")

            loadConv(block.se_block.conv1, weightKey: "\(prefix).se_block.conv1.weight", biasKey: "\(prefix).se_block.conv1.bias")
            loadConv(block.se_block.conv2, weightKey: "\(prefix).se_block.conv2.weight", biasKey: "\(prefix).se_block.conv2.bias")

            for j in 0..<7 {
                loadTDNN(block.res2net_block.blocks[j], prefix: "\(prefix).res2net_block.blocks.\(j)")
            }
        }

        loadTDNN(mfa, prefix: "mfa")

        loadTDNN(asp.tdnn, prefix: "asp.tdnn")
        loadConv(asp.conv, weightKey: "asp.conv.weight", biasKey: "asp.conv.bias")

        loadConv(fc, weightKey: "fc.weight", biasKey: "fc.bias")

        isWeightsLoaded = true

        let testWeight = block1.se_block.conv1.weight
        eval(testWeight)
    }
}
