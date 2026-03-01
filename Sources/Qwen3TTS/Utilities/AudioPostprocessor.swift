import Foundation

/// Audio postprocessing utilities for Qwen3-TTS output.
///
/// Quantized TTS models (especially 4-bit and 8-bit variants) produce systematic
/// artifacts in their output: airy static between speech segments and inconsistent
/// output levels. These utilities address both issues.
///
/// Typical usage after file-based generation:
/// ```swift
/// try await pipeline.generateToFile(text: text, outputURL: outputURL, ...)
/// AudioPostprocessor.postprocessWAVFileInPlace(outputURL)
/// ```
public enum AudioPostprocessor {

    /// Applies noise gate and peak normalization to a 24kHz 16-bit PCM WAV file in place.
    ///
    /// The noise gate suppresses the low-level airy static that quantized models produce
    /// between speech segments. Peak normalization brings quiet output up to -1 dBFS
    /// without reducing louder output.
    ///
    /// - Parameter url: URL of a WAV file with a standard 44-byte header.
    public static func postprocessWAVFileInPlace(_ url: URL) {
        guard let data = try? Data(contentsOf: url), data.count > 44 else { return }

        let header = data.prefix(44)
        let sampleBytes = data.dropFirst(44)
        let sampleCount = sampleBytes.count / 2
        guard sampleCount > 0 else { return }

        // Decode Int16 LE PCM → Float
        var samples = [Float](repeating: 0, count: sampleCount)
        sampleBytes.withUnsafeBytes { rawPtr in
            guard let baseAddr = rawPtr.baseAddress else { return }
            let int16Ptr = baseAddr.assumingMemoryBound(to: Int16.self)
            for i in 0..<sampleCount {
                samples[i] = Float(Int16(littleEndian: int16Ptr[i])) / 32767.0
            }
        }

        applyNoiseGate(&samples)
        peakNormalize(&samples)

        // Encode Float → Int16 LE PCM
        var output = Data(header)
        output.reserveCapacity(header.count + sampleCount * 2)
        for f in samples {
            let i16 = Int16(max(-32767, min(32767, f * 32767.0))).littleEndian
            withUnsafeBytes(of: i16) { output.append(contentsOf: $0) }
        }

        try? output.write(to: url, options: .atomic)
    }

    /// Noise gate with hold — suppresses airy static between speech segments.
    ///
    /// Uses 20ms windowed RMS with 120ms hold and linear crossfade at window boundaries
    /// to avoid clicks at gate open/close transitions.
    ///
    /// - Parameter samples: Float PCM samples at 24kHz, modified in place.
    public static func applyNoiseGate(_ samples: inout [Float]) {
        let windowSize = 480  // 20ms at 24kHz
        let threshold: Float = 0.008
        let holdWindows = 6   // Keep gate open ~120ms after signal drops below threshold
        let n = samples.count
        guard n > windowSize * 2 else { return }

        let numWindows = (n + windowSize - 1) / windowSize

        // Determine open/closed per window with hold
        var isOpen = [Bool](repeating: false, count: numWindows)
        var hold = 0
        for w in 0..<numWindows {
            let lo = w * windowSize
            let hi = min(lo + windowSize, n)
            var sq: Float = 0
            for i in lo..<hi { sq += samples[i] * samples[i] }
            let rms = sqrtf(sq / Float(hi - lo))
            if rms >= threshold {
                isOpen[w] = true
                hold = holdWindows
            } else if hold > 0 {
                isOpen[w] = true
                hold -= 1
            }
        }

        // Build per-sample gain: linearly interpolate between adjacent window midpoints
        let halfWin = windowSize / 2
        for i in 0..<n {
            let w = i / windowSize
            let pos = i % windowSize

            let g: Float
            if pos < halfWin && w > 0 {
                let t = Float(pos + halfWin) / Float(windowSize)
                g = (isOpen[w - 1] ? 1.0 : 0.0) * (1.0 - t) + (isOpen[w] ? 1.0 : 0.0) * t
            } else if pos >= halfWin && w + 1 < numWindows {
                let t = Float(pos - halfWin) / Float(windowSize)
                g = (isOpen[w] ? 1.0 : 0.0) * (1.0 - t) + (isOpen[w + 1] ? 1.0 : 0.0) * t
            } else {
                g = isOpen[min(w, numWindows - 1)] ? 1.0 : 0.0
            }
            samples[i] *= g
        }
    }

    /// Peak-normalizes samples to -1 dBFS. Only boosts — never reduces gain.
    ///
    /// Skips essentially-silent audio (peak < 0.01) to avoid amplifying pure noise.
    ///
    /// - Parameter samples: Float PCM samples, modified in place.
    public static func peakNormalize(_ samples: inout [Float]) {
        let peak = samples.reduce(0.0 as Float) { max($0, abs($1)) }
        guard peak > 0.01 else { return }      // Skip essentially-silent audio
        let targetPeak: Float = 0.891          // -1 dBFS
        guard peak < targetPeak else { return } // Skip if already at or above target
        let gain = targetPeak / peak
        for i in 0..<samples.count { samples[i] *= gain }
    }
}
