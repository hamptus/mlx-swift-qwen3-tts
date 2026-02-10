import Foundation

public enum AudioSampleWriter {
    public static func wavData(samples: [Float], sampleRate: Double = 24000) -> Data {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count * 2)
        let fileSize = 36 + dataSize

        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
        }

        return data
    }

    public static func write(samples: [Float], to url: URL, sampleRate: Double = 24000) throws {
        let data = wavData(samples: samples, sampleRate: sampleRate)
        try data.write(to: url, options: .atomic)
    }
}

/// Streaming WAV writer that writes samples incrementally to disk.
/// Writes a placeholder header first, then appends sample data, and updates the header on finalize.
public final class StreamingWAVWriter {
    public struct Result {
        public let sampleCount: Int
    }

    private let url: URL
    private let sampleRate: Double
    private let fileHandle: FileHandle
    public private(set) var sampleCount: Int = 0

    public init(to url: URL, sampleRate: Double = 24000) throws {
        self.url = url
        self.sampleRate = sampleRate

        // Write placeholder 44-byte WAV header
        let header = Data(count: 44)
        try header.write(to: url)

        self.fileHandle = try FileHandle(forWritingTo: url)
        fileHandle.seekToEndOfFile()
    }

    public func write(samples: [Float]) throws {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            withUnsafeBytes(of: int16.littleEndian) { data.append(contentsOf: $0) }
        }
        fileHandle.write(data)
        sampleCount += samples.count
    }

    public func finalize() -> Result {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(sampleCount * 2)
        let fileSize = 36 + dataSize

        var header = Data()
        header.append(contentsOf: "RIFF".utf8)
        withUnsafeBytes(of: fileSize.littleEndian) { header.append(contentsOf: $0) }
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        withUnsafeBytes(of: UInt32(16).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: numChannels.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: blockAlign.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample.littleEndian) { header.append(contentsOf: $0) }
        header.append(contentsOf: "data".utf8)
        withUnsafeBytes(of: dataSize.littleEndian) { header.append(contentsOf: $0) }

        fileHandle.seek(toFileOffset: 0)
        fileHandle.write(header)
        fileHandle.closeFile()

        return Result(sampleCount: sampleCount)
    }
}
