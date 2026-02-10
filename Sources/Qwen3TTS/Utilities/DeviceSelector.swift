import Foundation
import MLX

public enum DeviceSelector {
    // Cache the resolved device to avoid repeated checks
    private static var cachedDevice: Device?
    private static let cacheLock = NSLock()

    public static func resolveDevice() -> Device {
        // Return cached result if available
        cacheLock.lock()
        if let cached = cachedDevice {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let env = ProcessInfo.processInfo.environment["MLX_DEVICE"]?.lowercased()
        if env == "gpu" || env == "metal" {
            cacheDevice(.gpu)
            return .gpu
        }
        if env == "cpu" {
            cacheDevice(.cpu)
            return .cpu
        }

        // Default to GPU — MLX will fall back to CPU if Metal is unavailable
        let result: Device = .gpu
        cacheDevice(result)
        return result
    }

    private static func cacheDevice(_ device: Device) {
        cacheLock.lock()
        cachedDevice = device
        cacheLock.unlock()
    }

    public static func stream(for device: Device) -> StreamOrDevice {
        if device.deviceType == .gpu {
            return .gpu
        }
        return .cpu
    }

    public static func synchronizeIfNeeded(device: Device) {
        guard device.deviceType == .gpu else { return }
        Stream.defaultStream(device).synchronize()
    }
}
