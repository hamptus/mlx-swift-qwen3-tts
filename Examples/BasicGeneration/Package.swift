// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BasicGeneration",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "BasicGeneration",
            dependencies: [
                .product(name: "Qwen3TTS", package: "mlx-swift-qwen3-tts"),
            ],
            path: "Sources"
        ),
    ]
)
