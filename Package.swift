// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "mlx-swift-qwen3-tts",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "Qwen3TTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/Qwen3TTS"
        ),
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: ["Qwen3TTS"],
            path: "Tests/Qwen3TTSTests"
        ),
    ]
)
