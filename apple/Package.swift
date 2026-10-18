// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "EdgeLens", platforms: [.macOS(.v14)], products: [.executable(name: "EdgeLens", targets: ["EdgeLens"])], targets: [.executableTarget(name: "EdgeLens", resources: [.copy("Resources")])])
