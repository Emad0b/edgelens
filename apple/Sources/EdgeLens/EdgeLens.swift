import SwiftUI
import CoreML
import Darwin

struct Sample: Codable { let input: [Double]; let label: Int }
struct Dataset: Codable { let samples: [Sample]; let dataset_sha256: String }
struct Measurement: Codable, Identifiable {
    var id: String { variant }
    let variant: String
    let accuracy: Double
    let p50Milliseconds: Double
    let p95Milliseconds: Double
    let loadMilliseconds: Double
    let sampledPeakResidentBytes: UInt64
    let baselineResidentBytes: UInt64
    let sampleCount: Int
}
struct RunReport: Codable {
    let operatingSystem: String
    let hardware: String
    let computePolicy: String
    let datasetSHA256: String
    let timestamp: Date
    let thermalStateAtStart: Int
    let thermalStateAtEnd: Int
    let measurements: [Measurement]
}
func residentBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return status == KERN_SUCCESS ? UInt64(info.resident_size) : 0
}
func hardwareName() -> String {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var bytes = [CChar](repeating: 0, count: size)
    sysctlbyname("hw.model", &bytes, &size, nil, 0)
    return String(cString: bytes)
}
func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)]
}
func evaluate(_ variant: String, _ samples: [Sample], _ cpuOnly: Bool) throws -> Measurement {
    guard let url = Bundle.module.url(forResource: variant, withExtension: "mlpackage", subdirectory: "Resources") else {
        throw NSError(domain: "EdgeLens", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing model. Run the Core ML export on macOS before building the app."])
    }
    // Compile separately; compilation is excluded from load and inference timings.
    let compiled = try MLModel.compileModel(at: url)
    defer { try? FileManager.default.removeItem(at: compiled) }
    let config = MLModelConfiguration()
    config.computeUnits = cpuOnly ? .cpuOnly : .all
    let baseline = residentBytes()
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let model = try MLModel(contentsOf: compiled, configuration: config)
    let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e6
    // Inputs are prepared outside measured inference. Same normalisation as training.
    let inputs = try samples.map { sample -> MLDictionaryFeatureProvider in
        let array = try MLMultiArray(shape: [1, 64], dataType: .float32)
        for (i, value) in sample.input.enumerated() { array[i] = NSNumber(value: value) }
        return try MLDictionaryFeatureProvider(dictionary: ["pixels": MLFeatureValue(multiArray: array)])
    }
    for i in 0..<50 { _ = try model.prediction(from: inputs[i % inputs.count]) }
    var timings: [Double] = []
    var peak = residentBytes()
    // Repeated timing passes; accuracy is measured separately once per held-out example.
    for i in 0..<500 {
        let start = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: inputs[i % inputs.count])
        timings.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        peak = max(peak, residentBytes())
    }
    var correct = 0
    for (i, input) in inputs.enumerated() {
        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "EdgeLens", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model output contract mismatch"])
        }
        let predicted = (0..<10).max { logits[$0].doubleValue < logits[$1].doubleValue }!
        if predicted == samples[i].label { correct += 1 }
    }
    return Measurement(variant: variant, accuracy: Double(correct)/Double(samples.count),
        p50Milliseconds: percentile(timings, 0.5), p95Milliseconds: percentile(timings, 0.95),
        loadMilliseconds: loadMs, sampledPeakResidentBytes: peak,
        baselineResidentBytes: baseline, sampleCount: samples.count)
}
@MainActor final class Benchmark: ObservableObject {
    @Published var running = false
    @Published var status = "Export the models, then run a device benchmark."
    @Published var results: [Measurement] = []
    @Published var cpuOnly = true
    @Published var reportURL: URL?
    func run() {
        running = true
        status = "Running on this Mac: 50 warmups + 500 predictions per model…"
        let cpu = cpuOnly
        Task {
            do {
                let report = try await Task.detached(priority: .userInitiated) {
                    guard let url = Bundle.module.url(forResource: "test_samples", withExtension: "json", subdirectory: "Resources") else {
                        throw NSError(domain: "EdgeLens", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing test samples. Run python -m edgelens.export_coreml first."])
                    }
                    let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: url))
                    guard !dataset.samples.isEmpty, dataset.samples.allSatisfy({$0.input.count == 64 && (0..<10).contains($0.label)}) else {
                        throw NSError(domain: "EdgeLens", code: 4, userInfo: [NSLocalizedDescriptionKey: "Invalid evaluation dataset"])
                    }
                    let thermalStart = ProcessInfo.processInfo.thermalState.rawValue
                    var measurements: [Measurement] = []
                    for name in ["FP32", "FP16", "INT8"] {
                        measurements.append(try autoreleasepool { try evaluate(name, dataset.samples, cpu) })
                    }
                    return RunReport(operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                        hardware: hardwareName(), computePolicy: cpu ? "cpuOnly" : "all (allowed, not proof of Neural Engine use)",
                        datasetSHA256: dataset.dataset_sha256, timestamp: Date(), thermalStateAtStart: thermalStart,
                        thermalStateAtEnd: ProcessInfo.processInfo.thermalState.rawValue, measurements: measurements)
                }.value
                let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("EdgeLens")
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let url = folder.appendingPathComponent("device-\(Int(Date().timeIntervalSince1970)).json")
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                try encoder.encode(report).write(to: url, options: .atomic)
                results = report.measurements
                reportURL = url
                status = "Complete. Exported measured results to Documents/EdgeLens."
            } catch { status = error.localizedDescription }
            running = false
        }
    }
}
@main struct EdgeLensApp: App {
    @StateObject private var benchmark = Benchmark()
    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 22) {
                Text("EdgeLens").font(.system(size: 42, weight: .bold, design: .rounded))
                Text("ON-DEVICE MODEL BENCHMARK").font(.caption).tracking(3).foregroundStyle(.secondary)
                Text("Compare FP32, FP16 and INT8 weight-compressed Core ML models on the same held-out digits.")
                Toggle("CPU only (disable to allow all compute units)", isOn: $benchmark.cpuOnly).disabled(benchmark.running)
                Button(benchmark.running ? "Benchmarking…" : "Run benchmark", action: benchmark.run).buttonStyle(.borderedProminent).disabled(benchmark.running)
                if benchmark.running { ProgressView() }
                Text(benchmark.status).foregroundStyle(.secondary)
                ForEach(benchmark.results) { item in
                    HStack {
                        Text(item.variant).bold().frame(width: 65, alignment: .leading)
                        Text(String(format: "Accuracy %.2f%% · p50 %.3f ms · p95 %.3f ms", item.accuracy*100, item.p50Milliseconds, item.p95Milliseconds))
                        Spacer()
                        Text(String(format: "RSS %.1f MB", Double(item.sampledPeakResidentBytes)/1e6))
                    }.padding().background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                }
                if let url = benchmark.reportURL { ShareLink("Export JSON report", item: url) }
                Text("Memory is sampled whole-process resident memory, not isolated model allocation or an exact peak. Inference timings exclude preprocessing and compilation. INT8 weights do not imply INT8 arithmetic or a speedup.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }.padding(36).frame(minWidth: 760, minHeight: 560)
        }
    }
}

