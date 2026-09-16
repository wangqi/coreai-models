// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import FoundationModels
import Tokenizers

/// Unified Core AI runner that creates FM API-compatible LanguageModel instances.
///
/// ## Usage
/// ```swift
/// let url = URL(fileURLWithPath: "/path/to/model")
/// let runner = try CoreAIRunner(contentsOf: url)
/// let engine = try await runner.makeInferenceEngine()
/// ```
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
public struct CoreAIRunner {
    // MARK: - Properties

    private let bundle: LanguageBundle
    private let engineVariant: String?
    private let kvCacheStrategy: KVCacheStrategy
    private let prefillChunkSizeOverride: Int?
    private let prefillChunkThresholdOverride: Int?

    // MARK: - Initialization

    /// Creates a runner by loading a model bundle from a URL.
    public init(
        contentsOf url: URL,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto,
        prefillChunkSize: Int? = nil,
        prefillChunkThreshold: Int? = nil
    ) throws {
        self.init(
            bundle: try LanguageBundle(at: url),
            variant: variant,
            kvCacheStrategy: kvCacheStrategy,
            prefillChunkSize: prefillChunkSize,
            prefillChunkThreshold: prefillChunkThreshold
        )
    }

    /// Creates a runner from a LanguageBundle.
    public init(
        bundle: LanguageBundle,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto,
        prefillChunkSize: Int? = nil,
        prefillChunkThreshold: Int? = nil
    ) {
        self.bundle = bundle
        self.engineVariant = variant
        self.kvCacheStrategy = kvCacheStrategy
        self.prefillChunkSizeOverride = prefillChunkSize
        self.prefillChunkThresholdOverride = prefillChunkThreshold
    }

    // MARK: - Engine Creation

    /// Creates an inference engine using auto-detection.
    public func makeInferenceEngine() async throws -> any InferenceEngine {
        let config = makeConfig()
        let configData = try JSONEncoder().encode(config)

        let resolvedChunkSize = prefillChunkSizeOverride ?? bundle.language.prefillChunkSize
        let resolvedThreshold = prefillChunkThresholdOverride ?? bundle.language.prefillChunkThreshold

        let options = EngineOptions(
            variant: engineVariant,
            kvCacheStrategy: kvCacheStrategy,
            prefillChunkSize: resolvedChunkSize,
            prefillChunkThreshold: resolvedThreshold
        )

        return try await EngineFactory.createEngine(
            config: configData,
            modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main),
            options: options
        )
    }

    // MARK: - Private Helpers

    private func makeConfig() -> ModelConfig {
        let functionName = bundle.language.functionMap?.name(for: "main") ?? "main"
        let modelAsset = bundle.modelAssetPath
        return ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            source: ModelSource(
                hfModelId: bundle.tokenizer,
                modelDefinition: .pyTorch
            ),
            serializedModel: [modelAsset],
            function: functionName
        )
    }
}

#endif  // canImport(CoreAI)
