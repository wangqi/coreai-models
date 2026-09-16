// swift-tools-version: 6.0

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Trimmed for the AIAssistant app.  Three deliberate divergences from upstream, all needed to make
// the package linkable from an app that deploys to iOS 18.0 / macOS 26.0:
//
//   1. `platforms:` is lowered.  Upstream declares .iOS("27.0")/.macOS("27.0") and carries no
//      @available annotations at all, so SPM refuses to link it.  Instead every declaration in
//      CoreAIShared, CoreAILanguageModels and CoreAISpeech carries
//      @available(iOS 27.0, macOS 27.0, *), applied mechanically by
//      helper/scripts/coreai/patch_coreai_package.py.  Re-run that script after every upstream merge.
//   2. swift-transformers resolves to thirdparty/swift-transformers, the copy the app already
//      vendors, so the app links one Tokenizers module rather than two.
//   3. The CLI executables, their test targets and the diffusion / segmentation / object-detection
//      libraries are removed.  llm-server pulled in hummingbird and every tool pulled in
//      swift-argument-parser, none of which belong in an app binary.  The engine and streaming code
//      the app ports from llm-server's ChatHandler lives in AIChatModelCoreAI.swift instead.
//
// wangqi modified 2026-09-15

import PackageDescription

let package = Package(
    name: "coreai-models",
    platforms: [.macOS("26.0"), .iOS("18.0")],
    products: [
        .library(
            name: "CoreAILM",
            targets: [
                "CoreAILanguageModels"
            ]
        ),
        .library(
            name: "CoreAISpeech",
            targets: ["CoreAISpeech"]
        ),
    ],
    dependencies: [
        .package(path: "../swift-transformers"),
        .package(url: "https://github.com/mlc-ai/xgrammar", exact: "0.2.2"),
    ],
    targets: [
        .target(
            name: "CoreAILanguageModels",
            dependencies: [
                "CoreAIShared",
                "CXGrammar",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Sources/CoreAILanguageModels",
            swiftSettings: [
                .define("CXGRAMMAR_IMPORT"),
                .enableUpcomingFeature("MemberImportVisibility"),
                .enableExperimentalFeature("Lifetimes"),
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),

        // Shared utilities
        .target(
            name: "CoreAIShared",
            dependencies: [],
            path: "swift/Sources/CoreAIShared",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // Speech recognition library
        .target(
            name: "CoreAISpeech",
            dependencies: [
                "CoreAIShared",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Sources/CoreAISpeech",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // CXGrammar C bridge
        .target(
            name: "CXGrammar",
            dependencies: [
                .product(name: "XGrammar", package: "xgrammar")
            ],
            path: "swift/Sources/lib/CXGrammar",
            publicHeadersPath: "include"
        ),
    ],
    // xgrammar's headers are C++17 (std::variant, std::optional).  Upstream never declared a
    // standard because its own CLI builds picked one up from CMake; inside Xcode the SPM C++ target
    // compiles at the clang default and fails.  wangqi modified 2026-09-15
    cxxLanguageStandard: .cxx17
)
