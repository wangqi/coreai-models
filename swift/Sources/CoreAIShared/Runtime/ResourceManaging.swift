// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Lifecycle management for lazily-loaded resources (models, buffers).
///
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
public protocol ResourceManaging: Sendable {
    /// Load model weights and allocate inference buffers.
    func loadResources() async throws
    /// Release all resources. Safe to call multiple times.
    func unloadResources() async
}

// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
extension ResourceManaging {
    public func prewarmResources() async throws {
        try await loadResources()
        await unloadResources()
    }
}

#endif  // canImport(CoreAI)
