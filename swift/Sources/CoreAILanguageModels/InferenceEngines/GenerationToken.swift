// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// A token representing an active generation session.
///
/// Created by `generate()`, held by the iterator. The engine retains a
/// reference to the active token and can cancel it at any time. The iterator
/// checks `isCancelled` on each `next()` call.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
public final class GenerationToken: Sendable {
    private let _cancelled = Mutex(false)

    public var isCancelled: Bool { _cancelled.withLock { $0 } }

    public func cancel() { _cancelled.withLock { $0 = true } }
}

#endif  // canImport(CoreAI)
