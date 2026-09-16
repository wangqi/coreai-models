// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Async sequence of raw token IDs from constrained generation.
///
/// Concrete counterpart to `InferenceOutputSequence` for paths that yield
/// bare token IDs rather than full `InferenceOutput` values.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
public struct InferenceTokenSequence: AsyncSequence, Sendable {
    public typealias Element = Int32

    private let stream: AsyncThrowingStream<Int32, Error>
    private let _stopReasonStore: StopReasonStore

    init(stream: AsyncThrowingStream<Int32, Error>, stopReasonStore: StopReasonStore) {
        self.stream = stream
        self._stopReasonStore = stopReasonStore
    }

    public var stopReason: StopReason? { _stopReasonStore.stopReason }

    public func setStopReason(_ reason: StopReason) {
        _stopReasonStore.set(reason)
    }

    public func makeAsyncIterator() -> AsyncThrowingStream<Int32, Error>.AsyncIterator {
        stream.makeAsyncIterator()
    }
}

#endif  // canImport(CoreAI)
