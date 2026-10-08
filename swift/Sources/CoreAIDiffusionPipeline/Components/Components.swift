// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-10-07
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreGraphics

/// Output from a text encoder — hidden states with optional pooled embedding.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-10-07
@available(iOS 27.0, macOS 27.0, *)
public struct TextEncoderOutput: Sendable {
    /// Token-level embeddings [1, seq_len, hidden_dim].
    public let hiddenStates: NDArray
    /// Sentence-level embedding [1, hidden_dim]. Nil for single-output encoders (e.g. SD 1.5 CLIP-L).
    public let pooledOutput: NDArray?

    public init(hiddenStates: NDArray, pooledOutput: NDArray? = nil) {
        self.hiddenStates = hiddenStates
        self.pooledOutput = pooledOutput
    }
}

#endif  // canImport(CoreAI)
