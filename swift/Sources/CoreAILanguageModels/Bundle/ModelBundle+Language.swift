// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
extension ModelBundle {
    /// Lossy peek for inspection: returns a `LanguageBundle` if and only if
    /// this bundle's `kind == .llm` and the LLM payload decodes cleanly.
    ///
    /// Returns `nil` for any other kind, missing fields, or malformed JSON.
    /// Strict callers should use `LanguageBundle(at:)` or
    /// `LanguageBundle(bundle:)` directly.
    public var language: LanguageBundle? {
        try? LanguageBundle(bundle: self)
    }
}

#endif  // canImport(CoreAI)
