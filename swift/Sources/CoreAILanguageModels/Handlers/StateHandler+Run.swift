// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Run an inference step with combined primary + secondary states and output.
/// Zero-copy: bind(into:) uses reference-backed storage and _overrideLifetime.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
func runWithStates(
    function: InferenceFunction,
    inputs: [String: NDArray],
    primary: any SyncStateHandler,
    secondary: FixedNDArrayState?,
    outputArray: inout NDArray,
    outputName: String
) async throws {
    var states = InferenceFunction.MutableViews()
    primary.bind(into: &states)
    secondary?.bind(into: &states)

    var outputViews = InferenceFunction.MutableViews()
    outputViews.insert(&outputArray, for: outputName)

    _ = try await function.run(
        inputs: inputs,
        states: _unsafeEscapeMutableViews(consume states),
        outputViews: consume outputViews)
}

/// Run an inference step with no outputs.
///
/// The prefill graph only fills the KV cache, so it declares no outputs and there is
/// nothing to bind. Same states as `runWithStates`.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
func runWithStatesNoOutputs(
    function: InferenceFunction,
    inputs: [String: NDArray],
    primary: any SyncStateHandler,
    secondary: FixedNDArrayState?
) async throws {
    var states = InferenceFunction.MutableViews()
    primary.bind(into: &states)
    secondary?.bind(into: &states)

    _ = try await function.run(
        inputs: inputs,
        states: _unsafeEscapeMutableViews(consume states),
        outputViews: InferenceFunction.MutableViews())
}

#endif  // canImport(CoreAI)
