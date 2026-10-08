// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-10-07
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Supported scheduler algorithms.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-10-07
@available(iOS 27.0, macOS 27.0, *)
public enum SchedulerType: String, Sendable, CaseIterable {
    case pndm
    case dpmSolverMultistep = "dpmpp"
    case discreteFlow = "flow_match_euler"
}

#endif  // canImport(CoreAI)
