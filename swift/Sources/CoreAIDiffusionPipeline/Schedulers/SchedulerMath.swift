// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-10-07
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate

/// Compute weighted sum of Float arrays of equal length using BLAS.
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-10-07
@available(iOS 27.0, macOS 27.0, *)
func weightedSum(_ weights: [Float], _ values: [[Float]]) -> [Float] {
    precondition(!values.isEmpty && weights.count == values.count)
    let count = values[0].count
    assert(values.allSatisfy { $0.count == count })
    var result = [Float](repeating: 0.0, count: count)
    for i in 0..<values.count {
        let w = weights[i]
        values[i].withUnsafeBufferPointer { buf in
            cblas_saxpy(Int32(count), w, buf.baseAddress, 1, &result, 1)
        }
    }
    return result
}

/// Double-precision weights overload (DPM-Solver uses Double internally).
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-10-07
@available(iOS 27.0, macOS 27.0, *)
func weightedSum(_ weights: [Double], _ values: [[Float]]) -> [Float] {
    weightedSum(weights.map(Float.init), values)
}

/// Evenly spaced floats between [start, end].
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-10-07
@available(iOS 27.0, macOS 27.0, *)
func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
    guard count > 1 else { return count == 1 ? [start] : [] }
    let scale = (end - start) / Float(count - 1)
    return (0..<count).map { Float($0) * scale + start }
}

// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-10-07
@available(iOS 27.0, macOS 27.0, *)
extension Array {
    subscript(back index: Int) -> Element {
        self[count - index]
    }
}

#endif  // canImport(CoreAI)
