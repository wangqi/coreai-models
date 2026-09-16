// CoreAI.framework is absent from the iPhoneSimulator SDK; compile the module to empty there
// wangqi modified 2026-09-15
#if canImport(CoreAI)

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Returns the rightmost index in `buffer` such that the suffix from that
/// index to the end is NOT a non-empty prefix of `tag`.
///
/// Used by streaming parsers to decide how much of a buffer can be safely
/// emitted without cutting off a marker that might span two deltas. At most
/// `tag.count - 1` trailing characters are held back (the longest partial
/// prefix that could still complete on the next delta).
// Core AI is iOS 27+ but the app deploys to iOS 18; gate every declaration
// wangqi modified 2026-09-15
@available(iOS 27.0, macOS 27.0, *)
func lastSafeIndex(in buffer: String, forTag tag: String) -> String.Index {
    let maxHold = tag.count - 1
    guard !buffer.isEmpty, maxHold > 0 else { return buffer.endIndex }
    let holdStart = buffer.index(buffer.endIndex, offsetBy: -min(maxHold, buffer.count))
    for offset in 0..<buffer.distance(from: holdStart, to: buffer.endIndex) {
        let idx = buffer.index(holdStart, offsetBy: offset)
        if tag.starts(with: buffer[idx...]) {
            return idx
        }
    }
    return buffer.endIndex
}

#endif  // canImport(CoreAI)
