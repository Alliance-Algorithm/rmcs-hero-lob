#pragma once

namespace rmcs_hero_lob {

constexpr int kUnsetSequence = -1;
constexpr int kMaxSequenceValue = 255;

inline int advance_sequence(const int sequence) {
    if (sequence == kUnsetSequence)
        return 0;
    if (sequence < 0 || sequence >= kMaxSequenceValue)
        return 0;
    return sequence + 1;
}

} // namespace rmcs_hero_lob
