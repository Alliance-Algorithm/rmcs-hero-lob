#pragma once

#include <deque>

#include "rmcs_hero_lob/configs.hpp"

namespace rmcs_hero_lob {

class ReferenceFrameSelector {
public:
    explicit ReferenceFrameSelector(const PipelineConfig& config);

    ReferenceFrameResult Process(const FrameData& frame, const TrackingResult& tracking);
    void StartTrigger(double trigger_start_time_seconds);
    void Reset();

private:
    void TrimWindow(double current_timestamp_seconds);

    PipelineConfig config_;
    std::deque<std::pair<FrameData, TrackingResult>> anchor_window_;
    ReferenceFrameResult current_reference_ = {};
};

} // namespace rmcs_hero_lob
