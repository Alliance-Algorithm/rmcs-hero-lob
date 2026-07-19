#pragma once

#include <deque>

#include "rmcs_hero_lob/configs.hpp"

#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

class ReferenceFrameSelector {
public:
    explicit ReferenceFrameSelector(const PipelineConfig& config)
        : config_(config) {}

    ReferenceFrameResult Process(const FrameData& frame, const TrackingResult& tracking) {
        anchor_window_.emplace_back(frame, tracking);
        TrimWindow(frame.timestamp_seconds);

        if (!current_reference_.has_reference && frame.IsValid()) {
            current_reference_.mode = ReferenceMode::kStable;
            current_reference_.reference_frame = frame;
            current_reference_.has_reference = true;
            current_reference_.frozen = false;
        }

        return current_reference_;
    }

    void StartTrigger(double trigger_start_time_seconds) {
        current_reference_.trigger_end_time_seconds = trigger_start_time_seconds;
    }

    void Reset() {
        anchor_window_.clear();
        current_reference_ = {};
    }

    void SetReference(const cv::Mat& bgr, int64_t frame_index) {
        FrameData ref;
        ref.frame_index = frame_index;
        ref.bgr = bgr.clone();
        cv::cvtColor(bgr, ref.hsv, cv::COLOR_BGR2HSV);
        ref.timestamp_seconds = 0.0;

        current_reference_.mode = ReferenceMode::kDynamic;
        current_reference_.reference_frame = ref;
        current_reference_.has_reference = true;
        current_reference_.frozen = false;
    }

    ReferenceFrameResult GetReference() const { return current_reference_; }

private:
    void TrimWindow(double current_timestamp_seconds) {
        double cutoff = current_timestamp_seconds - config_.stable_window_seconds;
        while (!anchor_window_.empty() && anchor_window_.front().first.timestamp_seconds < cutoff) {
            anchor_window_.pop_front();
        }
    }

    PipelineConfig config_;
    std::deque<std::pair<FrameData, TrackingResult>> anchor_window_;
    ReferenceFrameResult current_reference_ = {};
};

} // namespace rmcs_hero_lob
