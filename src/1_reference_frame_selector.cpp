#include "rmcs_hero_lob/1_reference_frame_selector.hpp"

#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

ReferenceFrameSelector::ReferenceFrameSelector(const PipelineConfig& config)
    : config_(config) {}

ReferenceFrameResult ReferenceFrameSelector::Process(
    const FrameData& frame, const TrackingResult& tracking) {
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

void ReferenceFrameSelector::StartTrigger(double trigger_start_time_seconds) {
    current_reference_.trigger_end_time_seconds = trigger_start_time_seconds;
}

void ReferenceFrameSelector::Reset() {
    anchor_window_.clear();
    current_reference_ = {};
}

void ReferenceFrameSelector::SetReference(const cv::Mat& bgr, int64_t frame_index) {
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

ReferenceFrameResult ReferenceFrameSelector::GetReference() const {
    return current_reference_;
}

void ReferenceFrameSelector::TrimWindow(double current_timestamp_seconds) {
    double cutoff = current_timestamp_seconds - config_.stable_window_seconds;
    while (!anchor_window_.empty() && anchor_window_.front().first.timestamp_seconds < cutoff) {
        anchor_window_.pop_front();
    }
}

} // namespace rmcs_hero_lob
