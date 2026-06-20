#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

#include "rmcs_hero_lob/background_remover.hpp"
#include "rmcs_hero_lob/tracker_processor.hpp"
#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class ProcessingWindow {
public:
    using WindowId = uint64_t;

    enum class State {
        Accumulating,
        Completed,
        Consumed
    };

    ProcessingWindow(WindowId id, double start_time, double duration,
                     const TrajectoryWindowConfig& config)
        : id_(id)
        , start_time_(start_time)
        , end_time_(start_time + duration)
        , state_(State::Accumulating)
        , tracker_(config) {}

    void process(const FrameReference& ref, const cv::Mat& frame,
                 const BackgroundRemover& bg_remover, double current_time) {
        if (state_ != State::Accumulating) {
            return;
        }

        if (!reference_set_) {
            reference_frame_ = ref;
            reference_set_ = true;
        }

        if (current_time >= end_time_) {
            state_ = State::Completed;
            return;
        }

        tracker_.set_timestamp(current_time - start_time_);
        ForegroundResult fg = bg_remover.process(reference_frame_, frame);
        result_ = tracker_.process(fg);
    }

    bool is_completed(double current_time) const {
        return state_ == State::Completed || current_time >= end_time_;
    }

    bool is_consumed() const { return state_ == State::Consumed; }

    const TrajectoryResult& result() const { return result_; }
    const FrameReference& reference_frame() const { return reference_frame_; }

    void consume() {
        if (state_ == State::Completed) {
            state_ = State::Consumed;
        }
    }

    WindowId id() const { return id_; }
    State state() const { return state_; }
    double start_time() const { return start_time_; }
    double end_time() const { return end_time_; }

private:
    WindowId id_;
    double start_time_;
    double end_time_;
    State state_;

    TrackerProcessor tracker_;
    FrameReference reference_frame_;
    bool reference_set_ = false;
    TrajectoryResult result_;
};

} // namespace rmcs_hero_lob
