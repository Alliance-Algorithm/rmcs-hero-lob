#pragma once

#include <cmath>
#include <cstdint>
#include <optional>

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

    ProcessingWindow(WindowId id, uint32_t slot_number, double start_time, double duration,
                     const TrajectoryWindowConfig& config)
        : id_(id)
        , slot_number_(slot_number)
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
    bool is_accumulating() const { return state_ == State::Accumulating; }

    const TrajectoryResult& result() const { return result_; }
    const FrameReference& reference_frame() const { return reference_frame_; }

    void consume() {
        if (state_ == State::Completed) {
            state_ = State::Consumed;
        }
    }

    WindowId id() const { return id_; }
    uint32_t slot_number() const { return slot_number_; }
    State state() const { return state_; }
    double start_time() const { return start_time_; }
    double end_time() const { return end_time_; }
    double duration() const { return end_time_ - start_time_; }

    std::optional<uint32_t> take_progress_second(double current_time) {
        if (state_ != State::Accumulating) {
            return std::nullopt;
        }

        double elapsed = current_time - start_time_;
        if (elapsed < 1.0) {
            return std::nullopt;
        }

        auto elapsed_seconds = static_cast<uint32_t>(std::floor(elapsed));
        if (elapsed_seconds == 0 || elapsed_seconds == last_logged_elapsed_second_) {
            return std::nullopt;
        }

        last_logged_elapsed_second_ = elapsed_seconds;
        return elapsed_seconds;
    }

private:
    WindowId id_;
    uint32_t slot_number_;
    double start_time_;
    double end_time_;
    State state_;

    TrackerProcessor tracker_;
    FrameReference reference_frame_;
    bool reference_set_ = false;
    TrajectoryResult result_;
    uint32_t last_logged_elapsed_second_ = 0;
};

} // namespace rmcs_hero_lob
