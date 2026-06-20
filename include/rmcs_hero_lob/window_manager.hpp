#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rmcs_hero_lob/background_remover.hpp"
#include "rmcs_hero_lob/image_synthesis.hpp"
#include "rmcs_hero_lob/processing_window.hpp"
#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class WindowManager {
public:
    using WindowId = ProcessingWindow::WindowId;

    WindowManager(double window_duration, uint32_t max_windows,
                  const MotionForegroundConfig& motion_config,
                  const TrajectoryWindowConfig& trajectory_config)
        : window_duration_(window_duration)
        , max_windows_(max_windows)
        , trajectory_config_(trajectory_config)
        , background_remover_(motion_config)
        , image_synthesis_(trajectory_config) {}

    WindowId create_window(double current_time) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (active_count_locked() >= max_windows_) {
            return 0;
        }

        WindowId id = ++next_id_;
        windows_[id] = std::make_unique<ProcessingWindow>(
            id, current_time, window_duration_, trajectory_config_);
        return id;
    }

    void process_frame(const FrameReference& ref, const cv::Mat& frame, double current_time) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [id, window] : windows_) {
            (void)id;
            window->process(ref, frame, background_remover_, current_time);
        }
    }

    std::vector<WindowResult> collect_results(double current_time) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<WindowResult> results;
        for (auto& [id, window] : windows_) {
            if (!window->is_completed(current_time) || window->is_consumed()) {
                continue;
            }

            WindowResult result;
            result.window_id = window->id();
            result.trajectory = window->result();
            result.start_time = window->start_time();
            result.end_time = window->end_time();

            auto synth = image_synthesis_.process(window->reference_frame(), result.trajectory);
            result.output_image =
                synth.valid ? synth.output_image : window->reference_frame().bgr.clone();

            results.push_back(std::move(result));
            window->consume();
        }
        return results;
    }

    void cleanup(double current_time, double grace_period = 1.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = windows_.begin(); it != windows_.end();) {
            auto& window = it->second;
            if (window->is_consumed() && current_time > window->end_time() + grace_period) {
                it = windows_.erase(it);
            } else {
                ++it;
            }
        }
    }

    size_t active_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_count_locked();
    }

    bool can_create() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_count_locked() < max_windows_;
    }

private:
    size_t active_count_locked() const {
        size_t count = 0;
        for (const auto& [id, window] : windows_) {
            (void)id;
            if (!window->is_consumed()) {
                ++count;
            }
        }
        return count;
    }

    double window_duration_;
    uint32_t max_windows_;
    TrajectoryWindowConfig trajectory_config_;

    BackgroundRemover background_remover_;
    ImageSynthesis image_synthesis_;

    std::unordered_map<WindowId, std::unique_ptr<ProcessingWindow>> windows_;
    WindowId next_id_ = 0;

    mutable std::mutex mutex_;
};

} // namespace rmcs_hero_lob
