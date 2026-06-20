#pragma once

#include <algorithm>
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
        const uint32_t slot_number = allocate_slot_locked();
        windows_[id] = std::make_unique<ProcessingWindow>(
            id, slot_number, current_time, window_duration_, trajectory_config_);
        return id;
    }

    void process_frame(const FrameReference& ref, const cv::Mat& frame, double current_time) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [id, window] : windows_) {
            (void)id;
            window->process(ref, frame, background_remover_, current_time);
        }
    }

    std::vector<WindowProgressUpdate> collect_progress_updates(double current_time) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<WindowProgressUpdate> updates;
        for (auto& [id, window] : windows_) {
            (void)id;
            auto elapsed_seconds = window->take_progress_second(current_time);
            if (!elapsed_seconds.has_value()) {
                continue;
            }

            updates.push_back(WindowProgressUpdate{
                .window_id = window->id(),
                .slot_number = window->slot_number(),
                .elapsed_seconds = *elapsed_seconds,
                .total_duration_seconds = window->duration(),
            });
        }
        return updates;
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
            result.slot_number = window->slot_number();
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

    uint32_t slot_number_for(WindowId id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = windows_.find(id);
        if (it == windows_.end()) {
            return 0;
        }
        return it->second->slot_number();
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
    uint32_t allocate_slot_locked() const {
        std::vector<bool> used_slots(max_windows_ + 1, false);
        for (const auto& [id, window] : windows_) {
            (void)id;
            if (!window->is_consumed()) {
                used_slots[window->slot_number()] = true;
            }
        }

        for (uint32_t slot = 1; slot <= max_windows_; ++slot) {
            if (!used_slots[slot]) {
                return slot;
            }
        }

        return 0;
    }

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
