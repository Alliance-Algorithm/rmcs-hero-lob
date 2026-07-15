#pragma once

#include <vector>

#include "rmcs_hero_lob/configs.hpp"

namespace rmcs_hero_lob {

class TrackerProcessorFast {
public:
    struct ComponentInfo {
        cv::Rect bbox;
        cv::Point2f centroid = {};
        cv::Point2f smoothed_velocity = {};
        bool velocity_initialized = false;
    };

    explicit TrackerProcessorFast(const PipelineConfig& config);

    TrajectoryResult Process(const ForegroundMaskResult& foreground_mask);
    void Reset();

private:
    PipelineConfig config_;
    std::vector<ComponentInfo> previous_components_;
    cv::Mat trajectory_layer_;
    cv::Mat exposure_count_;
    cv::Mat mask_buffer_;
    int frame_count_ = 0;
    double previous_timestamp_ = 0.0;
};

} // namespace rmcs_hero_lob
