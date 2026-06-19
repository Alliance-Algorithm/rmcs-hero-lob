#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

namespace rmcs_hero_lob {

struct MotionForegroundConfig {
    int min_brightness_value = 128;
    int min_diff_value = 24;
    int open_kernel_size = 3;
    int close_kernel_size = 5;
};

struct TrajectoryWindowConfig {
    int min_component_area_pixels = 5;
    float vertical_motion_half_angle_degrees = 40.0F;
    float component_match_max_distance_pixels = 120.0F;
    float velocity_smoothing_alpha = 0.6F;
    float normalization_percentile = 0.99F;
};

struct HeroLobConfig {
    MotionForegroundConfig motion_foreground = {};
    TrajectoryWindowConfig trajectory_window = {};
};

struct FrameReference {
    cv::Mat bgr;
    bool valid = false;
};

struct ForegroundResult {
    bool valid = false;
    cv::Mat candidate_mask;
    cv::Mat candidate_bgr;
};

struct TrajectoryResult {
    bool valid = false;
    cv::Mat trajectory_layer;
    int accumulated_frames = 0;
};

struct SynthesisResult {
    bool valid = false;
    cv::Mat output_image;
};

struct VisionTarget {
    float x;
    float y;
    bool found;
    uint64_t frame_id;
};

struct VisionImageRef {
    const cv::Mat* image;
    uint64_t frame_id;
};

} // namespace rmcs_hero_lob
