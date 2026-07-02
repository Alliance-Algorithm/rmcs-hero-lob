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

struct WindowConfig {
    double duration_seconds = 3.0;
    uint32_t max_active_windows = 4;
    TrajectoryWindowConfig trajectory_window = {};
};

struct FrameReference {
    cv::Mat bgr;
    mutable cv::Mat ref_gray;
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
    float x = 0.f;
    float y = 0.f;
    bool found = false;
    uint64_t frame_id = 0;
};

struct VisionImageRef {
    const cv::Mat* image = nullptr;
    uint64_t frame_id = 0;
};

struct WindowResult {
    uint64_t window_id;
    uint32_t slot_number = 0;
    TrajectoryResult trajectory;
    cv::Mat output_image;
    double start_time;
    double end_time;
};

struct WindowProgressUpdate {
    uint64_t window_id = 0;
    uint32_t slot_number = 0;
    uint32_t elapsed_seconds = 0;
    double total_duration_seconds = 0.0;
};

} // namespace rmcs_hero_lob
