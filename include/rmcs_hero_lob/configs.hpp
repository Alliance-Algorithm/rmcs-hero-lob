#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

namespace rmcs_hero_lob {

struct CameraFrame {
    uint64_t frame_id;
    cv::Mat image;
};

enum class TargetColor {
    kUnknown,
    kRed,
    kBlue,
};

enum class TrackingState {
    kUninitialized,
    kTracking,
    kLost,
    kPlaceholder,
};

enum class ReferenceMode {
    kUninitialized,
    kStable,
    kDynamic,
    kFrozen,
};

struct ImageRegistratorConfig {
    double max_shift_pixels = 0.0;
    float downscale_factor = 0.5F;
};

struct ImageRegistratorOrbConfig {
    int max_features = 200;
    float match_ratio_threshold = 0.75F;
    double ransac_reproj_threshold = 3.0;
    int min_matches = 10;
    float downscale_factor = 0.5F;
    int exclude_top_left_width = 100;
    int exclude_top_left_height = 100;
};

struct MotionForegroundConfig {
    int min_brightness_value = 128;
    int min_diff_value = 24;
    int open_kernel_size = 3;
    int close_kernel_size = 5;
};

struct TrajectoryWindowConfig {
    double window_seconds = 3.0;
    int min_component_area_pixels = 5;
    float vertical_motion_half_angle_degrees = 40.0F;
    float component_match_max_distance_pixels = 120.0F;
    float velocity_smoothing_alpha = 0.6F;
    float normalization_percentile = 0.99F;
};

struct CompressionConfig {
    int background_output_width = 288;
    int background_output_height = 216;
    int track_output_width = 288;
    int track_output_height = 216;
    int exposure_output_width = 288;
    int exposure_output_height = 216;
};

struct PipelineConfig {
    double stable_window_seconds = 0.5;
    ImageRegistratorConfig image_registrator = {};
    ImageRegistratorOrbConfig image_registrator_orb = {};
    MotionForegroundConfig motion_foreground = {};
    TrajectoryWindowConfig trajectory_window = {};
    CompressionConfig compression = {};
    int process_start_frame = 60;
    int process_end_frame = 300;
    int history_queue_max_size = 15;
    int history_sample_interval = 4;
};

struct FrameData {
    std::int64_t frame_index = -1;
    double timestamp_seconds = 0.0;
    cv::Mat bgr;
    cv::Mat hsv;

    bool IsValid() const { return !bgr.empty(); }
};

struct AnchorSet {
    cv::Point2f guide_center = {};
    cv::Point2f left_top = {};
    cv::Point2f left_bottom = {};
    cv::Point2f right_top = {};
    cv::Point2f right_bottom = {};
    cv::Point2f direction = {1.0F, 0.0F};
    bool valid = false;
    bool placeholder = false;
};

struct DetectionResult {
    AnchorSet anchors = {};
    TargetColor color = TargetColor::kUnknown;
    bool valid = false;
};

struct TrackingResult {
    AnchorSet anchors = {};
    TrackingState state = TrackingState::kUninitialized;
    bool lost = true;
    bool valid = false;
};

struct ReferenceFrameResult {
    ReferenceMode mode = ReferenceMode::kUninitialized;
    FrameData reference_frame = {};
    bool has_reference = false;
    bool frozen = false;
    double trigger_end_time_seconds = 0.0;
};

struct RegistrationResult {
    bool valid = false;
    std::int64_t frame_index = -1;
    double timestamp_seconds = 0.0;
    cv::Matx23f transform = cv::Matx23f(1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
    cv::Mat registered_bgr;
    cv::Mat registered_hsv;
};

struct ForegroundMaskResult {
    bool valid = false;
    std::int64_t frame_index = -1;
    double timestamp_seconds = 0.0;
    cv::Rect roi = {};
    cv::Mat candidate_mask;
    cv::Mat candidate_bgr;
};

struct TrajectoryResult {
    bool valid = false;
    cv::Mat trajectory_layer;
    cv::Mat exposure_count;
    int accumulated_frames = 0;
};

struct SynthesisResult {
    bool valid = false;
    cv::Mat output_image;
};

struct CompressionResult {
    bool valid = false;
    cv::Mat output_image;
};

} // namespace rmcs_hero_lob
