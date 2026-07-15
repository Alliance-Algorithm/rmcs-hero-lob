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

struct HsvRangeConfig {
    int hue_min = 0;
    int hue_max = 0;
};

struct BrightnessThresholdConfig {
    int min_value = 220;
};

struct EdgeColorThresholdConfig {
    HsvRangeConfig red_low = {0, 12};
    HsvRangeConfig red_high = {166, 179};
    HsvRangeConfig blue = {95, 130};
    int min_saturation = 20;
    int min_value = 60;
    int edge_band_kernel_size = 5;
    int min_edge_pixels = 12;
    float min_color_ratio = 1.25F;
};

struct MorphologyConfig {
    int blur_kernel_size = 5;
    int open_kernel_size = 3;
    int close_kernel_size = 5;
};

struct GuideConstraintConfig {
    float min_area_ratio = 0.00005F;
    float max_area_ratio = 0.02F;
    float max_aspect_ratio_deviation = 0.45F;
    float min_circularity = 0.65F;
};

struct LightConstraintConfig {
    float min_length_ratio = 0.02F;
    float max_length_ratio = 0.30F;
    float min_width_ratio = 0.003F;
    float max_width_ratio = 0.08F;
    float min_aspect_ratio = 2.0F;
    float max_aspect_ratio = 18.0F;
    float min_fill_ratio = 0.45F;
};

struct StablePairRoiConfig {
    float half_width_radius_scale = 5.0F;
    float top_offset_radius_scale = 1.4F;
    float bottom_offset_radius_scale = 5.5F;
};

struct StableLightConstraintConfig {
    int local_min_value = 130;
    float min_length_ratio_to_roi_height = 0.20F;
    float max_length_ratio_to_roi_height = 0.80F;
    float min_width_ratio_to_roi_width = 0.01F;
    float max_width_ratio_to_roi_width = 0.20F;
    float min_aspect_ratio = 1.40F;
    float max_aspect_ratio = 12.0F;
    float min_fill_ratio = 0.15F;
    float guide_exclusion_radius_scale = 1.20F;
    float max_abs_angle_from_vertical_degrees = 30.0F;
    float min_center_y_offset_radius_scale = 1.4F;
    float max_center_y_offset_radius_scale = 6.5F;
    float min_center_x_offset_radius_scale = 1.0F;
    float max_center_x_offset_radius_scale = 4.0F;
    float center_exclusion_half_width_radius_scale = 0.60F;
    float min_area_radius_scale_squared = 0.005F;
    float max_area_radius_scale_squared = 8.0F;
    int vertical_open_kernel_width = 1;
    int vertical_open_kernel_height = 1;
    int vertical_close_kernel_width = 1;
    int vertical_close_kernel_height = 1;
};

struct StablePairConstraintConfig {
    float max_angle_difference_degrees = 18.0F;
    float max_length_delta_ratio = 1.0F;
    float max_center_y_delta_ratio = 0.30F;
    float max_center_y_delta_radius_scale = 0.8F;
    float max_distance_symmetry_ratio = 0.45F;
    float max_midpoint_x_offset_ratio = 0.60F;
    float max_midpoint_x_offset_radius_scale = 0.8F;
    float min_midpoint_y_offset_radius_scale = 1.4F;
    float max_midpoint_y_offset_radius_scale = 6.5F;
    float min_center_distance_ratio = 0.35F;
    float max_center_distance_ratio = 4.50F;
    float min_center_distance_radius_scale = 1.0F;
    float max_center_distance_radius_scale = 6.0F;
};

struct StablePairFallbackConfig {
    float split_width_ratio_to_roi_width = 0.16F;
    int min_peak_distance_pixels = 6;
    float min_valley_ratio = 0.85F;
};

struct PairConstraintConfig {
    float max_angle_difference_degrees = 12.0F;
    float max_length_delta_ratio = 0.35F;
    float max_center_y_delta_ratio = 0.45F;
    float min_center_distance_ratio = 0.40F;
    float max_center_distance_ratio = 4.50F;
    float max_overlap_ratio = 0.15F;
};

struct TripletConstraintConfig {
    float max_guide_midpoint_x_offset_ratio = 0.90F;
    float min_guide_midpoint_y_offset_ratio = 0.15F;
    float max_guide_midpoint_y_offset_ratio = 2.20F;
    float min_guide_radius_to_light_length_ratio = 0.15F;
    float max_guide_radius_to_light_length_ratio = 1.10F;
};

struct IdentifierConfig {
    BrightnessThresholdConfig brightness = {};
    EdgeColorThresholdConfig edge_color = {};
    MorphologyConfig morphology = {};
    GuideConstraintConfig guide = {};
    LightConstraintConfig light = {};
    StablePairRoiConfig stable_pair_roi = {};
    StableLightConstraintConfig stable_light = {};
    StablePairConstraintConfig stable_pair = {};
    StablePairFallbackConfig stable_pair_fallback = {};
    PairConstraintConfig pair = {};
    TripletConstraintConfig triplet = {};
};

struct MotionForegroundConfig {
    int warmup_frames = 5;
    int min_brightness_value = 128;
    int min_diff_value = 24;
    float background_alpha = 0.05F;
    int open_kernel_size = 3;
    int close_kernel_size = 5;
    int static_bright_value_threshold = 220;
};

struct TrajectoryWindowConfig {
    double window_seconds = 3.0;
    int min_component_area_pixels = 5;
    float vertical_motion_half_angle_degrees = 40.0F;
    float min_motion_pixels = 0.5F;
    float component_match_max_distance_pixels = 120.0F;
    float velocity_smoothing_alpha = 0.6F;
    float normalization_percentile = 0.99F;
    int min_tracking_frames = 5;
    float trajectory_decay = 0.95F;
};

struct PipelineConfig {
    double stable_window_seconds = 0.5;
    double lost_timeout_seconds = 0.2;
    double trigger_window_seconds = 3.0;
    int output_width = 288;
    int output_height = 216;
    IdentifierConfig identifier = {};
    MotionForegroundConfig motion_foreground = {};
    TrajectoryWindowConfig trajectory_window = {};
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

} // namespace rmcs_hero_lob
