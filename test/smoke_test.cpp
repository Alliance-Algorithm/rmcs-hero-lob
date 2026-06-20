#include <cassert>

#include <opencv2/core.hpp>

#include "rmcs_hero_lob/background_remover.hpp"
#include "rmcs_hero_lob/image_synthesis.hpp"
#include "rmcs_hero_lob/processing_window.hpp"
#include "rmcs_hero_lob/window_manager.hpp"

int main() {
    using namespace rmcs_hero_lob;

    MotionForegroundConfig motion_config;
    motion_config.min_brightness_value = 1;
    motion_config.min_diff_value = 1;
    motion_config.open_kernel_size = 0;
    motion_config.close_kernel_size = 0;

    TrajectoryWindowConfig trajectory_config;
    trajectory_config.min_component_area_pixels = 1;
    trajectory_config.component_match_max_distance_pixels = 1000.0F;
    trajectory_config.vertical_motion_half_angle_degrees = 90.0F;

    WindowManager manager(1.0, 2, motion_config, trajectory_config);
    FrameReference ref;
    ref.valid = true;
    ref.bgr = cv::Mat::zeros(4, 4, CV_8UC3);

    auto id = manager.create_window(0.0);
    if (id == 0) {
        return 1;
    }

    cv::Mat frame = ref.bgr.clone();
    frame.at<cv::Vec3b>(1, 1) = cv::Vec3b(255, 255, 255);
    manager.process_frame(ref, frame, 0.5);
    auto results = manager.collect_results(0.5);
    assert(results.empty());

    manager.process_frame(ref, frame, 1.0);
    results = manager.collect_results(1.0);
    assert(!results.empty());
    assert(results.front().window_id == id);
    assert(results.front().output_image.rows == 4);
    assert(results.front().output_image.cols == 4);

    return 0;
}
