#include <atomic>
#include <cinttypes>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>

#include <opencv2/core.hpp>

#include <hikcamera/capturer.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <std_msgs/msg/empty.hpp>

#include <rmcs_executor/component.hpp>

#include "rmcs_hero_lob/vision_data.hpp"
#include "rmcs_hero_lob/window_manager.hpp"

namespace rmcs_hero_lob {

class VisionPipeline
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    VisionPipeline()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {
        hikcamera::Config cam_config;
        cam_config.exposure_us = get_parameter("exposure_us").as_double();
        cam_config.framerate = get_parameter("framerate").as_double();
        cam_config.gain = get_parameter("gain").as_double();
        cam_config.invert_image = get_parameter("invert_image").as_bool();
        camera_.configure(cam_config);

        camera_fps_ = cam_config.framerate;

        if (has_parameter("min_brightness_value"))
            config_.motion_foreground.min_brightness_value =
                static_cast<int>(get_parameter("min_brightness_value").as_int());
        if (has_parameter("min_diff_value"))
            config_.motion_foreground.min_diff_value =
                static_cast<int>(get_parameter("min_diff_value").as_int());
        if (has_parameter("min_component_area_pixels"))
            config_.trajectory_window.min_component_area_pixels =
                static_cast<int>(get_parameter("min_component_area_pixels").as_int());
        if (has_parameter("vertical_motion_half_angle_degrees"))
            config_.trajectory_window.vertical_motion_half_angle_degrees =
                static_cast<float>(get_parameter("vertical_motion_half_angle_degrees").as_double());
        if (has_parameter("component_match_max_distance_pixels"))
            config_.trajectory_window.component_match_max_distance_pixels = static_cast<float>(
                get_parameter("component_match_max_distance_pixels").as_double());
        if (has_parameter("window_duration"))
            window_duration_ = get_parameter("window_duration").as_double();
        if (has_parameter("max_windows"))
            max_windows_ = static_cast<uint32_t>(get_parameter("max_windows").as_int());

        if (max_windows_ == 0) {
            max_windows_ = 1;
        }

        window_manager_ = std::make_unique<WindowManager>(
            window_duration_, max_windows_, config_.motion_foreground, config_.trajectory_window);

        trigger_sub_ = create_subscription<std_msgs::msg::Empty>(
            "/hero/lob/trigger", 10, [this](const std_msgs::msg::Empty::SharedPtr) {
                trigger_count_.fetch_add(1, std::memory_order_release);
            });

        register_output("/hero/lob/latest_frame", latest_frame_);
        register_output("/hero/lob/processed_image", processed_image_, VisionImageRef{nullptr, 0});
    }

    ~VisionPipeline() override {
        stop_flag_.store(true, std::memory_order_relaxed);
        if (vision_thread_.joinable()) {
            vision_thread_.join();
        }
    }

    void before_updating() override {
        vision_thread_ = std::thread(&VisionPipeline::vision_thread_func, this);
        RCLCPP_INFO(get_logger(), "Vision pipeline thread started (target ~%.0f Hz)", camera_fps_);
    }

    void update() override {
        if (processed_image_ready_) {
            *processed_image_ = VisionImageRef{&processed_image_buffer_, processed_image_frame_id_};
            processed_image_ready_ = false;
        }
    }

private:
    void vision_thread_func() {
        auto result = camera_.connect();
        if (!result) {
            RCLCPP_ERROR(get_logger(), "Failed to connect camera: %s", result.error().c_str());
            return;
        }
        RCLCPP_INFO(get_logger(), "Camera connected");

        uint64_t frame_id = 0;
        while (!stop_flag_.load(std::memory_order_relaxed)) {
            auto image = camera_.read_image();
            if (!image) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 1000, "Failed to read image: %s",
                    image.error().c_str());
                continue;
            }

            if (!reference_frame_.valid) {
                reference_frame_.bgr = image.value().clone();
                reference_frame_.valid = true;
                *latest_frame_ = image.value();
                ++frame_id;
                continue;
            }

            double timestamp = static_cast<double>(frame_id) / camera_fps_;

            while (trigger_count_.load(std::memory_order_acquire) > 0) {
                uint32_t expected = trigger_count_.load(std::memory_order_acquire);
                if (expected == 0) {
                    break;
                }
                if (!trigger_count_.compare_exchange_weak(
                        expected, expected - 1, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    continue;
                }

                auto wid = window_manager_->create_window(timestamp);
                if (wid != 0) {
                    const auto slot_number = window_manager_->slot_number_for(wid);
                    RCLCPP_INFO(
                        get_logger(),
                        "Trigger started for window %u (id=%" PRIu64 ") at %.2fs (duration: %.2fs)",
                        slot_number, wid, timestamp, window_duration_);
                } else {
                    RCLCPP_WARN(get_logger(), "Max windows reached, trigger ignored");
                }
            }

            window_manager_->process_frame(reference_frame_, image.value(), timestamp);
            *latest_frame_ = image.value();

            auto progress_updates = window_manager_->collect_progress_updates(timestamp);
            for (const auto& update : progress_updates) {
                RCLCPP_INFO(
                    get_logger(), "Window %u progress: %us/%.2fs (id=%" PRIu64 ")",
                    update.slot_number, update.elapsed_seconds, update.total_duration_seconds,
                    update.window_id);
            }

            auto results = window_manager_->collect_results(timestamp);
            if (!results.empty()) {
                for (const auto& output : results) {
                    RCLCPP_INFO(
                        get_logger(),
                        "Trigger completed for window %u (id=%" PRIu64
                        ", %.2fs -> %.2fs, accumulated_frames=%d)",
                        output.slot_number, output.window_id, output.start_time, output.end_time,
                        output.trajectory.accumulated_frames);
                    RCLCPP_INFO(
                        get_logger(), "Processed image ready for window %u (id=%" PRIu64 ", %dx%d)",
                        output.slot_number, output.window_id, output.output_image.cols,
                        output.output_image.rows);
                }

                const auto& output = results.back();
                processed_image_buffer_ = output.output_image.clone();
                processed_image_ready_ = !processed_image_buffer_.empty();
                processed_image_frame_id_ = frame_id;
            }

            window_manager_->cleanup(timestamp);

            ++frame_id;
        }

        auto disconnect_result = camera_.disconnect();
        if (!disconnect_result) {
            RCLCPP_WARN(
                get_logger(), "Failed to disconnect camera: %s", disconnect_result.error().c_str());
        }
        RCLCPP_INFO(get_logger(), "Vision pipeline thread stopped");
    }

    hikcamera::Camera camera_;

    HeroLobConfig config_;
    double window_duration_ = 3.0;
    uint32_t max_windows_ = 1;
    std::unique_ptr<WindowManager> window_manager_;
    FrameReference reference_frame_;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr trigger_sub_;
    std::atomic<uint32_t> trigger_count_{0};

    OutputInterface<cv::Mat> latest_frame_;
    OutputInterface<VisionImageRef> processed_image_;
    cv::Mat processed_image_buffer_;
    uint64_t processed_image_frame_id_{0};
    bool processed_image_ready_{false};

    std::thread vision_thread_;
    std::atomic<bool> stop_flag_{false};

    double camera_fps_ = 0.0;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::VisionPipeline, rmcs_executor::Component)
