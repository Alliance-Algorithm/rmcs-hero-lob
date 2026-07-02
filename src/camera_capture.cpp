#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include <hikcamera/capturer.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include <rmcs_executor/component.hpp>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class CameraCapture
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    CameraCapture()
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

        if (has_parameter("output_name"))
            output_name_ = get_parameter("output_name").as_string();

        register_output(output_name_, frame_output_);
    }

    ~CameraCapture() override {
        stop_flag_.store(true, std::memory_order_relaxed);
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
    }

    void before_updating() override {
        stop_flag_.store(false, std::memory_order_relaxed);
        capture_thread_ = std::thread(&CameraCapture::capture_thread_func, this);
        RCLCPP_INFO(get_logger(), "Camera capture thread started (target ~%.0f Hz)", camera_fps_);
    }

    void update() override {}

private:
    void capture_thread_func() {
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

            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                buffers_[write_idx_].frame = image.value().clone();
                buffers_[write_idx_].frame_id = frame_id;
                buffers_[write_idx_].valid = true;
            }
            write_idx_.store(1 - write_idx_.load(std::memory_order_relaxed), std::memory_order_release);

            ++frame_id;
        }

        auto disconnect_result = camera_.disconnect();
        if (!disconnect_result) {
            RCLCPP_WARN(
                get_logger(), "Failed to disconnect camera: %s", disconnect_result.error().c_str());
        }
        RCLCPP_INFO(get_logger(), "Camera capture thread stopped");
    }

    hikcamera::Camera camera_;
    double camera_fps_ = 0.0;

    struct FrameBuffer {
        cv::Mat frame;
        uint64_t frame_id = 0;
        bool valid = false;
    };

    FrameBuffer buffers_[2];
    std::atomic<int> write_idx_{0};
    mutable std::mutex buffer_mutex_;

    std::string output_name_ = "/camera_capture/latest_frame";
    OutputInterface<VisionImageRef> frame_output_;

    std::thread capture_thread_;
    std::atomic<bool> stop_flag_{false};
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::CameraCapture, rmcs_executor::Component)
