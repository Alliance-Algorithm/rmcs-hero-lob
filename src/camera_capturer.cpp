#include "rmcs_hero_lob/msgs/camera_frame.hpp"
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include <hikcamera/capturer.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include <rmcs_executor/component.hpp>

namespace rmcs_hero_lob {

class CameraCapture
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    CameraCapture()
        : Node{get_component_name(), rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        config_.exposure_us = static_cast<float>(get_parameter("exposure_us").as_double());
        config_.framerate = static_cast<float>(get_parameter("framerate").as_double());
        config_.invert_image = get_parameter("invert_image").as_bool();

        camera_.configure(config_);

        register_output("/hero_lob/camera_frame", camera_frame_, msgs::CameraFrame{});
        register_output("/hero_lob/camera_image", camera_image_, cv::Mat());
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
        RCLCPP_INFO(get_logger(), "Camera capture thread started (target ~%.0f Hz)", config_.framerate);
    }

    void update() override {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!frame_available_)
            return;

        *camera_image_ = latest_frame_;
        camera_frame_->image = latest_frame_;
        camera_frame_->frame_id = latest_frame_id_;
    }

private:
    void capture_thread_func() {
        auto result = camera_.connect();
        if (!result) {
            RCLCPP_ERROR(get_logger(), "Failed to connect camera: %s", result.error().c_str());
            return;
        }

        RCLCPP_INFO(get_logger(), "Camera connected");
        uint64_t frame_id = 0;

        using clock = std::chrono::steady_clock;
        auto fps_last_time = clock::now();
        uint64_t fps_frame_count = 0;

        while (!stop_flag_.load(std::memory_order_relaxed)) {
            auto image = camera_.read_image();

            if (!image) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 1000, "Failed to read image: %s", image.error().c_str());
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                ++frame_id;
                latest_frame_ = image->clone();
                latest_frame_id_ = frame_id;
                frame_available_ = true;
            }

            ++fps_frame_count;

            auto now = clock::now();
            auto elapsed = std::chrono::duration<double>(now - fps_last_time).count();
            if (elapsed >= 1.0) {
                double actual_fps = static_cast<double>(fps_frame_count) / elapsed;
                RCLCPP_INFO(get_logger(), "Camera FPS: %.1f Hz", actual_fps);
                fps_last_time = now;
                fps_frame_count = 0;
            }
        }

        auto disconnect_result = camera_.disconnect();
        if (!disconnect_result) {
            RCLCPP_WARN(get_logger(), "Failed to disconnect camera: %s", disconnect_result.error().c_str());
        }

        RCLCPP_INFO(get_logger(), "Camera capture thread stopped (%" PRIu64 " frames captured)", frame_id);
    }

    hikcamera::Config config_;
    hikcamera::Camera camera_;

    std::thread capture_thread_;
    std::atomic<bool> stop_flag_{false};

    mutable std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    uint64_t latest_frame_id_ = 0;
    bool frame_available_ = false;

    OutputInterface<msgs::CameraFrame> camera_frame_;
    OutputInterface<cv::Mat> camera_image_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::CameraCapture, rmcs_executor::Component)
