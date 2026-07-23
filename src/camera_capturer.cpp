#include "rmcs_hero_lob/configs.hpp"
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

        config_.camera_name = get_parameter_or<std::string>("camera_name", "");
        config_.exposure_us = static_cast<float>(get_parameter("exposure_us").as_double());
        config_.framerate = static_cast<float>(get_parameter("framerate").as_double());
        config_.invert_image = get_parameter("invert_image").as_bool();

        RCLCPP_INFO(get_logger(), "Configured camera_name: '%s'", config_.camera_name.c_str());
        if (auto names = hikcamera::list_camera_names()) {
            RCLCPP_INFO(get_logger(), "Discovered %zu Hik camera(s)", names->size());
            for (std::size_t index = 0; index < names->size(); ++index) {
                RCLCPP_INFO(get_logger(), "  [%zu] UserDefinedName: '%s'", index, (*names)[index].c_str());
            }
        } else {
            RCLCPP_WARN(get_logger(), "Failed to enumerate Hik cameras: %s", names.error().c_str());
        }

        camera_.configure(config_);

        register_output("/hero_lob/camera_frame", camera_frame_, CameraFrame{});
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

        CameraFrame camera_frame;

        camera_frame.frame_id = latest_frame_id_;
        camera_frame.image = latest_frame_.clone();

        *camera_frame_ = camera_frame;
    }

private:
    void capture_thread_func() {
        uint64_t frame_id = 0;
        using clock = std::chrono::steady_clock;
        auto status_last_time = clock::now();
        uint64_t status_frame_count = 0;
        bool connected = false;

        while (!stop_flag_.load(std::memory_order_relaxed)) {
            if (!connected) {
                auto result = camera_.connect();
                if (!result) {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Failed to connect camera: %s | camera_name='%s' exposure_us=%.1f "
                        "framerate=%.1f invert_image=%s discovered=[%s]",
                        result.error().c_str(), config_.camera_name.c_str(), config_.exposure_us,
                        config_.framerate, config_.invert_image ? "true" : "false",
                        format_discovered_camera_names().c_str());
                    sleep_for_interruptible(std::chrono::seconds(5));
                    continue;
                }

                connected = true;
                status_last_time = clock::now();
                status_frame_count = 0;
                RCLCPP_INFO(
                    get_logger(),
                    "Camera connected | camera_name='%s' exposure_us=%.1f framerate=%.1f",
                    config_.camera_name.c_str(), config_.exposure_us, config_.framerate);
                continue;
            }

            auto image = camera_.read_image();
            if (!image) {
                RCLCPP_WARN(
                    get_logger(),
                    "Failed to read image: %s | camera_name='%s' (will reconnect in 5s)",
                    image.error().c_str(), config_.camera_name.c_str());
                std::ignore = camera_.disconnect();
                connected = false;
                sleep_for_interruptible(std::chrono::seconds(5));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                ++frame_id;
                latest_frame_ = image->clone();
                latest_frame_id_ = frame_id;
                frame_available_ = true;
            }

            ++status_frame_count;

            auto now = clock::now();
            auto elapsed = std::chrono::duration<double>(now - status_last_time).count();
            if (elapsed >= 5.0) {
                const auto actual_fps =
                    elapsed > 0.0 ? static_cast<double>(status_frame_count) / elapsed : 0.0;
                RCLCPP_INFO(get_logger(), "connect frame:%.0f", actual_fps);
                status_last_time = now;
                status_frame_count = 0;
            }
        }

        if (connected) {
            auto disconnect_result = camera_.disconnect();
            if (!disconnect_result) {
                RCLCPP_WARN(
                    get_logger(), "Failed to disconnect camera: %s",
                    disconnect_result.error().c_str());
            }
        }

        RCLCPP_INFO(
            get_logger(), "Camera capture thread stopped (%" PRIu64 " frames captured)", frame_id);
    }

    auto format_discovered_camera_names() const -> std::string {
        auto names = hikcamera::list_camera_names();
        if (!names) {
            return std::string{"enum_failed: "} + names.error();
        }

        if (names->empty()) {
            return "none";
        }

        std::string joined;
        for (std::size_t index = 0; index < names->size(); ++index) {
            if (index != 0) {
                joined += ", ";
            }
            joined += '\'';
            joined += (*names)[index];
            joined += '\'';
        }
        return joined;
    }

    void sleep_for_interruptible(std::chrono::steady_clock::duration duration) {
        using clock = std::chrono::steady_clock;
        const auto deadline = clock::now() + duration;
        while (!stop_flag_.load(std::memory_order_relaxed) && clock::now() < deadline) {
            const auto remaining = deadline - clock::now();
            const auto slice = std::min(
                remaining, std::chrono::duration_cast<clock::duration>(std::chrono::milliseconds(100)));
            std::this_thread::sleep_for(slice);
        }
    }

    hikcamera::Config config_;
    hikcamera::Camera camera_;

    std::thread capture_thread_;
    std::atomic<bool> stop_flag_{false};

    mutable std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    uint64_t latest_frame_id_ = 0;
    bool frame_available_ = false;

    OutputInterface<CameraFrame> camera_frame_;
    OutputInterface<cv::Mat> camera_image_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::CameraCapture, rmcs_executor::Component)
