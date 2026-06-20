#include <algorithm>
#include <array>
#include <atomic>
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
#include <rmcs_utility/double_buffer.hpp>

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
        max_windows_ = std::min(max_windows_, MAX_WINDOWS);

        window_manager_ = std::make_unique<WindowManager>(
            window_duration_, max_windows_, config_.motion_foreground, config_.trajectory_window);

        trigger_sub_ = create_subscription<std_msgs::msg::Empty>(
            "/hero/lob/trigger", 10, [this](const std_msgs::msg::Empty::SharedPtr) {
                trigger_count_.fetch_add(1, std::memory_order_release);
            });

        for (uint32_t i = 0; i < max_windows_; ++i) {
            std::string ns = "/hero/lob/window_" + std::to_string(i);
            register_output(
                ns + "/vision_target", window_targets_[i], VisionTarget{0.f, 0.f, false, 0});
            register_output(ns + "/vision_image", window_images_[i], VisionImageRef{nullptr, 0});
        }

        register_output("/hero/lob/latest_frame", latest_frame_);
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
        for (uint32_t i = 0; i < max_windows_; ++i) {
            WindowPublication tmp;
            if (result_buffers_[i].read(tmp)) {
                last_results_[i] = tmp;
            }

            if (last_results_[i].valid) {
                *window_targets_[i] = last_results_[i].target;
                *window_images_[i] = last_results_[i].image_ref;
            }
        }
    }

private:
    using WindowId = ProcessingWindow::WindowId;

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

                WindowId wid = window_manager_->create_window(timestamp);
                if (wid != 0) {
                    RCLCPP_INFO(get_logger(), "Created window %lu at %.2fs", wid, timestamp);
                } else {
                    RCLCPP_WARN(get_logger(), "Max windows reached, trigger ignored");
                }
            }

            window_manager_->process_frame(reference_frame_, image.value(), timestamp);
            *latest_frame_ = image.value();

            auto results = window_manager_->collect_results(timestamp);

            for (const auto& output : results) {
                uint32_t idx = (output.window_id - 1) % max_windows_;

                WindowPublication publication;
                publication.valid = true;
                publication.start_time = output.start_time;
                publication.end_time = output.end_time;
                publication.window_id = output.window_id;
                publication.target = VisionTarget{0.f, 0.f, false, frame_id};

                int ws = image_write_slot_[idx].load(std::memory_order_relaxed);
                if (!image_slots_[idx][ws].allocated) {
                    image_slots_[idx][ws].image.create(
                        output.output_image.rows, output.output_image.cols,
                        output.output_image.type());
                    image_slots_[idx][ws].allocated = true;
                }
                output.output_image.copyTo(image_slots_[idx][ws].image);

                image_write_slot_[idx].store(1 - ws, std::memory_order_release);
                image_frame_id_[idx].store(frame_id, std::memory_order_release);

                publication.image_ref.image = &image_slots_[idx][ws].image;
                publication.image_ref.frame_id = frame_id;

                result_buffers_[idx].write(publication);

                RCLCPP_INFO(
                    get_logger(), "Window %lu completed, routed to index %u", output.window_id,
                    idx);
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
    uint32_t max_windows_ = 4;
    std::unique_ptr<WindowManager> window_manager_;
    FrameReference reference_frame_;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr trigger_sub_;
    std::atomic<uint32_t> trigger_count_{0};

    static constexpr uint32_t MAX_WINDOWS = 8;
    std::array<OutputInterface<VisionTarget>, MAX_WINDOWS> window_targets_;
    std::array<OutputInterface<VisionImageRef>, MAX_WINDOWS> window_images_;
    OutputInterface<cv::Mat> latest_frame_;

    std::array<rmcs_utility::DoubleBuffer<WindowPublication>, MAX_WINDOWS> result_buffers_;
    std::array<WindowPublication, MAX_WINDOWS> last_results_{};

    struct ImageSlot {
        cv::Mat image;
        bool allocated{false};
    };
    std::array<std::array<ImageSlot, 2>, MAX_WINDOWS> image_slots_{};
    std::array<std::atomic<int>, MAX_WINDOWS> image_write_slot_{};
    std::array<std::atomic<uint64_t>, MAX_WINDOWS> image_frame_id_{};
    std::array<uint64_t, MAX_WINDOWS> last_image_frame_id_{};

    std::thread vision_thread_;
    std::atomic<bool> stop_flag_{false};

    double camera_fps_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::VisionPipeline, rmcs_executor::Component)
