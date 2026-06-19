#include <array>
#include <atomic>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include <rmcs_executor/component.hpp>
#include <rmcs_utility/double_buffer.hpp>

#include "rmcs_hero_lob/vision_data.hpp"
#include "rmcs_hero_lob/background_remover.hpp"
#include "rmcs_hero_lob/tracker_processor.hpp"
#include "rmcs_hero_lob/image_synthesis.hpp"

namespace rmcs_hero_lob {

class VisionPipeline
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    VisionPipeline()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)},
          background_remover_(config_.motion_foreground),
          tracker_processor_(config_.trajectory_window),
          image_synthesis_(config_.trajectory_window) {
        camera_device_ = get_parameter("camera_device").as_int();
        camera_width_ = get_parameter("camera_width").as_int();
        camera_height_ = get_parameter("camera_height").as_int();
        camera_fps_ = get_parameter("camera_fps").as_double();

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
            config_.trajectory_window.component_match_max_distance_pixels =
                static_cast<float>(get_parameter("component_match_max_distance_pixels").as_double());

        register_output(
            "/hero/lob/vision_target", vision_target_, VisionTarget{0.f, 0.f, false, 0});
        register_output(
            "/hero/lob/vision_image", vision_image_,
            VisionImageRef{static_cast<const cv::Mat*>(nullptr), 0});
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
        VisionTarget tmp;
        if (target_buffer_.read(tmp)) {
            last_target_ = tmp;
        }
        *vision_target_ = last_target_;

        uint64_t fid = image_frame_id_.load(std::memory_order_acquire);
        if (fid != last_image_frame_id_) {
            image_read_slot_ = 1 - image_read_slot_;
            last_image_frame_id_ = fid;
        }

        VisionImageRef ref;
        ref.image = &image_slots_[image_read_slot_].image;
        ref.frame_id = fid;
        *vision_image_ = ref;
    }

private:
    void vision_thread_func() {
        cv::VideoCapture cap(camera_device_);
        if (!cap.isOpened()) {
            RCLCPP_ERROR(get_logger(), "Failed to open camera device %d", camera_device_);
            return;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
        cap.set(cv::CAP_PROP_FPS, camera_fps_);

        uint64_t frame_id = 0;
        while (!stop_flag_.load(std::memory_order_relaxed)) {
            cv::Mat frame;
            if (!cap.read(frame))
                continue;

            cv::Mat output = process_frame(frame);

            target_buffer_.write(VisionTarget{0.f, 0.f, false, frame_id});

            int ws = image_write_slot_.load(std::memory_order_relaxed);
            if (!image_slots_[ws].allocated) {
                image_slots_[ws].image.create(output.rows, output.cols, output.type());
                image_slots_[ws].allocated = true;
            }
            output.copyTo(image_slots_[ws].image);

            image_write_slot_.store(1 - ws, std::memory_order_release);
            image_frame_id_.store(frame_id, std::memory_order_release);
            ++frame_id;
        }

        RCLCPP_INFO(get_logger(), "Vision pipeline thread stopped");
    }

    cv::Mat process_frame(const cv::Mat& frame) {
        if (!reference_frame_.valid) {
            reference_frame_.bgr = frame.clone();
            reference_frame_.valid = true;
            tracker_processor_.reset();
            return frame;
        }

        ++frame_index_;
        double timestamp = static_cast<double>(frame_index_) / camera_fps_;

        tracker_processor_.set_timestamp(timestamp);

        ForegroundResult fg = background_remover_.process(reference_frame_, frame);
        TrajectoryResult traj = tracker_processor_.process(fg);
        SynthesisResult synth = image_synthesis_.process(reference_frame_, traj);
        return synth.valid ? synth.output_image : frame;
    }

    HeroLobConfig config_;
    BackgroundRemover background_remover_;
    TrackerProcessor tracker_processor_;
    ImageSynthesis image_synthesis_;
    FrameReference reference_frame_;
    int64_t frame_index_ = -1;

    OutputInterface<VisionTarget> vision_target_;
    OutputInterface<VisionImageRef> vision_image_;

    rmcs_utility::DoubleBuffer<VisionTarget> target_buffer_;
    VisionTarget last_target_{0.f, 0.f, false, 0};

    struct ImageSlot {
        cv::Mat image;
        bool allocated{false};
    };
    std::array<ImageSlot, 2> image_slots_;
    std::atomic<int> image_write_slot_{0};
    std::atomic<uint64_t> image_frame_id_{0};
    int image_read_slot_{1};
    uint64_t last_image_frame_id_{0};

    std::thread vision_thread_;
    std::atomic<bool> stop_flag_{false};

    int camera_device_;
    int camera_width_;
    int camera_height_;
    double camera_fps_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::VisionPipeline, rmcs_executor::Component)
