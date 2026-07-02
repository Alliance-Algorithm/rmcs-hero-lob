#include <atomic>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include <rmcs_executor/component.hpp>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class VideoReader
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    VideoReader()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {
        video_path_ = get_parameter("video_path").as_string();
        target_fps_ = get_parameter("target_fps").as_double();
        loop_ = get_parameter("loop").as_bool();

        if (has_parameter("output_name"))
            output_name_ = get_parameter("output_name").as_string();

        register_output(output_name_, frame_output_);

        RCLCPP_INFO(
            get_logger(), "VideoReader configured: path=%s, target_fps=%.1f, loop=%s",
            video_path_.c_str(), target_fps_, loop_ ? "true" : "false");
    }

    ~VideoReader() override {
        stop_flag_.store(true, std::memory_order_relaxed);
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
    }

    void before_updating() override {
        stop_flag_.store(false, std::memory_order_relaxed);
        reader_thread_ = std::thread(&VideoReader::reader_thread_func, this);
    }

    void update() override {}

private:
    void reader_thread_func() {
        cv::VideoCapture cap(video_path_);
        if (!cap.isOpened()) {
            RCLCPP_ERROR(get_logger(), "Failed to open video: %s", video_path_.c_str());
            return;
        }

        double file_fps = cap.get(cv::CAP_PROP_FPS);
        if (file_fps <= 0.0) {
            file_fps = target_fps_;
        }

        double interval_ms = 1000.0 / target_fps_;
        uint64_t frame_id = 0;

        RCLCPP_INFO(
            get_logger(), "Video opened: %.1f fps, target %.1f fps, interval %.1f ms", file_fps,
            target_fps_, interval_ms);

        while (!stop_flag_.load(std::memory_order_relaxed)) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                if (loop_) {
                    RCLCPP_INFO(get_logger(), "Video ended, looping");
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    continue;
                } else {
                    RCLCPP_INFO(get_logger(), "Video ended, stopping");
                    break;
                }
            }

            if (frame.empty()) {
                continue;
            }

            output_buffer_ = frame.clone();
            *frame_output_ = VisionImageRef{&output_buffer_, frame_id};

            ++frame_id;

            auto sleep_duration = std::chrono::milliseconds(static_cast<int>(interval_ms));
            std::this_thread::sleep_for(sleep_duration);
        }

        cap.release();
        RCLCPP_INFO(get_logger(), "VideoReader thread stopped");
    }

    std::string video_path_;
    double target_fps_ = 30.0;
    bool loop_ = true;
    std::string output_name_ = "/video_reader/latest_frame";

    cv::Mat output_buffer_;
    OutputInterface<VisionImageRef> frame_output_;

    std::thread reader_thread_;
    std::atomic<bool> stop_flag_{false};
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::VideoReader, rmcs_executor::Component)
