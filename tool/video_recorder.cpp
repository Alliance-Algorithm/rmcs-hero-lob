#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <hikcamera/capturer.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>
#include <std_msgs/msg/int32.hpp>

#include <rmcs_executor/component.hpp>

namespace rmcs_hero_lob {

class VideoRecorder
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    VideoRecorder()
        : Node{get_component_name(), rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        config_.exposure_us = static_cast<float>(get_parameter("exposure_us").as_double());
        config_.framerate = static_cast<float>(get_parameter("framerate").as_double());
        config_.invert_image = get_parameter("invert_image").as_bool();

        record_topic_ = get_parameter("record_topic").as_string();
        output_dir_ = get_parameter("output_dir").as_string();

        double pre_seconds = get_parameter("pre_seconds").as_double();
        double post_seconds = get_parameter("post_seconds").as_double();
        pre_count_ = static_cast<int>(std::lround(config_.framerate * pre_seconds));
        post_count_ = static_cast<int>(std::lround(config_.framerate * post_seconds));
        if (pre_count_ < 0)
            pre_count_ = 0;
        if (post_count_ < 0)
            post_count_ = 0;

        std::string fourcc = get_parameter("fourcc").as_string();
        if (fourcc.size() == 4)
            fourcc_ = cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
        else
            fourcc_ = 0;

        camera_.configure(config_);

        subscription_ = create_subscription<std_msgs::msg::Int32>(
            record_topic_, rclcpp::QoS(10),
            [this](const std_msgs::msg::Int32::SharedPtr) { trigger_.store(true, std::memory_order_relaxed); });
    }

    ~VideoRecorder() override {
        stop_flag_.store(true, std::memory_order_relaxed);
        queue_cv_.notify_all();
        if (capture_thread_.joinable())
            capture_thread_.join();
        if (writer_thread_.joinable())
            writer_thread_.join();
    }

    void before_updating() override {
        stop_flag_.store(false, std::memory_order_relaxed);
        writer_thread_ = std::thread(&VideoRecorder::writer_thread_func, this);
        capture_thread_ = std::thread(&VideoRecorder::capture_thread_func, this);
        RCLCPP_INFO(
            get_logger(), "Video recorder started (target ~%.0f Hz, control topic '%s', pre %d frames, post %d frames)",
            config_.framerate, record_topic_.c_str(), pre_count_, post_count_);
    }

    void update() override {}

private:
    enum class ItemKind { Open, Frame, Close };

    struct Item {
        ItemKind kind;
        cv::Mat frame;
        int index = 0;
        std::string filename;
        cv::Size size;
        bool is_color = true;
    };

    void enqueue(Item&& item) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push_back(std::move(item));
        }
        queue_cv_.notify_one();
    }

    void capture_thread_func() {
        auto result = camera_.connect();
        if (!result) {
            RCLCPP_ERROR(get_logger(), "Failed to connect camera: %s", result.error().c_str());
            return;
        }

        RCLCPP_INFO(get_logger(), "Camera connected");

        using clock = std::chrono::steady_clock;
        auto fps_last_time = clock::now();
        uint64_t fps_frame_count = 0;

        const std::size_t buffer_capacity = static_cast<std::size_t>(pre_count_) + 1;

        bool recording = false;
        int post_index = 0;
        int post_remaining = 0;

        while (!stop_flag_.load(std::memory_order_relaxed)) {
            auto image = camera_.read_image();

            if (!image) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 1000, "Failed to read image: %s", image.error().c_str());
                continue;
            }

            if (!recording && !writer_busy_.load(std::memory_order_relaxed)
                && trigger_.exchange(false, std::memory_order_relaxed)) {

                writer_busy_.store(true, std::memory_order_relaxed);
                recording = true;

                Item open{};
                open.kind = ItemKind::Open;
                open.filename = make_output_path();
                open.size = image->size();
                open.is_color = image->channels() == 3;
                enqueue(std::move(open));

                const int k = static_cast<int>(ring_buffer_.size());
                for (int i = 0; i < k; ++i) {
                    Item f{};
                    f.kind = ItemKind::Frame;
                    f.frame = ring_buffer_[static_cast<std::size_t>(i)];
                    f.index = i - (k - 1);
                    enqueue(std::move(f));
                }

                post_index = 1;
                post_remaining = post_count_;

                RCLCPP_INFO(
                    get_logger(), "Recording triggered: %dx%d @ %.0f Hz, %d pre frames", image->cols, image->rows,
                    config_.framerate, k);
            }

            if (recording) {
                Item f{};
                f.kind = ItemKind::Frame;
                f.frame = image->clone();
                f.index = post_index++;
                enqueue(std::move(f));

                if (--post_remaining <= 0) {
                    Item close{};
                    close.kind = ItemKind::Close;
                    enqueue(std::move(close));
                    recording = false;
                }
            } else {
                ring_buffer_.push_back(image->clone());
                while (ring_buffer_.size() > buffer_capacity)
                    ring_buffer_.pop_front();
            }

            ++fps_frame_count;

            auto now = clock::now();
            auto elapsed = std::chrono::duration<double>(now - fps_last_time).count();
            if (elapsed >= 1.0) {
                double actual_fps = static_cast<double>(fps_frame_count) / elapsed;
                RCLCPP_INFO(
                    get_logger(), "Capture FPS: %.1f Hz%s", actual_fps,
                    writer_busy_.load(std::memory_order_relaxed) ? " (recording)" : "");
                fps_last_time = now;
                fps_frame_count = 0;
            }
        }

        auto disconnect_result = camera_.disconnect();
        if (!disconnect_result) {
            RCLCPP_WARN(get_logger(), "Failed to disconnect camera: %s", disconnect_result.error().c_str());
        }

        RCLCPP_INFO(get_logger(), "Capture thread stopped");
    }

    void writer_thread_func() {
        cv::VideoWriter writer;
        uint64_t recorded_frames = 0;

        while (true) {
            Item item;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_.load(std::memory_order_relaxed); });

                if (queue_.empty()) {
                    if (stop_flag_.load(std::memory_order_relaxed))
                        break;
                    continue;
                }

                item = std::move(queue_.front());
                queue_.pop_front();
            }

            switch (item.kind) {
            case ItemKind::Open:
                writer.open(item.filename, cv::CAP_FFMPEG, fourcc_, config_.framerate, item.size, item.is_color);
                if (!writer.isOpened()) {
                    RCLCPP_ERROR(get_logger(), "Failed to open video writer: %s", item.filename.c_str());
                } else {
                    recorded_frames = 0;
                    RCLCPP_INFO(get_logger(), "Recording file opened: %s", item.filename.c_str());
                }
                break;

            case ItemKind::Frame:
                if (writer.isOpened()) {
                    stamp_frame(item.frame, item.index);
                    writer.write(item.frame);
                    ++recorded_frames;
                }
                break;

            case ItemKind::Close:
                if (writer.isOpened()) {
                    writer.release();
                    RCLCPP_INFO(get_logger(), "Recording finished (%" PRIu64 " frames written)", recorded_frames);
                }
                writer_busy_.store(false, std::memory_order_relaxed);
                break;
            }
        }

        if (writer.isOpened())
            writer.release();

        RCLCPP_INFO(get_logger(), "Writer thread stopped");
    }

    static void stamp_frame(cv::Mat& frame, int index) {
        std::string text = std::to_string(index);
        constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
        constexpr double scale = 1.0;
        constexpr int thickness = 2;
        cv::putText(frame, text, cv::Point(10, 40), font, scale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
        cv::putText(frame, text, cv::Point(10, 40), font, scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
    }

    std::string make_output_path() const {
        std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "record_%Y%m%d_%H%M%S.avi", &tm);

        std::string dir = output_dir_;
        if (!dir.empty() && dir.back() != '/')
            dir.push_back('/');
        return dir + buffer;
    }

    hikcamera::Config config_;
    hikcamera::Camera camera_;

    std::string record_topic_;
    std::string output_dir_;
    int fourcc_ = 0;
    int pre_count_ = 0;
    int post_count_ = 0;

    std::thread capture_thread_;
    std::thread writer_thread_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> trigger_{false};
    std::atomic<bool> writer_busy_{false};

    std::deque<cv::Mat> ring_buffer_;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<Item> queue_;

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::VideoRecorder, rmcs_executor::Component)
