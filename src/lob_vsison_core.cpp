#include "rmcs_executor/component.hpp"
#include "rmcs_hero_lob/configs.hpp"
#include "rmcs_hero_lob/pipeline.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>

namespace rmcs_hero_lob {

class LobVisionCore
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    LobVisionCore()
        : Node{get_component_name(), rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        register_input("/hero_lob/camera_frame", camera_frame_);
        register_input("/gimbal/bullet_fired", bullet_fired_, false);

        register_output("/hero_lob/image/shooting_background", latest_shooting_background_, cv::Mat());
        register_output("/hero_lob/image/shooting_track", latest_shooting_track_, cv::Mat());
        register_output("/hero_lob/image/shooting_exposure_image", latest_shooting_exposure_image_, cv::Mat());

        config_.process_start_frame = static_cast<int>(get_parameter("process_start_frame").as_int());
        config_.process_end_frame = static_cast<int>(get_parameter("process_end_frame").as_int());
        config_.history_queue_max_size = static_cast<int>(get_parameter("history_queue_max_size").as_int());
        config_.history_sample_interval = static_cast<int>(get_parameter("history_sample_interval").as_int());
    }

    ~LobVisionCore() override {
        stop_flag_.store(true, std::memory_order_relaxed);
        frame_cv_.notify_all();
        if (worker_thread_.joinable())
            worker_thread_.join();
    }

    void before_updating() override {
        stop_flag_.store(false, std::memory_order_relaxed);
        worker_thread_ = std::thread(&LobVisionCore::worker_func, this);
        RCLCPP_INFO(get_logger(), "LobVisionCore worker thread started");
    }

    void update() override {
        if (!camera_frame_->image.empty() && camera_frame_->frame_id != last_frame_id_) {
            last_frame_id_ = camera_frame_->frame_id;
            std::lock_guard<std::mutex> lock(frame_mutex_);
            frame_queue_.push_back(*camera_frame_);
            frame_cv_.notify_one();
        }

        if (!last_bullet_status_ && *bullet_fired_) {
            trigger_pending_.store(true, std::memory_order_relaxed);
            trigger_time_ = std::chrono::steady_clock::now();
            RCLCPP_INFO(get_logger(), "bullet_fired trigger detected");
        }
        last_bullet_status_ = *bullet_fired_;

        {
            std::lock_guard<std::mutex> lock(output_mutex_);
            if (!output_background_.empty())
                *latest_shooting_background_ = output_background_;
            if (!output_track_.empty())
                *latest_shooting_track_ = output_track_;
            if (!output_exposure_.empty())
                *latest_shooting_exposure_image_ = output_exposure_;
        }
    }

private:
    enum class State { IDLE, WAITING, PROCESSING, SYNTHESIZING, DONE };

    void worker_func() {
        State state = State::IDLE;

        std::vector<cv::Mat> history_queue;
        int history_sample_counter = 0;
        int frame_counter = 0;

        while (!stop_flag_.load(std::memory_order_relaxed)) {
            CameraFrame frame_data;
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait(
                    lock, [this] { return !frame_queue_.empty() || stop_flag_.load(std::memory_order_relaxed); });

                if (stop_flag_.load(std::memory_order_relaxed))
                    break;

                frame_data = std::move(frame_queue_.front());
                frame_queue_.pop_front();
            }

            switch (state) {
            case State::IDLE: {
                ++history_sample_counter;
                if (history_sample_counter % config_.history_sample_interval == 0) {
                    history_queue.push_back(frame_data.image.clone());
                    if (static_cast<int>(history_queue.size()) > config_.history_queue_max_size)
                        history_queue.erase(history_queue.begin());
                }

                if (trigger_pending_.exchange(false, std::memory_order_relaxed)) {
                    if (!history_queue.empty()) {
                        pipeline_.SetReferenceFromHistory(history_queue, static_cast<int64_t>(frame_data.frame_id));
                        pipeline_.ResetTracker();
                        auto bg_elapsed =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - trigger_time_).count();
                        RCLCPP_INFO(
                            get_logger(),
                            "background ready (%.3fs from trigger), %zu history frames, transition to WAITING",
                            bg_elapsed, history_queue.size());
                    }
                    frame_counter = 0;
                    state = State::WAITING;
                }
                break;
            }

            case State::WAITING: {
                ++frame_counter;
                if (frame_counter >= config_.process_start_frame) {
                    state = State::PROCESSING;
                    RCLCPP_INFO(get_logger(), "WAITING -> PROCESSING (frame %d)", frame_counter);
                }
                break;
            }

            case State::PROCESSING: {
                ++frame_counter;

                FrameData f;
                f.frame_index = static_cast<int64_t>(frame_data.frame_id);
                f.timestamp_seconds = static_cast<double>(frame_data.frame_id) / 120.0;
                f.bgr = frame_data.image;
                cv::cvtColor(frame_data.image, f.hsv, cv::COLOR_BGR2HSV);

                TrajectoryResult traj = pipeline_.ProcessFrame(f);

                if (traj.valid) {
                    cv::Mat track_display;
                    traj.trajectory_layer.convertTo(track_display, CV_8UC3, 1.0);
                    std::lock_guard<std::mutex> lock(output_mutex_);
                    output_track_ = track_display;
                }

                if (frame_counter >= config_.process_end_frame) {
                    state = State::SYNTHESIZING;
                    RCLCPP_INFO(get_logger(), "PROCESSING -> SYNTHESIZING (frame %d)", frame_counter);
                }
                break;
            }

            case State::SYNTHESIZING: {
                ReferenceFrameResult ref = pipeline_.GetReferenceFrameSelector().GetReference();
                if (ref.has_reference) {
                    std::lock_guard<std::mutex> lock(output_mutex_);
                    output_background_ = ref.reference_frame.bgr;
                }

                CompressionResult result = pipeline_.Finalize();
                if (result.valid && !result.output_image.empty()) {
                    std::lock_guard<std::mutex> lock(output_mutex_);
                    output_exposure_ = result.output_image;
                }

                auto total_elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - trigger_time_).count();
                RCLCPP_INFO(
                    get_logger(), "SYNTHESIZING -> DONE, output ready (%.3fs from trigger, %dx%d)", total_elapsed,
                    result.output_image.cols, result.output_image.rows);
                state = State::DONE;
                break;
            }

            case State::DONE:
                history_sample_counter = 0;
                history_queue.clear();
                state = State::IDLE;
                RCLCPP_INFO(get_logger(), "DONE -> IDLE, ready for next trigger");
                break;
            }
        }

        RCLCPP_INFO(get_logger(), "worker thread stopped");
    }

    InputInterface<bool> bullet_fired_;
    InputInterface<CameraFrame> camera_frame_;

    OutputInterface<cv::Mat> latest_shooting_background_;
    OutputInterface<cv::Mat> latest_shooting_track_;
    OutputInterface<cv::Mat> latest_shooting_exposure_image_;

    PipelineConfig config_;
    Pipeline pipeline_{config_};

    std::thread worker_thread_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> trigger_pending_{false};
    bool last_bullet_status_ = false;
    uint64_t last_frame_id_ = 0;
    std::chrono::steady_clock::time_point trigger_time_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::deque<CameraFrame> frame_queue_;

    std::mutex output_mutex_;
    cv::Mat output_background_;
    cv::Mat output_track_;
    cv::Mat output_exposure_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::LobVisionCore, rmcs_executor::Component)
