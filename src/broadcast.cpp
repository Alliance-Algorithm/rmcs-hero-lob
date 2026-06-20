#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <rmcs_executor/component.hpp>
#include <stdexcept>
#include <string>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class RawFrameBroadcast
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    RawFrameBroadcast()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        frame_id_ = get_parameter("frame_id").as_string();
        register_input(get_parameter("input_name").as_string(), latest_frame_);

        publish_interval_ = std::chrono::duration<double>(
            1.0 / get_parameter("publish_rate").as_double());
        jpeg_quality_ = get_parameter("jpeg_quality").as_int();

        publisher_ = create_publisher<sensor_msgs::msg::CompressedImage>(
            get_parameter("topic_name").as_string() + "/compressed", rclcpp::QoS{5}.best_effort());
    }

    void update() override {
        auto now = std::chrono::steady_clock::now();
        if (now - last_publish_time_ < publish_interval_) {
            return;
        }
        if (!latest_frame_.ready() || latest_frame_->empty()) {
            return;
        }
        last_publish_time_ = now;
        publish_mat(*latest_frame_);
    }

private:
    void publish_mat(const cv::Mat& mat) {
        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = now();
        msg.header.frame_id = frame_id_;
        msg.format = "jpeg;qimage2;cv_bridge";

        std::vector<uchar> buf;
        cv::imencode(".jpg", mat, buf, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});
        msg.data.assign(buf.begin(), buf.end());

        publisher_->publish(std::move(msg));
    }

    std::string frame_id_;
    InputInterface<cv::Mat> latest_frame_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;

    std::chrono::steady_clock::time_point last_publish_time_{std::chrono::steady_clock::now() - std::chrono::hours(1)};
    std::chrono::duration<double> publish_interval_{1.0 / 30.0};
    int jpeg_quality_{80};
};

class ProcessedImageBroadcast
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    ProcessedImageBroadcast()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        frame_id_ = get_parameter("frame_id").as_string();
        register_input(get_parameter("input_name").as_string(), processed_image_);

        publish_interval_ = std::chrono::duration<double>(
            1.0 / get_parameter("publish_rate").as_double());
        jpeg_quality_ = get_parameter("jpeg_quality").as_int();

        publisher_ = create_publisher<sensor_msgs::msg::CompressedImage>(
            get_parameter("topic_name").as_string() + "/compressed", rclcpp::QoS{5}.best_effort());
    }

    void update() override {
        auto now = std::chrono::steady_clock::now();
        if (now - last_publish_time_ < publish_interval_) {
            return;
        }
        if (!processed_image_.ready() || processed_image_->image == nullptr
            || processed_image_->image->empty()) {
            return;
        }
        if (processed_image_->frame_id == last_published_frame_id_) {
            return;
        }
        last_publish_time_ = now;
        publish_mat(*processed_image_->image);
        last_published_frame_id_ = processed_image_->frame_id;
    }

private:
    void publish_mat(const cv::Mat& mat) {
        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = now();
        msg.header.frame_id = frame_id_;
        msg.format = "jpeg;qimage2;cv_bridge";

        std::vector<uchar> buf;
        cv::imencode(".jpg", mat, buf, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});
        msg.data.assign(buf.begin(), buf.end());

        publisher_->publish(std::move(msg));
    }

    std::string frame_id_;
    InputInterface<VisionImageRef> processed_image_;
    uint64_t last_published_frame_id_{0};
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;

    std::chrono::steady_clock::time_point last_publish_time_{std::chrono::steady_clock::now() - std::chrono::hours(1)};
    std::chrono::duration<double> publish_interval_{1.0 / 30.0};
    int jpeg_quality_{80};
};

} // namespace rmcs_hero_lob

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::RawFrameBroadcast, rmcs_executor::Component)
PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::ProcessedImageBroadcast, rmcs_executor::Component)
