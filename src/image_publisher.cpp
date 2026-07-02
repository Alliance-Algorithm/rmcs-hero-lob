#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/timer.hpp>
#include <rmcs_executor/component.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/detail/header__struct.hpp>
#include <string>
#include <vector>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class ImagePublisher
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    ImagePublisher()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
        , logger_(get_logger()) {

        if (has_parameter("input_type"))
            input_type_ = get_parameter("input_type").as_string();

        jpeg_quality_ = get_parameter("jpeg_quality").as_int();

        if (has_parameter("max_width"))
            max_width_ = get_parameter("max_width").as_int();
        if (has_parameter("max_height"))
            max_height_ = get_parameter("max_height").as_int();
        if (has_parameter("show_frame_id"))
            show_frame_id_ = get_parameter("show_frame_id").as_bool();

        image_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            get_parameter("topic_name").as_string() + "/compressed", rclcpp::QoS{10}.best_effort());

        publish_freq_ = get_parameter("publish_freq").as_double();
        publish_freq_ = MAX(0, MIN(publish_freq_, 1000));

        if (input_type_ == "mat") {
            register_input(get_parameter("Interface_name").as_string(), input_mat_);
        } else {
            register_input(get_parameter("Interface_name").as_string(), input_ref_);
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000 / publish_freq_)), [this]() {
                const cv::Mat* mat_ptr = nullptr;
                uint64_t frame_id = 0;
                if (input_type_ == "mat") {
                    if (!input_mat_.ready() || input_mat_->empty()) {
                        RCLCPP_WARN_THROTTLE(
                            logger_, *get_clock(), 2000,
                            "ImagePublisher [%s]: empty mat received, skipping publish",
                            get_name());
                        return;
                    }
                    mat_ptr = &(*input_mat_);
                    frame_id = frame_counter_++;
                } else {
                    if (!input_ref_.ready() || input_ref_->image == nullptr
                        || input_ref_->image->empty()) {
                        RCLCPP_WARN_THROTTLE(
                            logger_, *get_clock(), 2000,
                            "ImagePublisher [%s]: empty VisionImageRef received, skipping publish",
                            get_name());
                        return;
                    }
                    image_buffer_ = input_ref_->image->clone();
                    mat_ptr = &image_buffer_;
                    frame_id = input_ref_->frame_id;
                }

                sensor_msgs::msg::CompressedImage msg;
                msg.header.stamp = now();
                msg.header.frame_id = get_name();
                msg.format = "jpeg;qimage2;cv_bridge";

                cv::Mat publish_mat = *mat_ptr;
                if (max_width_ > 0 || max_height_ > 0) {
                    int w = publish_mat.cols;
                    int h = publish_mat.rows;
                    int target_w = max_width_ > 0 ? max_width_ : w;
                    int target_h = max_height_ > 0 ? max_height_ : h;
                    if (w > target_w || h > target_h) {
                        double scale = std::min(
                            static_cast<double>(target_w) / w,
                            static_cast<double>(target_h) / h);
                        cv::resize(
                            publish_mat, scaled_buffer_,
                            cv::Size(static_cast<int>(w * scale), static_cast<int>(h * scale)),
                            0, 0, cv::INTER_AREA);
                        publish_mat = scaled_buffer_;
                    }
                }

                if (show_frame_id_) {
                    std::string label = std::to_string(frame_id);
                    int font_face = cv::FONT_HERSHEY_SIMPLEX;
                    double font_scale = std::max(0.5, publish_mat.cols / 640.0);
                    int thickness = std::max(1, static_cast<int>(font_scale));
                    cv::putText(
                        publish_mat, label, cv::Point(8, static_cast<int>(30 * font_scale)),
                        font_face, font_scale, cv::Scalar(0, 255, 0), thickness);
                }

                std::vector<uchar> buf;
                cv::imencode(".jpg", publish_mat, buf, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});
                msg.data.assign(buf.begin(), buf.end());

                image_publisher_->publish(std::move(msg));
            });
    }

    void update() override {}

private:
    rclcpp::Logger logger_;

    std::string input_type_ = "mat";
    double publish_freq_ = 60.0;
    int jpeg_quality_ = 80;
    int max_width_ = 0;
    int max_height_ = 0;
    bool show_frame_id_ = false;
    uint64_t frame_counter_ = 0;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_publisher_;

    InputInterface<cv::Mat> input_mat_;
    InputInterface<VisionImageRef> input_ref_;
    cv::Mat image_buffer_;
    cv::Mat scaled_buffer_;
};
} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::ImagePublisher, rmcs_executor::Component)
