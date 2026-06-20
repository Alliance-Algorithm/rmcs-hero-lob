#include <opencv2/core.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <rmcs_executor/component.hpp>
#include <string>

namespace rmcs_hero_lob {

class Broadcast
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    Broadcast()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        register_input("/hero/lob/latest_frame", latest_frame_);
        frame_id_ = get_parameter("frame_id").as_string();

        publisher_ = create_publisher<sensor_msgs::msg::Image>(
            get_parameter("topic_name").as_string(), rclcpp::QoS{5}.reliable());
    }

    void update() override {
        if (!latest_frame_.ready() || latest_frame_->empty()) {
            return;
        }

        const auto& mat = *latest_frame_;
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header{}, encoding_for(mat), mat).toImageMsg();
        msg->header.stamp = now();
        msg->header.frame_id = frame_id_;
        publisher_->publish(std::move(*msg));
    }

private:
    static std::string encoding_for(const cv::Mat& mat) {
        switch (mat.type()) {
        case CV_8UC1: return "mono8";
        case CV_8UC3: return "bgr8";
        case CV_8UC4: return "bgra8";
        default: return "passthrough";
        }
    }

    std::string frame_id_;
    InputInterface<cv::Mat> latest_frame_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

} // namespace rmcs_hero_lob

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::Broadcast, rmcs_executor::Component)
