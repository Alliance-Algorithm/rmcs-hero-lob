#include "rmcs_executor/component.hpp"
#include "rmcs_hero_lob/configs.hpp"
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

        register_output("/hero_lob/test", camera_image_test_);
    }

    void update() override {
        if (camera_frame_->image.empty()) {
            RCLCPP_INFO(get_logger(), "frame_empty");
        }

        RCLCPP_INFO(get_logger(), "frame id: %lu", camera_frame_->frame_id);

        *camera_image_test_ = camera_frame_->image;
    }

private:
    InputInterface<CameraFrame> camera_frame_;

    OutputInterface<cv::Mat> camera_image_test_;
};
} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::LobVisionCore, rmcs_executor::Component)