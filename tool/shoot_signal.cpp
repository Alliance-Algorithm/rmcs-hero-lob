#include "rmcs_executor/component.hpp"
#include "std_msgs/msg/int32.hpp"
#include <opencv2/core/mat.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
namespace rmcs_hero_lob {
class ShootSignal
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    ShootSignal()
        : Node{get_component_name(), rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        register_input("/gimbal/bullet_fired", bullet_fired_, false);

        publisher_ = create_publisher<std_msgs::msg::Int32>("/hero_lob/record_control", rclcpp::QoS(10));
    }

    void update() override {

        if (!last_status_ && *bullet_fired_) {
            std_msgs::msg::Int32 msg;
            msg.set__data(1);
            publisher_->publish(msg);
            RCLCPP_INFO(get_logger(), "signal publish");
        }
    }

private:
    InputInterface<bool> bullet_fired_;
    bool last_status_ = false;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
};
} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::ShootSignal, rmcs_executor::Component)