#include <rclcpp/logging.hpp>
#include <rmcs_executor/component.hpp>

namespace rmcs_hero_lob {

class Test : public rmcs_executor::Component {
public:
    Test() = default;

    void update() override {
        static bool logged = false;
        if (!logged) {
            RCLCPP_INFO(rclcpp::get_logger("rmcs_hero_lob"), "rmcs_hero_lob::Test loaded");
            logged = true;
        }
    }
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::Test, rmcs_executor::Component)
