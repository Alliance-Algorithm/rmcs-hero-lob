#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <rmcs_executor/component.hpp>

namespace rmcs_hero_lob {

class ImageBroadcast
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    ImageBroadcast()
        : Node{get_component_name(), rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        std::string interface_name = get_parameter("interface_name").as_string();
        double publish_rate = get_parameter("publish_rate").as_double();

        encoding_ = get_parameter("type").as_string();
        frame_id_ = get_parameter("frame_id").as_string();

        register_input(interface_name, image_);

        publisher_ = create_publisher<sensor_msgs::msg::Image>(interface_name, rclcpp::SensorDataQoS());
        publish_period_ = std::chrono::duration<double>(1.0 / publish_rate);
    }

    ~ImageBroadcast() override {
        stop_flag_.store(true, std::memory_order_relaxed);
        if (publish_thread_.joinable())
            publish_thread_.join();
    }

    void before_updating() override {
        stop_flag_.store(false, std::memory_order_relaxed);
        publish_thread_ = std::thread(&ImageBroadcast::publish_loop, this);
    }

    void update() override {
        if (image_->empty())
            return;

        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_ = *image_;
        has_image_ = true;
    }

private:
    void publish_loop() {
        using clock = std::chrono::steady_clock;
        auto next_publish_time = clock::now();

        while (!stop_flag_.load(std::memory_order_relaxed)) {
            cv::Mat image;
            {
                std::lock_guard<std::mutex> lock(image_mutex_);
                if (has_image_)
                    image = latest_image_;
            }

            if (!image.empty()) {
                cv_bridge::CvImage image_msg;
                image_msg.header.stamp = get_clock()->now();
                image_msg.header.frame_id = frame_id_;
                image_msg.encoding = encoding_;
                image_msg.image = image;

                publisher_->publish(*image_msg.toImageMsg());
            }

            next_publish_time += std::chrono::duration_cast<clock::duration>(publish_period_);
            std::this_thread::sleep_until(next_publish_time);
        }
    }

    std::string encoding_;
    std::string frame_id_;

    std::chrono::duration<double> publish_period_{};
    std::thread publish_thread_;
    std::atomic<bool> stop_flag_{false};
    std::mutex image_mutex_;
    cv::Mat latest_image_;
    bool has_image_ = false;

    InputInterface<cv::Mat> image_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::ImageBroadcast, rmcs_executor::Component)
