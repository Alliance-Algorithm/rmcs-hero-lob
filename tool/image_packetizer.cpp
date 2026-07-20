#include "rmcs_hero_lob/packetizer.hpp"
#include "rmcs_hero_lob/sequence_utils.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include <rmcs_executor/component.hpp>

namespace rmcs_hero_lob {

class ImagePacketizer
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    ImagePacketizer()
        : Node{get_component_name(), rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {

        std::string input_interface_name = get_parameter("interface_name").as_string();
        std::string output_interface_name = input_interface_name + "packages";
        std::string sequence_output_interface_name = input_interface_name + "package_seq";
        std::string sequence_input_interface_name = input_interface_name + "seq";
        fec_data_per_group_ = static_cast<std::uint8_t>(get_parameter("fec_data_per_group").as_int());
        fec_fec_per_group_ = static_cast<std::uint8_t>(get_parameter("fec_fec_per_group").as_int());
        jpeg_quality_ = static_cast<int>(get_parameter("jpeg_quality").as_int());
        int configured_message_type = static_cast<int>(kMessageTypeImage);
        get_parameter_or("message_type", configured_message_type, configured_message_type);
        if (configured_message_type < 0 || configured_message_type > 0xFF)
            throw std::runtime_error("Parameter 'message_type' must be in [0, 255]");
        message_type_ = static_cast<std::uint8_t>(configured_message_type);

        register_input(input_interface_name, image_);
        register_input(sequence_input_interface_name, image_seq_input_);
        register_output(output_interface_name, packets_, std::vector<std::array<std::uint8_t, kPacketSize>>{});
        register_output(sequence_output_interface_name, sequence_, kUnsetSequence);
    }

    void update() override {
        if (image_->empty())
            return;

        const int current_seq = *image_seq_input_;
        if (current_seq == last_processed_seq_)
            return;
        last_processed_seq_ = current_seq;

        std::vector<unsigned char> jpeg_bytes;
        std::vector<int> encode_params{cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
        if (!cv::imencode(".jpg", *image_, jpeg_bytes, encode_params))
            return;

        auto raw_packets =
            packetize_jpeg(
                std::span<const std::uint8_t>{jpeg_bytes.data(), jpeg_bytes.size()}, image_seq_, message_type_);

        const std::size_t group_count = (raw_packets.size() + fec_data_per_group_ - 1) / fec_data_per_group_;
        const std::size_t total_packet_count = raw_packets.size() + group_count * fec_fec_per_group_;
        if (total_packet_count > kMaxPacketCount)
            RCLCPP_WARN(
                get_logger(), "Total packets %zu exceeds protocol limit %zu, packet_sequence may overflow",
                total_packet_count, kMaxPacketCount);

        std::vector<std::array<std::uint8_t, kPacketSize>> output;
        if (!raw_packets.empty())
            build_fec_packets(raw_packets, output);

        *packets_ = std::move(output);
        *sequence_ = static_cast<int>(image_seq_);
        ++image_seq_;
    }

private:
    void build_fec_packets(
        const std::vector<std::array<std::uint8_t, kPacketSize>>& raw_packets,
        std::vector<std::array<std::uint8_t, kPacketSize>>& out_packets) {

        std::vector<PayloadBuffer> data_payloads;
        data_payloads.reserve(raw_packets.size());
        for (const auto& pkt : raw_packets) {
            PayloadBuffer buf;
            std::copy_n(pkt.begin() + kHeaderSize, kPayloadSize, buf.begin());
            data_payloads.push_back(buf);
        }

        const std::uint8_t k = fec_data_per_group_;
        const std::uint8_t r = fec_fec_per_group_;
        const std::uint8_t D = static_cast<std::uint8_t>(data_payloads.size());
        const std::uint8_t full_groups = D / k;
        std::uint8_t k_prime = D % k;
        std::uint8_t total_groups = full_groups;
        if (k_prime == 0 && full_groups > 0)
            k_prime = k;
        else if (k_prime > 0)
            ++total_groups;

        out_packets.clear();
        std::uint8_t seq = 0;

        for (std::uint8_t g = 0; g < total_groups; ++g) {
            const bool is_last_group = (g == total_groups - 1);
            const std::uint8_t kg = is_last_group ? k_prime : k;

            std::vector<PayloadBuffer> group_data(data_payloads.begin() + g * k, data_payloads.begin() + g * k + kg);
            auto fec = rs_encode(group_data, kg, r);

            for (std::uint8_t d = 0; d < kg; ++d) {
                std::array<std::uint8_t, kPacketSize> packet;
                packet.fill(0);
                packet[0] = message_type_;
                packet[1] = (seq == 0) ? kStatusStart : static_cast<std::uint8_t>(0x00);
                packet[2] = image_seq_;
                packet[3] = seq;
                std::copy_n(group_data[d].begin(), kPayloadSize, packet.begin() + kHeaderSize);
                out_packets.push_back(std::move(packet));
                ++seq;
            }

            for (std::uint8_t f = 0; f < r; ++f) {
                const bool is_last_fec = is_last_group && (f == r - 1);
                std::array<std::uint8_t, kPacketSize> packet;
                packet.fill(0);
                packet[0] = message_type_;
                packet[1] = make_fec_status(k_prime, is_last_fec);
                packet[2] = image_seq_;
                packet[3] = seq;
                std::copy_n(fec[f].begin(), kPayloadSize, packet.begin() + kHeaderSize);
                out_packets.push_back(std::move(packet));
                ++seq;
            }
        }
    }

    std::uint8_t image_seq_ = 0;
    std::uint8_t fec_data_per_group_ = kFecDataPerGroup;
    std::uint8_t fec_fec_per_group_ = kFecFecPerGroup;
    std::uint8_t message_type_ = kMessageTypeImage;
    int jpeg_quality_ = 95;

    InputInterface<cv::Mat> image_;
    InputInterface<int> image_seq_input_;
    int last_processed_seq_ = -1;

    OutputInterface<std::vector<std::array<std::uint8_t, kPacketSize>>> packets_;
    OutputInterface<int> sequence_;
};

} // namespace rmcs_hero_lob

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_hero_lob::ImagePacketizer, rmcs_executor::Component)
