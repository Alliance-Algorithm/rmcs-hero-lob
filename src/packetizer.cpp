#include "rmcs_hero_lob/packetizer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rmcs_hero_lob {
namespace {

constexpr std::uint16_t g_exp_table_size = 512;
constexpr std::uint16_t g_gf_order = 255;

class Gf256 {
public:
    static std::uint8_t add(std::uint8_t a, std::uint8_t b) { return static_cast<std::uint8_t>(a ^ b); }
    static std::uint8_t sub(std::uint8_t a, std::uint8_t b) { return static_cast<std::uint8_t>(a ^ b); }

    static std::uint8_t mul(std::uint8_t a, std::uint8_t b) {
        if (a == 0 || b == 0)
            return 0;
        return exp_table[(log_table[a] + log_table[b]) % g_gf_order];
    }

    static std::uint8_t inv(std::uint8_t a) {
        if (a == 0)
            throw std::runtime_error("Inverse of zero in GF(256)");
        return exp_table[g_gf_order - log_table[a]];
    }

private:
    friend struct Gf256Init;
    static inline std::array<std::uint8_t, g_exp_table_size> exp_table{};
    static inline std::array<std::uint8_t, g_gf_order + 1> log_table{};

    static void initialize() {
        std::uint16_t value = 1;
        for (std::uint16_t i = 0; i < g_gf_order; ++i) {
            exp_table[i] = static_cast<std::uint8_t>(value);
            log_table[value] = static_cast<std::uint8_t>(i);
            value <<= 1;
            if (value >= 256)
                value ^= 0x11D;
        }
        for (std::uint16_t i = g_gf_order; i < g_exp_table_size; ++i)
            exp_table[i] = exp_table[i - g_gf_order];
    }
};

struct Gf256Init {
    Gf256Init() { Gf256::initialize(); }
};
Gf256Init g_init;

std::vector<std::uint8_t> build_cauchy_row(std::uint8_t row_idx, std::uint8_t k_data) {
    std::vector<std::uint8_t> row(k_data);
    for (std::uint8_t col = 0; col < k_data; ++col) {
        const std::uint8_t x = static_cast<std::uint8_t>(k_data + row_idx);
        const std::uint8_t y = col;
        row[col] = Gf256::inv(Gf256::add(x, y));
    }
    return row;
}

} // namespace

std::uint8_t make_fec_status(const std::uint8_t k_prime, const bool is_last) {
    return static_cast<std::uint8_t>((k_prime << kKPrimeShift) | (is_last ? kStatusEnd : kStatusFec));
}

std::vector<std::array<std::uint8_t, kPacketSize>>
    packetize_jpeg(
        const std::span<const std::uint8_t> jpeg_bytes, const std::uint8_t image_sequence,
        const std::uint8_t message_type) {

    const std::size_t packet_count = (jpeg_bytes.size() + kPayloadSize - 1) / kPayloadSize;
    if (packet_count == 0)
        return {};
    if (packet_count > kMaxPacketCount)
        throw std::runtime_error("JPEG payload exceeds packet sequence capacity");

    std::vector<std::array<std::uint8_t, kPacketSize>> packets(packet_count);
    for (std::size_t index = 0; index < packet_count; ++index) {
        auto& packet = packets[index];
        packet.fill(0);
        packet[0] = message_type;

        const bool is_first = index == 0;
        const bool is_last = index + 1 == packet_count;
        std::uint8_t status = 0x00;
        if (is_last)
            status = kStatusEnd;
        else if (is_first)
            status = kStatusStart;
        packet[1] = status;
        packet[2] = image_sequence;
        packet[3] = static_cast<std::uint8_t>(index);

        const std::size_t offset = index * kPayloadSize;
        const std::size_t copy_size = std::min(kPayloadSize, jpeg_bytes.size() - offset);
        std::copy_n(jpeg_bytes.begin() + static_cast<std::ptrdiff_t>(offset), copy_size, packet.begin() + kHeaderSize);
    }
    return packets;
}

std::vector<PayloadBuffer>
    rs_encode(const std::vector<PayloadBuffer>& data_payloads, const std::uint8_t k_data, const std::uint8_t r_fec) {

    if (data_payloads.size() != k_data)
        throw std::runtime_error("rs_encode: data_payloads.size() != k_data");

    std::vector<PayloadBuffer> fec_payloads(r_fec);
    for (auto& buf : fec_payloads)
        buf.fill(0);

    for (std::uint8_t fec_idx = 0; fec_idx < r_fec; ++fec_idx) {
        const auto coeffs = build_cauchy_row(fec_idx, k_data);
        for (std::size_t byte = 0; byte < kPayloadSize; ++byte) {
            std::uint8_t sum = 0;
            for (std::uint8_t data_idx = 0; data_idx < k_data; ++data_idx)
                sum = Gf256::add(sum, Gf256::mul(coeffs[data_idx], data_payloads[data_idx][byte]));
            fec_payloads[fec_idx][byte] = sum;
        }
    }
    return fec_payloads;
}

} // namespace rmcs_hero_lob
