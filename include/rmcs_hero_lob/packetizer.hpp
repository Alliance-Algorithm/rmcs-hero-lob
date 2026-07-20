#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rmcs_hero_lob {

constexpr std::size_t kPacketSize = 300;
constexpr std::size_t kHeaderSize = 4;
constexpr std::size_t kPayloadSize = kPacketSize - kHeaderSize;
constexpr std::size_t kMaxPacketCount = 255;
constexpr std::uint8_t kMessageTypeImage = 0x01;
constexpr std::uint8_t kStatusStart = 0x01;
constexpr std::uint8_t kStatusEnd = 0x02;
constexpr std::uint8_t kStatusFec = 0x03;
constexpr std::uint8_t kStatusMask = 0x0F;
constexpr std::uint8_t kKPrimeShift = 4;
constexpr std::uint8_t kFecDataPerGroup = 10;
constexpr std::uint8_t kFecFecPerGroup = 3;

using PayloadBuffer = std::array<std::uint8_t, kPayloadSize>;

std::uint8_t make_fec_status(std::uint8_t k_prime, bool is_last);

std::vector<std::array<std::uint8_t, kPacketSize>> packetize_jpeg(
    std::span<const std::uint8_t> jpeg_bytes, std::uint8_t image_sequence,
    std::uint8_t message_type = kMessageTypeImage);

std::vector<PayloadBuffer>
    rs_encode(const std::vector<PayloadBuffer>& data_payloads, std::uint8_t k_data, std::uint8_t r_fec);

} // namespace rmcs_hero_lob
