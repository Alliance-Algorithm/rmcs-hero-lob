#include <gtest/gtest.h>

#include "rmcs_hero_lob/packetizer.hpp"
#include "rmcs_hero_lob/sequence_utils.hpp"

namespace rmcs_hero_lob {

TEST(PacketizerTest, PacketizeSinglePacketUsesDefaultMessageType) {
    const std::vector<std::uint8_t> jpeg_bytes{0xFF, 0xD8, 0x11, 0x22, 0xFF, 0xD9};
    const auto packets = packetize_jpeg(jpeg_bytes, 7);

    ASSERT_EQ(packets.size(), 1U);
    EXPECT_EQ(packets[0][0], kMessageTypeImage);
    EXPECT_EQ(packets[0][1], kStatusEnd);
    EXPECT_EQ(packets[0][2], 7);
    EXPECT_EQ(packets[0][3], 0);
}

TEST(PacketizerTest, PacketizeMultiPacketUsesConfiguredMessageType) {
    std::vector<std::uint8_t> jpeg_bytes(kPayloadSize + 8, 0x55);
    jpeg_bytes[0] = 0xFF;
    jpeg_bytes[1] = 0xD8;
    jpeg_bytes[kPayloadSize + 6] = 0xFF;
    jpeg_bytes[kPayloadSize + 7] = 0xD9;

    const std::uint8_t message_type = 0x23;
    const auto packets = packetize_jpeg(jpeg_bytes, 3, message_type);

    ASSERT_EQ(packets.size(), 2U);
    EXPECT_EQ(packets[0][0], message_type);
    EXPECT_EQ(packets[1][0], message_type);
    EXPECT_EQ(packets[0][1], kStatusStart);
    EXPECT_EQ(packets[1][1], kStatusEnd);
    EXPECT_EQ(packets[0][2], 3);
    EXPECT_EQ(packets[1][2], 3);
    EXPECT_EQ(packets[0][3], 0);
    EXPECT_EQ(packets[1][3], 1);
}

TEST(PacketizerTest, AdvanceSequenceUsesUnsetSentinelAndWrapsAt255) {
    EXPECT_EQ(advance_sequence(kUnsetSequence), 0);
    EXPECT_EQ(advance_sequence(0), 1);
    EXPECT_EQ(advance_sequence(254), 255);
    EXPECT_EQ(advance_sequence(255), 0);
}

} // namespace rmcs_hero_lob
