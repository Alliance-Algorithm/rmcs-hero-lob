#pragma once

#include <cstdint>
#include <opencv2/core/mat.hpp>
namespace rmcs_hero_lob::msgs {

struct CameraFrame {
    cv::Mat image;
    uint64_t frame_id;
};
} // namespace rmcs_hero_lob::msgs
