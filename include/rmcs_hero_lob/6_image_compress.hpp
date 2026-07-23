#pragma once

#include "rmcs_hero_lob/configs.hpp"

#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

class Compression {
public:
    explicit Compression(const PipelineConfig& config)
        : config_(config) {}

    CompressionResult Process(const SynthesisResult& synthesis) const {
        CompressionResult result;
        if (!synthesis.valid || synthesis.output_image.empty()) {
            return result;
        }

        const auto& cfg = config_.compression;
        if (cfg.exposure_output_width > 0 && cfg.exposure_output_height > 0) {
            cv::resize(
                synthesis.output_image, result.output_image,
                cv::Size(cfg.exposure_output_width, cfg.exposure_output_height), 0, 0,
                cv::INTER_AREA);
        } else {
            result.output_image = synthesis.output_image;
        }

        result.valid = true;
        return result;
    }

private:
    PipelineConfig config_;
};
} // namespace rmcs_hero_lob