#pragma once

#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class ImageSynthesis {
public:
    explicit ImageSynthesis(const TrajectoryWindowConfig& config) : config_(config) {}

    SynthesisResult process(const FrameReference& reference, const TrajectoryResult& trajectory) const {
        SynthesisResult result;
        if (!reference.valid || !trajectory.valid || trajectory.trajectory_layer.empty()) {
            if (reference.valid) {
                result.valid = true;
                result.output_image = reference.bgr.clone();
            }
            return result;
        }
        cv::Mat layer = trajectory.trajectory_layer;
        cv::Mat flat = layer.reshape(1, 1);
        std::vector<float> values;
        values.reserve(flat.total());
        const float* ptr = flat.ptr<float>(0);
        for (size_t i = 0; i < flat.total(); ++i) {
            if (ptr[i] > 1e-6F) {
                values.push_back(ptr[i]);
            }
        }
        std::sort(values.begin(), values.end());
        int nonzero_count = static_cast<int>(values.size());
        int p99_index = static_cast<int>(nonzero_count * config_.normalization_percentile);
        if (p99_index >= nonzero_count) {
            p99_index = nonzero_count - 1;
        }
        float max_val = (nonzero_count > 0) ? values[static_cast<size_t>(p99_index)] : 0.0F;
        if (max_val < 1e-6F) {
            result.valid = true;
            result.output_image = reference.bgr.clone();
            return result;
        }
        cv::Mat normalized;
        layer.convertTo(normalized, CV_32F, 255.0 / max_val);
        cv::Mat clamped;
        normalized.convertTo(clamped, CV_8UC3);
        cv::Mat output;
        cv::add(reference.bgr, clamped, output);
        result.valid = true;
        result.output_image = output;
        return result;
    }

private:
    TrajectoryWindowConfig config_;
};

} // namespace rmcs_hero_lob
