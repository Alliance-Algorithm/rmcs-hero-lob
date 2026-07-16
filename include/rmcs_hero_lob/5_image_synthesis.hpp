#pragma once

#include "rmcs_hero_lob/configs.hpp"

#include <vector>

#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

class ImageSynthesis {
public:
    explicit ImageSynthesis(const PipelineConfig& config)
        : config_(config) {}

    SynthesisResult Process(const ReferenceFrameResult& reference, const TrajectoryResult& trajectory) const {
        SynthesisResult result;
        if (!reference.has_reference || !trajectory.valid || trajectory.trajectory_layer.empty()) {
            if (reference.has_reference) {
                result.valid = true;
                result.output_image = reference.reference_frame.bgr.clone();
            }
            return result;
        }
        const auto& tw = config_.trajectory_window;
        cv::Mat layer = trajectory.trajectory_layer;

        if (!trajectory.exposure_count.empty()) {
            cv::Mat count_3ch;
            cv::merge(
                std::vector<cv::Mat>{trajectory.exposure_count, trajectory.exposure_count, trajectory.exposure_count},
                count_3ch);
            cv::Mat count_f;
            count_3ch.convertTo(count_f, CV_32F);
            cv::Mat safe_count;
            cv::max(count_f, 1.0F, safe_count);
            cv::divide(layer, safe_count, layer);
        }

        std::vector<float> flat;
        layer.reshape(1, 1).copyTo(flat);
        std::sort(flat.begin(), flat.end());
        auto nonzero_it = std::lower_bound(flat.begin(), flat.end(), 1e-6F);
        int nonzero_count = static_cast<int>(std::distance(nonzero_it, flat.end()));
        int p99_index = static_cast<int>(nonzero_count * tw.normalization_percentile);
        if (p99_index >= nonzero_count) {
            p99_index = nonzero_count - 1;
        }
        float max_val = (nonzero_count > 0) ? *(nonzero_it + p99_index) : 0.0F;
        if (max_val < 1e-6F) {
            result.valid = true;
            result.output_image = reference.reference_frame.bgr.clone();
            return result;
        }

        cv::Mat normalized;
        layer.convertTo(normalized, CV_8UC3, 255.0 / max_val * 0.6);

        cv::Mat lab;
        cv::cvtColor(normalized, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        cv::merge(lab_channels, lab);
        cv::Mat enhanced;
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);

        cv::Mat ref = reference.reference_frame.bgr;
        if (ref.size() != enhanced.size()) {
            cv::resize(ref, ref, enhanced.size(), 0, 0, cv::INTER_AREA);
        }
        cv::Mat output;
        cv::add(ref, enhanced, output);
        result.valid = true;
        result.output_image = output;
        return result;
    }

private:
    PipelineConfig config_;
};

} // namespace rmcs_hero_lob
