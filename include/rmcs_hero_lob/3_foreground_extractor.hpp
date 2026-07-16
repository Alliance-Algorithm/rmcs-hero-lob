#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rmcs_hero_lob/configs.hpp"

namespace rmcs_hero_lob {

class BackgroundRemover {
public:
    explicit BackgroundRemover(const PipelineConfig& config)
        : config_(config) {}

    ForegroundMaskResult Process(const ReferenceFrameResult& reference, const RegistrationResult& registration) {
        ForegroundMaskResult result;
        if (!reference.has_reference || registration.registered_bgr.empty()) {
            return result;
        }
        const auto& fg = config_.motion_foreground;
        cv::Mat ref_gray, cur_gray;
        cv::cvtColor(reference.reference_frame.bgr, ref_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(registration.registered_bgr, cur_gray, cv::COLOR_BGR2GRAY);
        if (ref_gray.size() != cur_gray.size()) {
            cv::resize(ref_gray, ref_gray, cur_gray.size(), 0, 0, cv::INTER_AREA);
        }
        cv::Mat diff;
        cv::absdiff(ref_gray, cur_gray, diff);
        cv::Mat bright_mask;
        cv::threshold(cur_gray, bright_mask, fg.min_brightness_value, 255, cv::THRESH_BINARY);
        cv::Mat diff_mask;
        cv::threshold(diff, diff_mask, fg.min_diff_value, 255, cv::THRESH_BINARY);
        cv::Mat candidate_mask;
        cv::bitwise_and(diff_mask, bright_mask, candidate_mask);
        if (fg.open_kernel_size > 0) {
            cv::Mat open_kernel =
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(fg.open_kernel_size, fg.open_kernel_size));
            cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_OPEN, open_kernel);
        }
        if (fg.close_kernel_size > 0) {
            cv::Mat close_kernel =
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(fg.close_kernel_size, fg.close_kernel_size));
            cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_CLOSE, close_kernel);
        }
        result.valid = true;
        result.frame_index = registration.frame_index;
        result.timestamp_seconds = registration.timestamp_seconds;
        result.roi = cv::Rect(0, 0, candidate_mask.cols, candidate_mask.rows);
        result.candidate_mask = candidate_mask;
        result.candidate_bgr = registration.registered_bgr;
        return result;
    }

    void Reset() {}

private:
    PipelineConfig config_;
};

} // namespace rmcs_hero_lob
