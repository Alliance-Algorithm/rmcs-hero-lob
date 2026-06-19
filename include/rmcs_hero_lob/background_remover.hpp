#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class BackgroundRemover {
public:
    explicit BackgroundRemover(const MotionForegroundConfig& config) : config_(config) {}

    ForegroundResult process(const FrameReference& reference, const cv::Mat& frame) {
        ForegroundResult result;
        if (!reference.valid || frame.empty()) {
            return result;
        }
        cv::Mat ref_gray, cur_gray;
        cv::cvtColor(reference.bgr, ref_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, cur_gray, cv::COLOR_BGR2GRAY);
        cv::Mat diff;
        cv::absdiff(ref_gray, cur_gray, diff);
        cv::Mat bright_mask;
        cv::threshold(cur_gray, bright_mask, config_.min_brightness_value, 255,
                      cv::THRESH_BINARY);
        cv::Mat diff_mask;
        cv::threshold(diff, diff_mask, config_.min_diff_value, 255, cv::THRESH_BINARY);
        cv::Mat candidate_mask;
        cv::bitwise_and(diff_mask, bright_mask, candidate_mask);
        if (config_.open_kernel_size > 0) {
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(config_.open_kernel_size, config_.open_kernel_size));
            cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_OPEN, kernel);
        }
        if (config_.close_kernel_size > 0) {
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(config_.close_kernel_size, config_.close_kernel_size));
            cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_CLOSE, kernel);
        }
        result.valid = true;
        result.candidate_mask = candidate_mask;
        result.candidate_bgr = frame;
        return result;
    }

private:
    MotionForegroundConfig config_;
};

} // namespace rmcs_hero_lob
