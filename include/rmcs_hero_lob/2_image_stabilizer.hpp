#pragma once

#include "rmcs_hero_lob/configs.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

class ImageRegistratorOrb {
public:
    explicit ImageRegistratorOrb(const PipelineConfig& config)
        : config_(config)
        , orb_(cv::ORB::create(config.image_registrator_orb.max_features)) {}

    RegistrationResult Process(const ReferenceFrameResult& reference, const FrameData& frame) {
        RegistrationResult result;
        result.frame_index = frame.frame_index;
        result.timestamp_seconds = frame.timestamp_seconds;

        if (!reference.has_reference || frame.bgr.empty()) {
            result.valid = false;
            return result;
        }

        const float scale = config_.image_registrator_orb.downscale_factor;
        const int ref_idx = reference.reference_frame.frame_index;

        bool ref_changed = (ref_idx != cached_ref_frame_index_);

        if (ref_changed) {
            cv::Mat ref_gray;
            cv::cvtColor(reference.reference_frame.bgr, ref_gray, cv::COLOR_BGR2GRAY);

            const int ew = config_.image_registrator_orb.exclude_top_left_width;
            const int eh = config_.image_registrator_orb.exclude_top_left_height;
            if (feature_mask_.empty() || feature_mask_.rows != ref_gray.rows || feature_mask_.cols != ref_gray.cols) {
                feature_mask_ = cv::Mat::ones(ref_gray.size(), CV_8UC1) * 255;
                if (ew > 0 && eh > 0) {
                    int mw = (scale < 1.0F) ? static_cast<int>(ew * scale) : ew;
                    int mh = (scale < 1.0F) ? static_cast<int>(eh * scale) : eh;
                    feature_mask_(cv::Rect(0, 0, mw, mh)).setTo(0);
                }
            }

            if (scale < 1.0F) {
                cv::Mat ref_small;
                cv::resize(ref_gray, ref_small, cv::Size(), scale, scale, cv::INTER_AREA);
                cv::Mat mask_small;
                cv::resize(feature_mask_, mask_small, cv::Size(), scale, scale, cv::INTER_NEAREST);
                orb_->detectAndCompute(ref_small, mask_small, ref_kp_, ref_desc_);
                for (auto& kp : ref_kp_) {
                    kp.pt.x /= scale;
                    kp.pt.y /= scale;
                }
            } else {
                orb_->detectAndCompute(ref_gray, feature_mask_, ref_kp_, ref_desc_);
            }
            cached_ref_frame_index_ = ref_idx;
        }

        cv::Mat cur_gray;
        cv::cvtColor(frame.bgr, cur_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> cur_kp;
        cv::Mat cur_desc;
        if (scale < 1.0F) {
            cv::Mat cur_small;
            cv::resize(cur_gray, cur_small, cv::Size(), scale, scale, cv::INTER_AREA);
            cv::Mat mask_small;
            cv::resize(feature_mask_, mask_small, cv::Size(), scale, scale, cv::INTER_NEAREST);
            orb_->detectAndCompute(cur_small, mask_small, cur_kp, cur_desc);
            for (auto& kp : cur_kp) {
                kp.pt.x /= scale;
                kp.pt.y /= scale;
            }
        } else {
            orb_->detectAndCompute(cur_gray, feature_mask_, cur_kp, cur_desc);
        }

        if (ref_desc_.empty() || cur_desc.empty() || ref_kp_.size() < 4 || cur_kp.size() < 4) {
            result.valid = true;
            result.transform = cv::Matx23f(1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
            result.registered_bgr = frame.bgr.clone();
            cv::cvtColor(result.registered_bgr, result.registered_hsv, cv::COLOR_BGR2HSV);
            return result;
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(cur_desc, ref_desc_, knn_matches, 2);

        const float ratio = config_.image_registrator_orb.match_ratio_threshold;
        std::vector<cv::DMatch> good_matches;
        for (const auto& m : knn_matches) {
            if (m.size() == 2 && m[0].distance < ratio * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }

        if (static_cast<int>(good_matches.size()) < config_.image_registrator_orb.min_matches) {
            result.valid = true;
            result.transform = cv::Matx23f(1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
            result.registered_bgr = frame.bgr.clone();
            cv::cvtColor(result.registered_bgr, result.registered_hsv, cv::COLOR_BGR2HSV);
            return result;
        }

        std::vector<cv::Point2f> pts_cur(good_matches.size());
        std::vector<cv::Point2f> pts_ref(good_matches.size());
        for (size_t i = 0; i < good_matches.size(); ++i) {
            pts_cur[i] = cur_kp[good_matches[i].queryIdx].pt;
            pts_ref[i] = ref_kp_[good_matches[i].trainIdx].pt;
        }

        cv::Mat inlier_mask;
        cv::Mat H = cv::findHomography(
            pts_cur, pts_ref, cv::RANSAC, config_.image_registrator_orb.ransac_reproj_threshold, inlier_mask);

        if (H.empty()) {
            result.valid = true;
            result.transform = cv::Matx23f(1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
            result.registered_bgr = frame.bgr.clone();
            cv::cvtColor(result.registered_bgr, result.registered_hsv, cv::COLOR_BGR2HSV);
            return result;
        }

        const float out_scale = 0.5F;
        cv::Mat H_scaled = H.clone();
        H_scaled.row(0) *= out_scale;
        H_scaled.row(1) *= out_scale;
        cv::Size out_size(
            static_cast<int>(reference.reference_frame.bgr.cols * out_scale),
            static_cast<int>(reference.reference_frame.bgr.rows * out_scale));
        cv::Mat registered_bgr;
        cv::warpPerspective(frame.bgr, registered_bgr, H_scaled, out_size);

        result.valid = true;
        result.transform = cv::Matx23f(
            static_cast<float>(H_scaled.at<double>(0, 0)), static_cast<float>(H_scaled.at<double>(0, 1)),
            static_cast<float>(H_scaled.at<double>(0, 2)), static_cast<float>(H_scaled.at<double>(1, 0)),
            static_cast<float>(H_scaled.at<double>(1, 1)), static_cast<float>(H_scaled.at<double>(1, 2)));
        result.registered_bgr = registered_bgr;
        cv::cvtColor(registered_bgr, result.registered_hsv, cv::COLOR_BGR2HSV);

        return result;
    }

private:
    PipelineConfig config_;
    cv::Ptr<cv::ORB> orb_;
    int cached_ref_frame_index_ = -1;
    std::vector<cv::KeyPoint> ref_kp_;
    cv::Mat ref_desc_;
    cv::Mat feature_mask_;
};

} // namespace rmcs_hero_lob
