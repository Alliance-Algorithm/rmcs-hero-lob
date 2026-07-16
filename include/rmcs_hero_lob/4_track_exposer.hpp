#pragma once

#include "rmcs_hero_lob/configs.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

class TrackerProcessorFast {
public:
    struct ComponentInfo {
        cv::Rect bbox;
        cv::Point2f centroid = {};
        cv::Point2f smoothed_velocity = {};
        bool velocity_initialized = false;
    };

    explicit TrackerProcessorFast(const PipelineConfig& config)
        : config_(config) {}

    TrajectoryResult Process(const ForegroundMaskResult& foreground_mask) {
        TrajectoryResult result;
        if (!foreground_mask.valid || foreground_mask.candidate_mask.empty()) {
            if (frame_count_ > 0) {
                result.valid = true;
                result.trajectory_layer = trajectory_layer_;
                result.accumulated_frames = frame_count_;
            }
            return result;
        }
        const auto& tw = config_.trajectory_window;

        cv::Mat labels, stats, centroids;
        int num_labels =
            cv::connectedComponentsWithStats(foreground_mask.candidate_mask, labels, stats, centroids, 8, CV_32S);

        std::vector<ComponentInfo> current_components;
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < tw.min_component_area_pixels) {
                continue;
            }
            ComponentInfo comp;
            comp.centroid = cv::Point2f(
                static_cast<float>(centroids.at<double>(i, 0)), static_cast<float>(centroids.at<double>(i, 1)));
            comp.bbox = cv::Rect(
                stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
            current_components.push_back(comp);
        }

        if (trajectory_layer_.empty()) {
            trajectory_layer_ = cv::Mat::zeros(foreground_mask.candidate_mask.size(), CV_32FC3);
            exposure_count_ = cv::Mat::zeros(foreground_mask.candidate_mask.size(), CV_32FC1);
        }

        for (auto& comp : current_components) {
            const ComponentInfo* best_match = nullptr;
            float best_dist = std::numeric_limits<float>::max();
            for (const auto& prev : previous_components_) {
                float dx = comp.centroid.x - prev.centroid.x;
                float dy = comp.centroid.y - prev.centroid.y;
                float dist_sq = dx * dx + dy * dy;
                if (dist_sq < best_dist) {
                    best_dist = dist_sq;
                    best_match = &prev;
                }
            }
            best_dist = std::sqrt(best_dist);

            bool should_accumulate = false;
            if (best_match != nullptr && best_dist < tw.component_match_max_distance_pixels) {
                comp.velocity_initialized = best_match->velocity_initialized;
                double dt = foreground_mask.timestamp_seconds - previous_timestamp_;
                if (dt > 1e-6) {
                    cv::Point2f raw_velocity(
                        static_cast<float>((comp.centroid.x - best_match->centroid.x) / dt),
                        static_cast<float>((comp.centroid.y - best_match->centroid.y) / dt));
                    if (!comp.velocity_initialized) {
                        comp.smoothed_velocity = raw_velocity;
                        comp.velocity_initialized = true;
                    } else {
                        float alpha = tw.velocity_smoothing_alpha;
                        comp.smoothed_velocity = alpha * raw_velocity + (1.0F - alpha) * best_match->smoothed_velocity;
                    }
                }
                float angle = std::abs(
                    std::atan2(comp.smoothed_velocity.x, comp.smoothed_velocity.y) * 180.0F / static_cast<float>(M_PI));
                if (angle < tw.vertical_motion_half_angle_degrees) {
                    should_accumulate = true;
                }
            } else {
                should_accumulate = true;
            }

            if (should_accumulate) {
                int label_id = labels.at<int>(static_cast<int>(comp.centroid.y), static_cast<int>(comp.centroid.x));
                cv::Mat mask_roi = (labels(comp.bbox) == label_id);

                cv::Mat bgr_roi = foreground_mask.candidate_bgr(comp.bbox);
                cv::Mat traj_roi = trajectory_layer_(comp.bbox);
                cv::Mat count_roi = exposure_count_(comp.bbox);

                cv::Mat bgr_f;
                bgr_roi.convertTo(bgr_f, CV_32F);

                cv::Mat mask_f;
                mask_roi.convertTo(mask_f, CV_32F, 1.0 / 255.0);

                cv::Mat mask_3ch;
                cv::merge(std::vector<cv::Mat>{mask_f, mask_f, mask_f}, mask_3ch);

                cv::Mat masked_bgr;
                cv::multiply(bgr_f, mask_3ch, masked_bgr);
                cv::add(traj_roi, masked_bgr, traj_roi);

                cv::add(count_roi, mask_f, count_roi);
            }
        }

        previous_components_ = current_components;
        previous_timestamp_ = foreground_mask.timestamp_seconds;
        frame_count_++;
        result.valid = true;
        result.trajectory_layer = trajectory_layer_;
        result.exposure_count = exposure_count_;
        result.accumulated_frames = frame_count_;
        return result;
    }

    void Reset() {
        previous_components_.clear();
        trajectory_layer_ = cv::Mat();
        exposure_count_ = cv::Mat();
        mask_buffer_ = cv::Mat();
        frame_count_ = 0;
        previous_timestamp_ = 0.0;
    }

private:
    PipelineConfig config_;
    std::vector<ComponentInfo> previous_components_;
    cv::Mat trajectory_layer_;
    cv::Mat exposure_count_;
    cv::Mat mask_buffer_;
    int frame_count_ = 0;
    double previous_timestamp_ = 0.0;
};

} // namespace rmcs_hero_lob
