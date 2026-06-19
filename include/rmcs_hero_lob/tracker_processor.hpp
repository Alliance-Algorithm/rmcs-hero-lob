#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rmcs_hero_lob/vision_data.hpp"

namespace rmcs_hero_lob {

class TrackerProcessor {
public:
    struct ComponentInfo {
        cv::Mat mask;
        cv::Point2f centroid = {};
        cv::Point2f smoothed_velocity = {};
        bool velocity_initialized = false;
    };

    explicit TrackerProcessor(const TrajectoryWindowConfig& config) : config_(config) {}

    TrajectoryResult process(const ForegroundResult& foreground) {
        TrajectoryResult result;
        if (!foreground.valid || foreground.candidate_mask.empty()) {
            if (frame_count_ > 0) {
                result.valid = true;
                result.trajectory_layer = trajectory_layer_;
                result.accumulated_frames = frame_count_;
            }
            return result;
        }
        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(
            foreground.candidate_mask, labels, stats, centroids, 8, CV_32S);
        std::vector<ComponentInfo> current_components;
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < config_.min_component_area_pixels) {
                continue;
            }
            ComponentInfo comp;
            comp.centroid = cv::Point2f(
                static_cast<float>(centroids.at<double>(i, 0)),
                static_cast<float>(centroids.at<double>(i, 1)));
            comp.mask = (labels == i);
            current_components.push_back(comp);
        }
        if (trajectory_layer_.empty()) {
            trajectory_layer_ = cv::Mat::zeros(
                foreground.candidate_mask.size(), CV_32FC3);
        }
        for (auto& comp : current_components) {
            const ComponentInfo* best_match = nullptr;
            float best_dist = std::numeric_limits<float>::max();
            for (const auto& prev : previous_components_) {
                float dx = comp.centroid.x - prev.centroid.x;
                float dy = comp.centroid.y - prev.centroid.y;
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_match = &prev;
                }
            }
            bool should_accumulate = false;
            if (best_match != nullptr &&
                best_dist < config_.component_match_max_distance_pixels) {
                comp.velocity_initialized = best_match->velocity_initialized;
                double dt = current_timestamp_ - previous_timestamp_;
                if (dt > 1e-6) {
                    cv::Point2f raw_velocity(
                        static_cast<float>((comp.centroid.x - best_match->centroid.x) / dt),
                        static_cast<float>((comp.centroid.y - best_match->centroid.y) / dt));
                    if (!comp.velocity_initialized) {
                        comp.smoothed_velocity = raw_velocity;
                        comp.velocity_initialized = true;
                    } else {
                        float alpha = config_.velocity_smoothing_alpha;
                        comp.smoothed_velocity =
                            alpha * raw_velocity +
                            (1.0F - alpha) * best_match->smoothed_velocity;
                    }
                }
                float angle = std::abs(
                    std::atan2(comp.smoothed_velocity.x, comp.smoothed_velocity.y) *
                    180.0F / static_cast<float>(M_PI));
                if (angle < config_.vertical_motion_half_angle_degrees) {
                    should_accumulate = true;
                }
            } else {
                should_accumulate = true;
            }
            if (should_accumulate) {
                cv::Rect bbox = cv::boundingRect(comp.mask);
                cv::Mat mask_roi = comp.mask(bbox);
                cv::Mat bgr_roi = foreground.candidate_bgr(bbox);
                cv::Mat traj_roi = trajectory_layer_(bbox);
                cv::Mat mask_f;
                mask_roi.convertTo(mask_f, CV_32F, 1.0 / 255.0);
                for (int r = 0; r < bbox.height; ++r) {
                    const float* mask_row = mask_f.ptr<float>(r);
                    const uchar* bgr_row = bgr_roi.ptr<uchar>(r);
                    float* traj_row = traj_roi.ptr<float>(r);
                    for (int col = 0; col < bbox.width; ++col) {
                        float m = mask_row[col];
                        if (m < 1e-6F) {
                            continue;
                        }
                        int idx = col * 3;
                        traj_row[idx + 0] += bgr_row[idx + 0] * m;
                        traj_row[idx + 1] += bgr_row[idx + 1] * m;
                        traj_row[idx + 2] += bgr_row[idx + 2] * m;
                    }
                }
            }
        }
        previous_components_ = current_components;
        previous_timestamp_ = current_timestamp_;
        frame_count_++;
        result.valid = true;
        result.trajectory_layer = trajectory_layer_;
        result.accumulated_frames = frame_count_;
        return result;
    }

    void reset() {
        previous_components_.clear();
        trajectory_layer_ = cv::Mat();
        frame_count_ = 0;
        previous_timestamp_ = 0.0;
    }

    void set_timestamp(double ts) { current_timestamp_ = ts; }

private:
    TrajectoryWindowConfig config_;
    std::vector<ComponentInfo> previous_components_;
    cv::Mat trajectory_layer_;
    int frame_count_ = 0;
    double previous_timestamp_ = 0.0;
    double current_timestamp_ = 0.0;
};

} // namespace rmcs_hero_lob
