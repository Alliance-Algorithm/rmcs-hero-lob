#pragma once

#include "rmcs_hero_lob/configs.hpp"
#include <opencv2/features2d.hpp>

namespace rmcs_hero_lob {

class ImageRegistratorOrb {
public:
    explicit ImageRegistratorOrb(const PipelineConfig& config);

    RegistrationResult Process(const ReferenceFrameResult& reference, const FrameData& frame);

    void PrintTimingStats() const;

private:
    PipelineConfig config_;
    cv::Ptr<cv::ORB> orb_;
    int cached_ref_frame_index_ = -1;
    std::vector<cv::KeyPoint> ref_kp_;
    cv::Mat ref_desc_;
    cv::Mat feature_mask_;

    struct TimingStats {
        double input_validation = 0.0;
        double ref_frame_cache_check = 0.0;
        double ref_frame_preprocess = 0.0;
        double cur_frame_preprocess = 0.0;
        double feature_matching = 0.0;
        double match_verification = 0.0;
        double homography_estimation = 0.0;
        double image_transform = 0.0;
        double result_output = 0.0;
        int frame_count = 0;
    };

    TimingStats timing_stats_;
};

} // namespace rmcs_hero_lob
