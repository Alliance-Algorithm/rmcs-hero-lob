#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "rmcs_hero_lob/1_reference_frame_selector.hpp"
#include "rmcs_hero_lob/2_image_stabilizer.hpp"
#include "rmcs_hero_lob/3_foreground_extractor.hpp"
#include "rmcs_hero_lob/4_track_exposer.hpp"
#include "rmcs_hero_lob/5_image_synthesis.hpp"
#include "rmcs_hero_lob/6_image_compress.hpp"

#include "rmcs_hero_lob/configs.hpp"

#include <opencv2/imgproc.hpp>

namespace rmcs_hero_lob {

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config = {})
        : config_(config)
        , reference_frame_selector_(config)
        , image_registrator_(config)
        , background_remover_(config)
        , tracker_processor_fast_(config)
        , image_synthesis_(config)
        , compression_(config) {}

    void SetReferenceFromHistory(const std::vector<cv::Mat>& images, int64_t frame_id) {
        if (images.empty())
            return;

        const int height = images[0].rows;
        const int width = images[0].cols;
        const int channels = images[0].channels();
        const int count = static_cast<int>(images.size());
        const int pixels = height * width;

        cv::Mat median(height, width, images[0].type(), cv::Scalar::all(0));
        std::vector<unsigned char> buffer(pixels * channels * count);

        for (int f = 0; f < count; ++f) {
            const unsigned char* src = images[f].ptr<unsigned char>(0);
            std::memcpy(buffer.data() + f * pixels * channels, src, pixels * channels);
        }

        unsigned char* result_data = median.ptr<unsigned char>(0);
        std::vector<unsigned char> channel_values(count);

        for (int i = 0; i < pixels; ++i) {
            for (int c = 0; c < channels; ++c) {
                for (int f = 0; f < count; ++f)
                    channel_values[f] = buffer[f * pixels * channels + i * channels + c];
                std::nth_element(channel_values.begin(), channel_values.begin() + count / 2, channel_values.end());
                result_data[i * channels + c] = channel_values[count / 2];
            }
        }

        reference_frame_selector_.SetReference(median, frame_id);
    }

    TrajectoryResult ProcessFrame(const FrameData& frame) {
        ReferenceFrameResult reference = reference_frame_selector_.GetReference();

        RegistrationResult registration = image_registrator_.Process(reference, frame);

        ForegroundMaskResult foreground = background_remover_.Process(reference, registration);

        last_trajectory_ = tracker_processor_fast_.Process(foreground);
        return last_trajectory_;
    }

    CompressionResult Finalize() {
        ReferenceFrameResult reference = reference_frame_selector_.GetReference();
        TrajectoryResult trajectory = last_trajectory_;

        SynthesisResult synthesis = image_synthesis_.Process(reference, trajectory);
        if (!synthesis.valid || synthesis.output_image.empty()) {
            return {};
        }

        return compression_.Process(synthesis);
    }

    void ResetTracker() { tracker_processor_fast_.Reset(); }

    cv::Mat CompressImage(const cv::Mat& image) const {
        if (image.empty())
            return {};
        const auto& bg = config_.compression;
        if (bg.background_output_width <= 0 || bg.background_output_height <= 0)
            return image;
        cv::Mat result;
        cv::resize(
            image, result, cv::Size(bg.background_output_width, bg.background_output_height), 0, 0,
            cv::INTER_AREA);
        return result;
    }

    ReferenceFrameSelector& GetReferenceFrameSelector() { return reference_frame_selector_; }

    TrackerProcessorFast& GetTracker() { return tracker_processor_fast_; }

    const TrajectoryResult& GetLastTrajectory() const { return last_trajectory_; }

private:
    PipelineConfig config_;
    ReferenceFrameSelector reference_frame_selector_;
    ImageRegistratorOrb image_registrator_;
    BackgroundRemover background_remover_;
    TrackerProcessorFast tracker_processor_fast_;
    ImageSynthesis image_synthesis_;
    Compression compression_;
    TrajectoryResult last_trajectory_;
};

} // namespace rmcs_hero_lob
