#include <gtest/gtest.h>

#include "rmcs_hero_lob/2_image_stabilizer.hpp"
#include "rmcs_hero_lob/pipeline.hpp"

namespace rmcs_hero_lob {
namespace {

constexpr int kProcessingWidth = 720;
constexpr int kProcessingHeight = 540;
constexpr int kRawWidth = 1440;
constexpr int kRawHeight = 1080;
constexpr int kOutputWidth = 288;
constexpr int kOutputHeight = 216;

cv::Mat MakeImage(int width, int height, const cv::Scalar& color) {
    return cv::Mat(height, width, CV_8UC3, color);
}

ReferenceFrameResult MakeReference(const cv::Mat& bgr) {
    ReferenceFrameResult reference;
    reference.has_reference = true;
    reference.mode = ReferenceMode::kDynamic;
    reference.reference_frame.frame_index = 1;
    reference.reference_frame.bgr = bgr.clone();
    cv::cvtColor(reference.reference_frame.bgr, reference.reference_frame.hsv, cv::COLOR_BGR2HSV);
    return reference;
}

} // namespace

TEST(ImageRegistratorOrbSizeTest, FallbackKeepsReferenceSizeWhenCurrentFrameIsRawSize) {
    PipelineConfig config;
    ImageRegistratorOrb registrator(config);

    const auto reference = MakeReference(MakeImage(kProcessingWidth, kProcessingHeight, cv::Scalar(16, 16, 16)));
    FrameData frame;
    frame.frame_index = 2;
    frame.timestamp_seconds = 2.0 / 120.0;
    frame.bgr = MakeImage(kRawWidth, kRawHeight, cv::Scalar(64, 64, 64));

    const auto result = registrator.Process(reference, frame);

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.registered_bgr.cols, kProcessingWidth);
    EXPECT_EQ(result.registered_bgr.rows, kProcessingHeight);
    EXPECT_EQ(result.registered_hsv.cols, kProcessingWidth);
    EXPECT_EQ(result.registered_hsv.rows, kProcessingHeight);
}

TEST(PipelineSizeTest, ProcessingAndFinalOutputsUseConfiguredSizes) {
    PipelineConfig config;
    Pipeline pipeline(config);

    std::vector<cv::Mat> history{
        MakeImage(kProcessingWidth, kProcessingHeight, cv::Scalar(10, 10, 10)),
        MakeImage(kProcessingWidth, kProcessingHeight, cv::Scalar(20, 20, 20)),
        MakeImage(kProcessingWidth, kProcessingHeight, cv::Scalar(30, 30, 30)),
    };

    pipeline.SetReferenceFromHistory(history, 1);
    const auto reference = pipeline.GetReferenceFrameSelector().GetReference();
    ASSERT_TRUE(reference.has_reference);
    EXPECT_EQ(reference.reference_frame.bgr.cols, kProcessingWidth);
    EXPECT_EQ(reference.reference_frame.bgr.rows, kProcessingHeight);

    FrameData frame;
    frame.frame_index = 2;
    frame.timestamp_seconds = 2.0 / 120.0;
    frame.bgr = MakeImage(kProcessingWidth, kProcessingHeight, cv::Scalar(220, 220, 220));
    cv::cvtColor(frame.bgr, frame.hsv, cv::COLOR_BGR2HSV);

    const auto trajectory = pipeline.ProcessFrame(frame);
    ASSERT_TRUE(trajectory.valid);
    EXPECT_EQ(trajectory.trajectory_layer.cols, kProcessingWidth);
    EXPECT_EQ(trajectory.trajectory_layer.rows, kProcessingHeight);
    EXPECT_EQ(trajectory.exposure_count.cols, kProcessingWidth);
    EXPECT_EQ(trajectory.exposure_count.rows, kProcessingHeight);

    const auto compressed = pipeline.Finalize();
    ASSERT_TRUE(compressed.valid);
    EXPECT_EQ(compressed.output_image.cols, kOutputWidth);
    EXPECT_EQ(compressed.output_image.rows, kOutputHeight);
}

} // namespace rmcs_hero_lob
