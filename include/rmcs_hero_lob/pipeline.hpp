#pragma once

#include <string>

#include "rmcs_hero_lob/1_reference_frame_selector.hpp"
#include "rmcs_hero_lob/2_image_stabilizer.hpp"
#include "rmcs_hero_lob/3_foreground_extractor.hpp"
#include "rmcs_hero_lob/4_track_exposer.hpp"
#include "rmcs_hero_lob/5_image_synthesis.hpp"
#include "rmcs_hero_lob/6_image_compress.hpp"

#include "rmcs_hero_lob/configs.hpp"

namespace rmcs_hero_lob {

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config = {});

    bool Run(const std::string& input_video, const std::string& output_image);

private:
    PipelineConfig config_;
    ReferenceFrameSelector reference_frame_selector_;
    ImageRegistratorOrb image_registrator_;
    BackgroundRemover background_remover_;
    TrackerProcessorFast tracker_processor_fast_;
    ImageSynthesis image_synthesis_;
};

} // namespace rmcs_hero_lob
