#pragma once

#include "rmcs_hero_lob/configs.hpp"

namespace rmcs_hero_lob {

class ImageSynthesis {
public:
    explicit ImageSynthesis(const PipelineConfig& config);

    SynthesisResult Process(const ReferenceFrameResult& reference, const TrajectoryResult& trajectory) const;

private:
    PipelineConfig config_;
};

} // namespace rmcs_hero_lob
