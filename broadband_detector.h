#pragma once

#include "detection.h"

struct BBdet {
  int sb1 = 0;
  int sb2 = 0;
  float f1_MHz = 0.f;
  float f2_MHz = 0.f;
  float fctr_MHz = 0.f;
  float bw_MHz = 0.f;
  float snr = 0.f;
  float peak_blk_sk = 0.f;
};

class BroadbandDetector {
 public:
  void BroadbandDetect(int n_subband, int nf_subband, int n_subband_dilation, int debug,
                       float bb_det_threshold, float bb_det_threshold_sk, float f0_sb_MHz, float df_sb_MHz,
                       const float* subband_std_bb_det, const float* blk_sk,
                       const float* cpu_column_sums, const float* cpu_subband_mean,
                       const float* cpu_subband_std);

  const float* subbandDetected() const { return bb_subband_detected_; }
  int nDetections() const { return n_bb_det_; }
  const BBdet* detections() const { return bb_det_; }

 private:
  float bb_subband_detected_[SubbandNormalizer::kNominalSubbands];
  BBdet bb_det_[SubbandNormalizer::kNominalSubbands / 2];
  int n_bb_det_ = 0;
};
