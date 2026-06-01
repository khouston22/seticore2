#include <cmath>
#include <fmt/core.h>

#include "broadband_detector.h"

// BroadbandDetector: subband broadband detection

// Detect broadband RFI from elevated subband std; dilate, cluster, and score segments
void BroadbandDetector::BroadbandDetect(int n_subband, int nf_subband, int n_subband_dilation,
                                        int coarse_channel, int debug, float n_avg,
                                        float f0_sb_MHz, float df_sb_MHz, const float* cpu_column_sums,
                                        const float* cpu_subband_mean, const float* cpu_subband_std,
                                        const float* subband_mean_no_clip,
                                        const float* subband_std_no_clip) {
  // Block SK statistics and detection thresholds
  float blk_sk_clip_mean, blk_sk_clip_std;
  float blk_sk_no_clip_mean, blk_sk_no_clip_std;

  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    blk_sk_clip_[i_subband] =
        (2 * nf_subband * n_avg + 1.) / nf_subband *
        pow(cpu_subband_std[i_subband] / cpu_subband_mean[i_subband], 2.0);
    blk_sk_no_clip_[i_subband] =
        (2 * nf_subband * n_avg + 1.) / nf_subband *
        pow(subband_std_no_clip[i_subband] / subband_mean_no_clip[i_subband], 2.0);
  }
  StatsUtil::meanStdDev(blk_sk_clip_, n_subband, &blk_sk_clip_mean, &blk_sk_clip_std);
  StatsUtil::meanStdDev(blk_sk_no_clip_, n_subband, &blk_sk_no_clip_mean, &blk_sk_no_clip_std);

  if (debug >= 1 && coarse_channel == 0) {
    fmt::print("chnl {} n_subband={} sigma clipped mean values after scale (x1000):\n", coarse_channel,
               n_subband);
    StatsUtil::printXSegment(const_cast<float*>(cpu_subband_mean), n_subband, 1000.0);
    fmt::print("chnl {} n_subband={} sigma clipped std  values after scale (x1000):\n", coarse_channel,
               n_subband);
    StatsUtil::printFXSegment(const_cast<float*>(cpu_subband_std), n_subband, 1000.0, f0_sb_MHz,
                              df_sb_MHz);
  }

  if (debug >= 2 && coarse_channel == 0) {
    fmt::print("chnl {} n_subband={} no clip std  values after scale (x1000):\n", coarse_channel,
               n_subband);
    StatsUtil::printFXSegment(const_cast<float*>(subband_std_no_clip), n_subband, 1000.0, f0_sb_MHz,
                              df_sb_MHz);
  }

  float subband_std_mean_nominal = 1.0f / sqrt(2 * n_avg);
  float subband_std_mean_norm[SubbandNormalizer::kNominalSubbands];
  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    subband_std_mean_norm[i_subband] =
        cpu_subband_std[i_subband] / cpu_subband_mean[i_subband] / subband_std_mean_nominal;
  }

  if (debug >= 1) {
    fmt::print("chnl {} n_subband={} sigma clipped std/mean values over expected after scale (x100):\n",
               coarse_channel, n_subband);
    StatsUtil::printFXSegment(subband_std_mean_norm, n_subband, 100.0, f0_sb_MHz, df_sb_MHz);

    fmt::print("chnl {} n_subband={} clipped SK  values after scale (x100), mean={:.3f}, std={:.3f}:\n",
               coarse_channel, n_subband, blk_sk_clip_mean, blk_sk_clip_std);
    StatsUtil::printFXSegment(blk_sk_clip_, n_subband, 100.0, f0_sb_MHz, df_sb_MHz);
    fmt::print("chnl {} n_subband={} no clip SK  values after scale (x100), mean={:.3f}, std={:.3f}:\n",
               coarse_channel, n_subband, blk_sk_no_clip_mean, blk_sk_no_clip_std);
    StatsUtil::printFXSegment(blk_sk_no_clip_, n_subband, 100.0, f0_sb_MHz, df_sb_MHz);
  }

  float bb_z_det = 5.f;
  float bb_det_threshold = 1.02f / sqrt(2 * n_avg) * (1.f + bb_z_det / sqrt(nf_subband));
  float bb_det_threshold_sk = pow(bb_det_threshold, 2.0) * 2 * n_avg;

  int bb_subband_prelim_det_count = 0;

  // Threshold subband std for preliminary detections
  // Use float value as flag for simplify printout in debug
  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    if (subband_std_no_clip[i_subband] > bb_det_threshold) {
      bb_subband_detected_[i_subband] = 1.0f;
      bb_subband_prelim_det_count++;
    } else {
      bb_subband_detected_[i_subband] = 0.0f;
    }
  }

  if (debug >= 1) {
    if (bb_subband_prelim_det_count == 0) {
      fmt::print("Broadband detections, threshold={:.3f} (SK {:.2f}): No BB detections\n",
                 bb_det_threshold, bb_det_threshold_sk);
    }
  }

  if (n_subband_dilation>0) {
    // Dilate detection flags at rising/falling edges
    // This expands each broadband detection by n_subband_dilation subbands to left or right,
    // possibly combining multiple BB detections into one
    for (int i_subband = n_subband_dilation; i_subband < n_subband; i_subband++) {
      if ((bb_subband_detected_[i_subband] > 0.f) &&
          (bb_subband_detected_[i_subband - 1] == 0.f)) {
        for (int i_edge = 1; i_edge <= n_subband_dilation; i_edge++) {
          bb_subband_detected_[i_subband - i_edge] = 1.0f;
        }
      }
    }
    for (int i_subband = n_subband - n_subband_dilation - 1; i_subband >= 0; i_subband--) {
      if ((bb_subband_detected_[i_subband] > 0.f) &&
          (bb_subband_detected_[i_subband + 1] == 0.f)) {
        for (int i_edge = 1; i_edge <= n_subband_dilation; i_edge++) {
          bb_subband_detected_[i_subband + i_edge] = 1.0f;
        }
      }
    }
  }

  // Count dilated detections and optional debug print
  int bb_subband_det_count = 0;
  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    if (bb_subband_detected_[i_subband] > 0.f) {
      bb_subband_det_count++;
    }
  }

  if (debug >= 1) {
    if (bb_subband_det_count == 0) {
      fmt::print("Broadband detections after dilation ({}), threshold={:.3f} (SK {:.2f}): No BB detections\n",
                 n_subband_dilation, bb_det_threshold, bb_det_threshold_sk);
    } else {
      fmt::print("Broadband detections after dilation ({}), threshold={:.3f} (SK {:.2f}): {} subband "
                 "detections\n",
                 n_subband_dilation, bb_det_threshold, bb_det_threshold_sk, bb_subband_det_count);
      StatsUtil::printFXSegment(bb_subband_detected_, n_subband, 1.0, f0_sb_MHz, df_sb_MHz);
    }
  }

  // Cluster contiguous subbands into BBdet segments
  int bb_det_idx = -1;
  bool in_bb_cluster = false;

  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    if (bb_subband_detected_[i_subband] > 0.f) {
      if (!in_bb_cluster) {
        in_bb_cluster = true;
        bb_det_idx++;
        bb_det_[bb_det_idx].sb1 = i_subband;
        if (df_sb_MHz >= 0.) {
          bb_det_[bb_det_idx].f1_MHz = f0_sb_MHz + (i_subband - 0.5) * df_sb_MHz;
        } else {
          bb_det_[bb_det_idx].f2_MHz = f0_sb_MHz + (i_subband - 0.5) * df_sb_MHz;
        }
      }
      bb_det_[bb_det_idx].sb2 = i_subband;
      if (df_sb_MHz >= 0.) {
        bb_det_[bb_det_idx].f2_MHz = f0_sb_MHz + (i_subband + 0.5) * df_sb_MHz;
      } else {
        bb_det_[bb_det_idx].f1_MHz = f0_sb_MHz + (i_subband + 0.5) * df_sb_MHz;
      }
      int i_subband_ctr = (bb_det_[bb_det_idx].sb1 + bb_det_[bb_det_idx].sb2) / 2;
      bb_det_[bb_det_idx].fctr_MHz = f0_sb_MHz + i_subband_ctr * df_sb_MHz;
      bb_det_[bb_det_idx].bw_MHz = bb_det_[bb_det_idx].f2_MHz - bb_det_[bb_det_idx].f1_MHz;
    } else {
      in_bb_cluster = false;
    }
  }
  
  n_bb_det_ = bb_det_idx + 1;

  // Peak SNR and block SK per broadband detection
  for (int i_bb_det = 0; i_bb_det < n_bb_det_; i_bb_det++) {
    int i_bb_det1 = bb_det_[i_bb_det].sb1 * nf_subband;
    int n_bb_pts = (bb_det_[i_bb_det].sb2 - bb_det_[i_bb_det].sb1 + 1) * nf_subband;
    float peak_value = StatsUtil::max(&cpu_column_sums[i_bb_det1], n_bb_pts);
    int i_subband1 = bb_det_[i_bb_det].sb1;
    bb_det_[i_bb_det].snr =
        (peak_value - cpu_subband_mean[i_subband1]) / cpu_subband_std[i_subband1];

    int i_bb_sb1 = bb_det_[i_bb_det].sb1;
    int n_bb_sb = bb_det_[i_bb_det].sb2 - bb_det_[i_bb_det].sb1 + 1;
    bb_det_[i_bb_det].peak_blk_sk = StatsUtil::max(&blk_sk_no_clip_[i_bb_sb1], n_bb_sb);
  }

  // Debug summary of each detection
  if (debug >= 1) {
    for (int i_bb_det = 0; i_bb_det < n_bb_det_; i_bb_det++) {
      fmt::print("BB det {:3d}: subband {:3d} - {:3d}, {:8.2f} - {:8.2f} MHz, center {:8.2f} MHz, BW {:5.0f} KHz, "
                 "Peak BlkSK {:5.2f}, SNR {:5.2f} dB\n",
                 i_bb_det, bb_det_[i_bb_det].sb1, bb_det_[i_bb_det].sb2, bb_det_[i_bb_det].f1_MHz,
                 bb_det_[i_bb_det].f2_MHz, bb_det_[i_bb_det].fctr_MHz, bb_det_[i_bb_det].bw_MHz * 1e3,
                 bb_det_[i_bb_det].peak_blk_sk, 10. * log10(bb_det_[i_bb_det].snr));
    }
    fmt::print("\n");
  }
}
