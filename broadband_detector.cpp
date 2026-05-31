#include <cmath>
#include <fmt/core.h>

#include "broadband_detector.h"

void BroadbandDetector::BroadbandDetect(int n_subband, int nf_subband, int n_subband_dilation,
                                        int debug, float bb_det_threshold, float bb_det_threshold_sk,
                                        float f0_sb_MHz, float df_sb_MHz,
                                        const float* subband_std_bb_det, const float* blk_sk,
                                        const float* cpu_column_sums, const float* cpu_subband_mean,
                                        const float* cpu_subband_std) {
  int bb_subband_prelim_det_count = 0;

  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    if (subband_std_bb_det[i_subband] > bb_det_threshold) {
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

  for (int i_bb_det = 0; i_bb_det < n_bb_det_; i_bb_det++) {
    int i_bb_det1 = bb_det_[i_bb_det].sb1 * nf_subband;
    int n_bb_pts = (bb_det_[i_bb_det].sb2 - bb_det_[i_bb_det].sb1 + 1) * nf_subband;
    float peak_value = StatsUtil::max(&cpu_column_sums[i_bb_det1], n_bb_pts);
    int i_subband1 = bb_det_[i_bb_det].sb1;
    bb_det_[i_bb_det].snr =
        (peak_value - cpu_subband_mean[i_subband1]) / cpu_subband_std[i_subband1];

    int i_bb_sb1 = bb_det_[i_bb_det].sb1;
    int n_bb_sb = bb_det_[i_bb_det].sb2 - bb_det_[i_bb_det].sb1 + 1;
    bb_det_[i_bb_det].peak_blk_sk = StatsUtil::max(&blk_sk[i_bb_sb1], n_bb_sb);
  }

  if (debug >= 1) {
    for (int i_bb_det = 0; i_bb_det < n_bb_det_; i_bb_det++) {
      fmt::print("BB det {:3d}: subband {:3d} - {:3d}, {:8.2f} - {:8.2f} MHz, center {:8.2f} MHz, BW {:5.0f} KHz, "
                 "Peak BlkSK {:5.2f}, SNR {:5.2f} dB\n",
                 i_bb_det, bb_det_[i_bb_det].sb1, bb_det_[i_bb_det].sb2, bb_det_[i_bb_det].f1_MHz,
                 bb_det_[i_bb_det].f2_MHz, bb_det_[i_bb_det].fctr_MHz, bb_det_[i_bb_det].bw_MHz * 1e3,
                 bb_det_[i_bb_det].peak_blk_sk, 10. * log10(bb_det_[i_bb_det].snr));
    }
  }
}
