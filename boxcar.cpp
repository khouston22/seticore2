#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fmt/core.h>

#include "boxcar.h"

namespace {

// Boxcar: binary decomposition helpers

// Return bit i of n (used to select power-of-2 boxcar terms)
int readBit(int n, int bit_idx) {
  return (n & (1 << bit_idx)) >> bit_idx;
}

}  // namespace

// BoxcarWorkspace: CPU boxcar filtering and debug

// Build zero-padded power-of-2 boxcar prefix-sum table
void BoxcarWorkspace::computeP2SumsCpu(const float* dd_sums_line, int log2_max_p2_runtime,
                                       float* p2_path_sums, int n_zp) const {
  int n_freq = num_freq_;
  int nbox_p2_max = 1 << log2_max_p2_runtime;
  int nbox_max = 2 * nbox_p2_max - 1;
  int n_freq_ext = n_freq + 2 * n_zp;

  assert(n_zp >= nbox_max);

  // Zero-pad both ends of each p2 row
  for (int i_nbox = 0; i_nbox <= log2_max_p2_runtime; i_nbox++) {
    float* p2_row = &p2_path_sums[i_nbox * n_freq_ext];
    memset(&p2_row[0], 0, n_zp * sizeof(float));
    memset(&p2_row[n_freq + n_zp], 0, n_zp * sizeof(float));
  }

  // Copy input line into center of p2=1 row
  float* p2_row = &p2_path_sums[0];
  for (int i_freq = n_zp; i_freq < n_freq + n_zp; i_freq++) {
    p2_row[i_freq] = dd_sums_line[i_freq - n_zp];
  }

  // Build p2=2,4,... rows by pairwise sum with doubling stride
  int stride = 1;
  for (int i_nbox = 1; i_nbox <= log2_max_p2_runtime; i_nbox++) {
    float* p2_row_new = &p2_path_sums[i_nbox * n_freq_ext];
    p2_row = &p2_path_sums[(i_nbox - 1) * n_freq_ext];
    int nbox_p2 = 1 << i_nbox;

    for (int i_freq = 0; i_freq < n_freq + n_zp + nbox_p2; i_freq++) {
      p2_row_new[i_freq + stride] = p2_row[i_freq] + p2_row[i_freq + stride];
    }
    stride = 2 * stride;
  }
}

// Compose arbitrary-width boxcar sum from p2 table via binary decomposition
void BoxcarWorkspace::computeSumCpu(const float* p2_path_sums, float* work, int nbox,
                                    int log2_max_p2_runtime, int n_zp,
                                    float* nbox_path_sum) const {
  int n_freq = num_freq_;
  assert(nbox >= 1);
  assert(nbox < (1 << (log2_max_p2_runtime + 1)));

  int n_freq_ext = n_freq + 2 * n_zp;
  memset(work, 0, 2 * n_freq_ext * sizeof(float));

  float* work_out_row = &work[0];
  const float* p2_row = &p2_path_sums[0];

  // Seed work row from p2=1 if bit 0 of nbox is set
  if (readBit(nbox, 0)) {
    for (int i_freq = 0; i_freq < n_freq + 2 * n_zp; i_freq++) {
      work_out_row[i_freq] = p2_row[i_freq];
    }
  }

  int work_out_idx = 0;
  int work_in_idx = 1;

  // Add selected p2 rows into double-buffered work rows
  for (int i_nbox = 1; i_nbox <= log2_max_p2_runtime; i_nbox++) {
    int nbox_p2 = 1 << i_nbox;
    if (readBit(nbox, i_nbox) == 1) {
      work_in_idx = work_out_idx;
      work_out_idx = 1 - work_out_idx;

      work_out_row = &work[work_out_idx * n_freq_ext];
      const float* work_in_row = &work[work_in_idx * n_freq_ext];
      p2_row = &p2_path_sums[i_nbox * n_freq_ext];

      for (int i_freq = nbox_p2; i_freq < n_freq + n_zp + nbox_p2; i_freq++) {
        work_out_row[i_freq] = work_in_row[i_freq - nbox_p2] + p2_row[i_freq];
      }
    }
  }

  // Extract centered, normalized boxcar sum into output
  int shift = nbox / 2;
  float scale = 1.f / nbox;
  for (int i_freq = 0; i_freq < n_freq; i_freq++) {
    nbox_path_sum[i_freq] = work_out_row[n_zp + i_freq + shift] * scale;
  }
}

// Build boxcar widths for drift search (drift width plus powers of two or four)
// nbox will be incremented by 4x (5log10(4) = 3 dB per increment) beyond what is required
// to compensate for drift alone
// outputs a list of nbox values to evaluate
// max_nbox_bw is the max desired nbox size in bins
// nbox_max is the max nbox value according to allocated memory
vector<int> BoxcarWorkspace::buildNboxList(int drift_block, int max_nbox_bw, int nbox_max) {
  vector<int> nbox_list;
  nbox_list.reserve(32);

  int nbox_drift;
  if (drift_block >= 0) {
    nbox_drift = drift_block + 1;
  } else {
    nbox_drift = -drift_block;
  }

  nbox_list.push_back(nbox_drift);

  float min_nbox_bw = 2.0f * nbox_drift;
  if (max_nbox_bw > min_nbox_bw) {
    int i_bw_max = static_cast<int>(floor(log2(max(1, min(max_nbox_bw,nbox_max)+1))));
    int i_bw_min = static_cast<int>(ceil(log2(min_nbox_bw)));
    if (((i_bw_max-i_bw_min)%2)>0) {
      i_bw_min++;
    }
    if (i_bw_max >= i_bw_min) {
      for (int i_bw = i_bw_min; i_bw <= i_bw_max; i_bw+=2) {
        nbox_list.push_back(min((1 << i_bw),nbox_max));
      }
    }
  }

  return nbox_list;
}

// Debug print boxcar width list for a drift block
void BoxcarWorkspace::printNboxList(const vector<int>& nbox_list, int drift_block, int nbox_max) {
  fmt::print("drift_block={}, max_Nbox={}, n_Nbox={}, Nbox = ", drift_block, nbox_max, nbox_list.size());
  for (int nbox : nbox_list) {
    fmt::print("{} ", nbox);
  }
  fmt::print("\n");
}

// Debug print scaled segment of a boxcar array
void BoxcarWorkspace::printNboxSegment(const float* x, int n_pts, int start_offset, float scale) {
  for (int i_ofs = start_offset; i_ofs < start_offset + n_pts; i_ofs++) {
    if (i_ofs % 10 == 0) {
      fmt::print("\n{:6d}   ", i_ofs);
    }
    fmt::print("{:8.0f} ", x[i_ofs] * scale);
  }
  fmt::print("\n");
}
