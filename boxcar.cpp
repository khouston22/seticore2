#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "boxcar.h"

namespace {

int readBit(int n, int bit_idx) {
  return (n & (1 << bit_idx)) >> bit_idx;
}

}  // namespace

void BoxcarWorkspace::computeP2SumsCpu(const float* dd_sums_line, int log2_max_p2_runtime,
                                       float* p2_path_sums, int n_zp) const {
  int n_freq = num_freq_;
  int nbox_p2_max = 1 << log2_max_p2_runtime;
  int nbox_max = 2 * nbox_p2_max - 1;
  int n_freq_ext = n_freq + 2 * n_zp;

  assert(n_zp >= nbox_max);

  for (int i_nbox = 0; i_nbox <= log2_max_p2_runtime; i_nbox++) {
    float* p2_row = &p2_path_sums[i_nbox * n_freq_ext];
    memset(&p2_row[0], 0, n_zp * sizeof(float));
    memset(&p2_row[n_freq + n_zp], 0, n_zp * sizeof(float));
  }

  float* p2_row = &p2_path_sums[0];
  for (int i_freq = n_zp; i_freq < n_freq + n_zp; i_freq++) {
    p2_row[i_freq] = dd_sums_line[i_freq - n_zp];
  }

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

  if (readBit(nbox, 0)) {
    for (int i_freq = 0; i_freq < n_freq + 2 * n_zp; i_freq++) {
      work_out_row[i_freq] = p2_row[i_freq];
    }
  }

  int work_out_idx = 0;
  int work_in_idx = 1;

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

  int shift = nbox / 2;
  float scale = 1.f / nbox;
  for (int i_freq = 0; i_freq < n_freq; i_freq++) {
    nbox_path_sum[i_freq] = work_out_row[n_zp + i_freq + shift] * scale;
  }
}

vector<int> BoxcarWorkspace::buildNboxList(int drift_block, int max_nbox_bw, int nbox_p2_max) {
  vector<int> nbox_list;
  nbox_list.reserve(32);

  int nbox_drift;
  if (drift_block >= 0) {
    nbox_drift = drift_block + 1;
  } else {
    nbox_drift = -drift_block;
  }

  nbox_list.push_back(nbox_drift);

  float min_nbox_bw = 1.5f * nbox_drift;
  if (max_nbox_bw > min_nbox_bw) {
    int n_nbox_bw = static_cast<int>(floor(log2(max(1, max_nbox_bw))));
    int i_bw_min = static_cast<int>(ceil(log2(min_nbox_bw)));
    if (n_nbox_bw >= i_bw_min) {
      for (int i_bw = i_bw_min; i_bw <= n_nbox_bw; i_bw++) {
        nbox_list.push_back((1 << i_bw) + 1);
      }
    }
  }

  return nbox_list;
}

void BoxcarWorkspace::printNboxList(const vector<int>& nbox_list, int drift_block) {
  printf("drift_block=%d, n_Nbox=%zu, Nbox = ", drift_block, nbox_list.size());
  for (int nbox : nbox_list) {
    printf("%d ", nbox);
  }
  printf("\n");
}

void BoxcarWorkspace::printNboxSegment(const float* x, int n_pts, int start_offset, float scale) {
  for (int i_ofs = start_offset; i_ofs < start_offset + n_pts; i_ofs++) {
    if (i_ofs % 10 == 0) {
      printf("\n%6d   ", i_ofs);
    }
    printf("%8.0f ", x[i_ofs] * scale);
  }
  printf("\n");
}
