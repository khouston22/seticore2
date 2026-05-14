#include "dedoppler_hit.h"

#include <assert.h>
#include <fmt/core.h>
#include <string>

#include "util.h"

using namespace std;

DedopplerHit::DedopplerHit(const FilterbankMetadata& metadata, int _index, 
              double _freq_MHz_ctr, double _freq_MHz1, double _freq_MHz2,
              int _drift_steps, double _drift_rate, float _snr, int _beam, int _coarse_channel,
              int _num_timesteps, float _power, float _hit_BlkSK, float _hit_SK, float _hit_max_min)
     :  index(_index), freq_MHz_ctr(_freq_MHz_ctr), 
        freq_MHz1(_freq_MHz1), freq_MHz2(_freq_MHz2), 
        drift_steps(_drift_steps), drift_rate(_drift_rate),
        snr(_snr), coarse_channel(_coarse_channel),
        beam(metadata.isCoherentBeam(_beam) ? _beam : NO_BEAM),
        num_timesteps(_num_timesteps), power(_power), 
        hit_BlkSK(_hit_BlkSK), hit_SK(_hit_SK), hit_max_min(_hit_max_min),
        incoherent_power(0.0) {
}
           

string DedopplerHit::toString() const {
  return fmt::format("coarse channel = {}, index = {}, snr = {:.5f}, "
                     "drift rate = {:.5f} ({})",
                     coarse_channel, index, snr, drift_rate,
                     pluralize(drift_steps, "bin"));
}

int DedopplerHit::lowIndex() const {
  return min(index, index + drift_steps);
}

int DedopplerHit::highIndex() const {
  return max(index, index + drift_steps);
}

int DedopplerHit::expectedIndex(int timesteps) const {
  double drift_steps_per_timestep = ((double) (drift_steps)) /
    ((double) (num_timesteps - 1));
  return index + (int) round(timesteps * drift_steps_per_timestep);
}

// If we have incoherent power, we want to use the ratio of power to incoherent power.
// Otherwise, we just want to sort by SNR.
float DedopplerHit::score() const {
  if (incoherent_power > 0) {
    return power / incoherent_power;
  } else {
    assert(snr >= 0.0);
    return -1.0 / (1.0 + snr);
  }
}

// Sort first by coarse channel, then by low index, then by high index
bool operator<(const DedopplerHit& lhs, const DedopplerHit& rhs) {
  if (lhs.coarse_channel != rhs.coarse_channel) {
    return lhs.coarse_channel < rhs.coarse_channel;
  }
  if (lhs.lowIndex() != rhs.lowIndex()) {
    return lhs.lowIndex() < rhs.lowIndex();
  }
  return lhs.highIndex() < rhs.highIndex();
}


// Alternate sort comparator that just compares drift steps
bool driftStepsLessThan(const DedopplerHit& lhs, const DedopplerHit& rhs) {
  return lhs.drift_steps < rhs.drift_steps;
}

