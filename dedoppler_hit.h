#pragma once

#include <string>

#include "filterbank_metadata.h"

using namespace std;

const int NO_BEAM = -1;

class DedopplerHit {
public:
  // Which frequency bin the hit starts at, within the coarse channel, at t=0
  int index;

  // Relevant frequencies (chosen to be consistent with turbo_seti dat files)
  double freq_MHz_ctr;    // center frequency in MHz = (freq_MHz1+freq_MHz2)/2
  double freq_MHz1;       // chirp start freq (at time 0) 
  double freq_MHz2;       // chirp end freq (at end of obs time)
  // TODO: resolve whether freq_MHz2 > freq_MHz1 always in dat file (FreqStart and FreqEnd)
  // double freq_MHz1;       // Lower limit of chirp = freq_MHz_ctr - abs(drift_rate)/2 
  // double freq_MHz2;       // Upper limit of chirp = freq_MHz_ctr + abs(drift_rate)/2
  
  // How many bins the hit drifts over.
  // Like (ending index - starting index), this is positive for rightward drift,
  // negative for leftward drift.
  // This is zero for a vertical line.
  // Includes drift over the full rounded-up power-of-two time range.
  int drift_steps;

  // The drift rate in Hz/s
  double drift_rate;

  // The signal-to-noise ratio for the hit
  float snr;

  // Which coarse channel the hit is in.
  int coarse_channel;

  // Which beam the hit is in. NO_BEAM if there is none, or for the incoherent beam.
  int beam;

  // This does *not* use rounded-up-to-a-power-of-two timesteps.
  int num_timesteps;

  // The total power used in the numerator to calculate snr.
  float power;

  // The total power in the incoherent beam, calculated along the same line.
  float incoherent_power;
  
  DedopplerHit(const FilterbankMetadata& metadata, int _index, 
              double _freq_MHz_ctr, double _freq_MHz1, double _freq_MHz2,
              int _drift_steps, double _drift_rate, float _snr, int _beam, int _coarse_channel,
              int _num_timesteps, float _power);

  string toString() const;

  // Lowest index that contains a bin with this signal
  int lowIndex() const;

  // Highest index that contains a bin with this signal
  int highIndex() const;

  // The index within the coarse channel that you should expect if you extrapolate
  // this hit in time.
  // Time is measured in "number of timesteps since the start of this hit".
  int expectedIndex(int timesteps) const;
  
  // The highest-scoring hits are the ones that get turned into stamps
  float score() const;
};


bool operator<(const DedopplerHit& lhs, const DedopplerHit& rhs);

bool driftStepsLessThan(const DedopplerHit& lhs, const DedopplerHit& rhs);


