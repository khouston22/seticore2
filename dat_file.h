#pragma once

#include <iostream>
#include <fstream>

#include "filterbank_file.h"

using namespace std;

/*
  This class contains helper methods for the .dat file format, to maintain
  output consistency with turboseti.
*/
class DatFile {
 private:
  const FilterbankFile& metadata;
  ofstream file;
  int hit_count;
  
 public:
  DatFile(const string& filename, const FilterbankFile& metadata, double max_drift);
  ~DatFile();

  void reportHit(int coarse_channel, int freq_index, int drift_bins, double drift_rate, double snr);
};