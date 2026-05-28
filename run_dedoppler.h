#pragma once

#include <string>

using namespace std;

void runDedoppler(const string& input_filename, const string& output_filename,
                  double max_drift, double min_drift, double snr_threshold,
                  bool do_hit_screen, bool write_BB_hits_to_dat, int debug);

