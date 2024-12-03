#include <assert.h>
#include <fmt/core.h>
#include <iostream>

#include "filterbank_buffer.h"
#include "taylor.h"
#include "util.h"

/*
  Performance testing the taylor tree inner loops.
 */
int main(int argc, char* argv[]) {
  const int num_timesteps = 256;
  int num_channels = 1 << 20;
  
  bool mananged_buffer = false;  // explict host & device mallocs, explict transfers
  // bool mananged_buffer = true;   // uniform memory, cudaMallocManaged

  FilterbankBuffer input(makeNoisyBuffer(num_timesteps, num_channels,mananged_buffer));
 
  FilterbankBuffer buffer1(num_timesteps, num_channels,mananged_buffer);
  FilterbankBuffer buffer2(num_timesteps, num_channels,mananged_buffer);

  for (int drift_block = -2; drift_block <= 2; ++drift_block) {
    cout << "\ndrift block " << drift_block << endl;
    long start = timeInMS();
    basicTaylorTree(input.d_sg_data, buffer1.d_sg_data, buffer2.d_sg_data,
                    num_timesteps, num_channels, drift_block);
    cudaDeviceSynchronize();
    long end = timeInMS();
    cout << fmt::format("the basic algorithm: elapsed time {:.3f}s\n",
                        (end - start) / 1000.0);

    start = timeInMS();
    optimizedTaylorTree(input.d_sg_data, buffer1.d_sg_data, buffer2.d_sg_data,
                        num_timesteps, num_channels, drift_block);

    cudaDeviceSynchronize();
    end = timeInMS();
    cout << fmt::format("optimized algorithm: elapsed time {:.3f}s\n",
                        (end - start) / 1000.0);
    
  }

}
