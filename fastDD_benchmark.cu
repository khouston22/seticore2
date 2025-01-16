#include <assert.h>
#include <fmt/core.h>
#include <iostream>

#include "filterbank_buffer.h"
#include "taylor.h"
#include "util.h"

#include "fastDD_cpu.h"

/*
  Performance testing the taylor tree inner loops.
  ./fastDD_eval1 [exp_n Nt N0]
 */
int main(int argc, char* argv[]) {
  int exp_n = 20;
  int Nt = 256;
  int N0 = 8;
  
  if (argc>=4) {
    N0 = atoi(argv[3]);
  }
  if (argc>=3) {
    Nt = atoi(argv[2]);
  }
  if (argc>=2) {
    exp_n = atoi(argv[1]);
  }
  printf("\nexp=%d, %dK, Nt=%d, N0=%d\n",exp_n,1 << (exp_n-10),Nt,N0);
  
  //const int num_timesteps = 256;
  //const int num_timesteps = 128;
  //int num_channels = 1 << 21;  // 2048K
  //int num_channels = 1 << 20;  // 1024K
  //int num_channels = 1 << 19;  // 512K
  //int num_channels = 1 << 18;  // 256K
  int num_timesteps = Nt;
  int num_channels = 1 << exp_n;

# define DO_TAYLOR 0
# define DO_FASTDD_GPU 1
# define _MANAGED_INPUT 0

#if DO_TAYLOR  

  float *d_sg;
  float *buffer1, *buffer2;
  #if _MANAGED_INPUT
    cout << fmt::format("Taylor DD Evaluation, CUDA-Managed Input Buffer\n");
    FilterbankBuffer input(makeNoisyBuffer(num_timesteps, num_channels,true));
    FilterbankBuffer _buffer1(num_timesteps, num_channels);
    FilterbankBuffer _buffer2(num_timesteps, num_channels);
    d_sg = input.sg_data;
    buffer1 = _buffer1.sg_data;
    buffer2 = _buffer2.sg_data;
  #else
    cout << fmt::format("Taylor DD Evaluation, Unmanaged Input Buffer\n");

    FilterbankBuffer input(makeNoisyBuffer(num_timesteps, num_channels,false));
  
    d_sg = input.d_sg_data;
    cudaMemcpy(input.d_sg_data,input.sg_data,input.bytes,cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-d_sg");
  
    cudaMalloc(&buffer1, num_channels * num_timesteps * sizeof(float));
    checkCuda("Dedopplerer buffer1 malloc");
    cudaMalloc(&buffer2, num_channels * num_timesteps * sizeof(float));
    checkCuda("Dedopplerer buffer2 malloc");
  #endif

  for (int drift_block = -2; drift_block <= 2; ++drift_block) {
    cout << "\ndrift block " << drift_block << endl;
    long start = timeInMS();
    basicTaylorTree(d_sg, buffer1, buffer2,
                    num_timesteps, num_channels, drift_block);
    cudaDeviceSynchronize();
    long end = timeInMS();
    cout << fmt::format("the basic algorithm: elapsed time {:.3f}s\n",
                        (end - start) / 1000.0);

    start = timeInMS();
    optimizedTaylorTree(d_sg, buffer1, buffer2,
                        num_timesteps, num_channels, drift_block);

    cudaDeviceSynchronize();
    end = timeInMS();
    cout << fmt::format("optimized algorithm: elapsed time {:.3f}s\n",
                        (end - start) / 1000.0);
    
  }

  cudaFree(buffer1);
  cudaFree(buffer2);
  
#elif DO_FASTDD_GPU 
  DD_metadata *DD_meta,DD_meta1;
  DD_meta = &DD_meta1;

  double f_sg_min= 0.;       // SG start freq (bin center) Hz
  //double df_sg = foff;        // SG freq increment Hz
  double df_sg = 1.0;        // SG freq increment Hz
  int Nf_sg = num_channels;  // SG number of frequency values
  //double dt_sg = tsamp;       // SG time increment per line (time resolution, lines might be averaged)
  double dt_sg = 1.0;         // SG time increment per line (time resolution, lines might be averaged)
  int Nt_sg = num_timesteps; // SG number of lines (time values)
  double dfdt_min_nom =  0.0; // desired DD minimum frequency rate Hz/sec (input)
  double dfdt_max_nom =  1.0; // desired DD maximum frequency rate Hz/sec (input)
  double Lf = 1.0;            // PFB overlap factor (=1 for FFT filter bank) 
  //int Nt = num_timesteps;    // total number of time samples to integrate (Nt/N0 is power of 2)   
  //int N0 = MIN(8,num_timesteps);  // first stage number of time samples   
  
  gen_fastDD_metadata(DD_meta,f_sg_min,df_sg,Nf_sg,dt_sg,Nt_sg,dfdt_min_nom,dfdt_max_nom,Lf,Nt,N0);

  long start_ms,end_ms;

  float *gpu_det_DD;
  float *det_DD_work[2];

  #if _MANAGED_INPUT
    cout << fmt::format("fastDD GPU Evaluation, CUDA-Managed Input Buffer\n");
    FilterbankBuffer input(makeNoisyBuffer(num_timesteps, num_channels,true));
    FilterbankBuffer buffer1(DD_meta->Nr_ext, num_channels,true);
    FilterbankBuffer buffer2(DD_meta->Nr_ext, num_channels,true);
    det_DD_work[0] = (float *) buffer1.sg_data;
    det_DD_work[1] = (float *) buffer2.sg_data;
  #else
    cout << fmt::format("fastDD GPU Evaluation, Unmanaged Input Buffer\n");
    int n_byte_work = Nf_sg*DD_meta->Nr_ext*sizeof(float);
    float *d_work1,*d_work2;
    cudaMalloc(&d_work1, n_byte_work);
    cudaMalloc(&d_work2, n_byte_work);
    det_DD_work[0] = d_work1;
    det_DD_work[1] = d_work2;
    FilterbankBuffer input(makeNoisyBuffer(num_timesteps, num_channels,false));
    cudaMemcpy(input.d_sg_data,input.sg_data,input.bytes,cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-d_sg");
  #endif

  for (int drift_block = -2; drift_block <= 2; ++drift_block) {

    dfdt_min_nom =  drift_block*1.0;     // desired DD minimum frequency rate Hz/sec (input)
    dfdt_max_nom =  (drift_block+1)*1.0; // desired DD maximum frequency rate Hz/sec (input)

    gen_fastDD_metadata(DD_meta,f_sg_min,df_sg,Nf_sg,dt_sg,Nt_sg,dfdt_min_nom,dfdt_max_nom,Lf,Nt,N0);

    cout << "\ndrift block " << drift_block << endl;
    start_ms = timeInMS();

    #if _MANAGED_INPUT
      gpu_det_DD = fastDD_gpu(input.sg_data,det_DD_work,DD_meta);
    #else
      gpu_det_DD = fastDD_gpu(input.d_sg_data,det_DD_work,DD_meta);
    #endif

    cudaDeviceSynchronize();
    
    end_ms = timeInMS();

    printf("N0=%d, fast_DD_gpu complete, %.3f sec\n",N0,(end_ms-start_ms)*.001);
  }
  printf("gpu_det_DD pointer value =%.2f MB\n",((long) gpu_det_DD)/1024./1024.);

  #if _MANAGED_INPUT
  #else
    cudaFree(d_work1);
    cudaFree(d_work2);
  #endif
 
#else   // fast_DD_cpu case
  DD_metadata *DD_meta,DD_meta1;
  DD_meta = &DD_meta1;
  
  double f_sg_min= 1e6;       // SG start freq (bin center) Hz
  double df_sg = 1.0;         // SG freq increment Hz
  int Nf_sg = num_channels;  // SG number of frequency values
  double dt_sg = 1.0;         // SG time increment per line (time resolution, lines might be averaged)
  int Nt_sg = num_timesteps; // SG number of lines (time values)
  int drift_block = 0;
  double dfdt_min_nom =  drift_block*1.0;     // desired DD minimum frequency rate Hz/sec (input)
  double dfdt_max_nom =  (drift_block+1)*1.0; // desired DD maximum frequency rate Hz/sec (input)
  double Lf = 1.0;            // PFB overlap factor (=1 for FFT filter bank) 
  //int Nt = num_timesteps;    // total number of time samples to integrate (Nt/N0 is power of 2)   
  //int N0 = 8;                // first stage number of time samples   

  gen_fastDD_metadata(DD_meta,f_sg_min,df_sg,Nf_sg,dt_sg,Nt_sg,dfdt_min_nom,dfdt_max_nom,Lf,Nt,N0);

  print_fastDD_metadata(DD_meta);

  float *det_DD;
  float *det_DD_work[2];
          
  #if _MANAGED_INPUT
    cout << fmt::format("fastDD CPU Evaluation, CUDA-Managed Input Buffer\n");
    FilterbankBuffer input(makeNoisyBuffer(num_timesteps, num_channels,true));
  #else
    cout << fmt::format("fastDD CPU Evaluation, Unmanaged Input Buffer\n");
    float *sg_data;
    sg_data = (float *) calloc(DD_meta->Nt_sg*DD_meta->Nf_sg,sizeof(float));
  #endif

  det_DD_work[0] = (float *) calloc(DD_meta->Nr_ext*DD_meta->Nf_sg,sizeof(float));
  det_DD_work[1] = (float *) calloc(DD_meta->Nr_ext*DD_meta->Nf_sg,sizeof(float));

  // float *cpu_column_sums;
  // float *cpu_top_path_sums;
  // int *cpu_top_drift_blocks;
  // int *cpu_top_path_offsets;

  // cpu_column_sums = (float *) calloc(DD_meta->Nf_sg,sizeof(float));
  // cpu_top_path_sums = (float *) calloc(DD_meta->Nf_sg,sizeof(float));
  // cpu_top_drift_blocks = (int *) calloc(DD_meta->Nf_sg,sizeof(int));
  // cpu_top_path_offsets = (int *) calloc(DD_meta->Nf_sg,sizeof(int));
  // cpu_top_path_offsets = (int *) calloc(DD_meta->Nf_sg,sizeof(int));

  double work_size_MB = DD_meta->Nr_ext*DD_meta->Nf_sg*sizeof(float)/1024./1024.;

  printf("\n Two det_DD_work arrays %d x %d allocated, each %.0f MBytes\n",DD_meta->Nr_ext,DD_meta->Nf_sg,
          work_size_MB);
  
  /* run fastDD */

  long start_ms,end_ms;

  for (int drift_block = -2; drift_block <= 2; ++drift_block) {

    dfdt_min_nom =  drift_block*1.0;     // desired DD minimum frequency rate Hz/sec (input)
    dfdt_max_nom =  (drift_block+1)*1.0; // desired DD maximum frequency rate Hz/sec (input)

    gen_fastDD_metadata(DD_meta,f_sg_min,df_sg,Nf_sg,dt_sg,Nt_sg,dfdt_min_nom,dfdt_max_nom,Lf,Nt,N0);

    cout << "\ndrift block " << drift_block << endl;
    //print_fastDD_metadata(DD_meta);

    start_ms = timeInMS();

    #if _MANAGED_INPUT
      det_DD = fastDD_cpu(input.sg_data,det_DD_work,DD_meta);
    #else
      det_DD = fastDD_cpu(sg_data,det_DD_work,DD_meta);
    #endif

    end_ms = timeInMS();

    printf("N0=%d, fast_DD_cpu complete, %.3f sec\n",N0,(end_ms-start_ms)*.001);

  }
                              
  printf("det_DD[0]=%.2f\n",det_DD[0]);

  free(det_DD_work[0]);
  free(det_DD_work[1]);

  #if _MANAGED_INPUT
  #else
    free(sg_data);
  #endif  

  // free(cpu_column_sums);
  // free(cpu_top_path_sums);
  // free(cpu_top_drift_blocks);
  // free(cpu_top_path_offsets);
#endif
}
