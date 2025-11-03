
#include "boxcar_sum.h"
#include <stdio.h> 

using namespace std;

#define TEST_GPU 1

int main() {
  printf("Boxcar test\n"); // Prints a message to the console

  int n_freq = 1 << 12;
  int log2_max_p2 = 4;
  int Nbox_p2_max = 1 << log2_max_p2;
  int Nbox_max = 2*Nbox_p2_max - 1;
  int n_zp = 2*Nbox_p2_max;
  int n_freq_ext = n_freq + 2*n_zp;
  int n_p2 = log2_max_p2 + 1;

  printf("n_freq=%d, log2_max_p2=%d, Nbox_p2_max=%d, Nbox_max=%d, n_zp=%d\n",
          n_freq,log2_max_p2,Nbox_p2_max,Nbox_max,n_zp);

  float *DD_sums_line;
  float *p2_path_sums;
  float *work;
  float *Nbox_path_sum;
 
  #if TEST_GPU
    float *gpu_DD_sums_line, *gpu_p2_path_sums, *gpu_work, *gpu_Nbox_path_sum;
    cudaMalloc(&gpu_DD_sums_line, n_freq*sizeof(float));
    cudaMallocHost(&DD_sums_line, n_freq*sizeof(float));
    checkCuda("DD_sums_line malloc");
    cudaMalloc(&gpu_p2_path_sums, n_freq_ext*n_p2*sizeof(float));
    cudaMallocHost(&p2_path_sums, n_freq_ext*n_p2*sizeof(float));
    checkCuda("p2_path_sums malloc");
    cudaMalloc(&gpu_work, 2*n_freq_ext*sizeof(float));
    cudaMallocHost(&work, 2*n_freq_ext*sizeof(float));
    checkCuda("work malloc");
    cudaMalloc(&gpu_Nbox_path_sum, n_freq*sizeof(float));
    cudaMallocHost(&Nbox_path_sum, n_freq*sizeof(float));
    checkCuda("Nbox_path_sum malloc");
    cudaMemsetAsync(gpu_work, 0, 2*n_freq_ext*sizeof(float));
  #else
    DD_sums_line = (float *) malloc(n_freq*sizeof(float));
    p2_path_sums = (float *) malloc(n_freq_ext*n_p2*sizeof(float));
    work = (float *) malloc(2*n_freq_ext*sizeof(float));
    Nbox_path_sum = (float *) malloc(n_freq*sizeof(float));
  #endif

  // set up input line in cpu
  memset(DD_sums_line, 0, n_freq*sizeof(float)); 
  
  int i_freq;
  int sig_start = n_freq/2;
  // int sig_start = 0;
  // int sig_start = n_freq - sig_width;
  float sig_value = 1.0;

  #if 1
    int sig_width = 1;
    for (i_freq=sig_start; i_freq<sig_start+sig_width; i_freq++) {
      DD_sums_line[i_freq] = sig_value;
    }
  #else
    int sig_width = 5;
    for (i_freq=sig_start; i_freq<sig_start+sig_width; i_freq++) {
      DD_sums_line[i_freq] = sig_value + i_freq - sig_start;
    }
  #endif

  #if TEST_GPU
    // copy DD_sums_line to GPU
    cudaMemcpy(gpu_DD_sums_line,DD_sums_line,
               n_freq*sizeof(float), cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-DD_sums_line");
  #endif

  
  // Generate power of 2 sum vectors

  #if TEST_GPU
    #if 1
      gen_boxcar_p2_sums_gpu(gpu_DD_sums_line,gpu_p2_path_sums,n_freq,log2_max_p2,n_zp);
      cudaMemcpy(p2_path_sums,gpu_p2_path_sums,
                 n_freq_ext*n_p2*sizeof(float), cudaMemcpyDeviceToHost);
      checkCuda("cudaMemcpy-gen_boxcar_p2_sums_gpu");
    #else
      // still compute in cpu
      gen_boxcar_p2_sums_cpu( DD_sums_line,p2_path_sums,n_freq,log2_max_p2,n_zp);
    #endif
  #else
    gen_boxcar_p2_sums_cpu( DD_sums_line,p2_path_sums,n_freq,log2_max_p2,n_zp);
  #endif

  // Print out power of 2 boxcar sum vectors
  int print_ofs = 20;
  int print_n_pts = 60;

  for (int i_Nbox=0; i_Nbox<=log2_max_p2; i_Nbox++) {
    int start_idx = n_freq_ext*i_Nbox + sig_start + n_zp - print_ofs;
    int Nbox_p2 = 1 << i_Nbox;
    printf("\nNbox_p2=%d, sig_start=%d %d\n",Nbox_p2,sig_start,n_freq-sig_start);
    int n_pts = MIN(print_n_pts,n_freq_ext - (sig_start - print_ofs));
    print_Nbox_segment(&p2_path_sums[start_idx],n_pts,1.);
  }

  // Generate boxcar sums for arbitrary Nbox values

  for (int Nbox=1; Nbox<=Nbox_max; Nbox++) {
    #if TEST_GPU
      gen_boxcar_sum_gpu(gpu_p2_path_sums,gpu_work,gpu_Nbox_path_sum,Nbox,n_freq,log2_max_p2,n_zp);
      cudaMemcpy(Nbox_path_sum,gpu_Nbox_path_sum,
                 n_freq*sizeof(float), cudaMemcpyDeviceToHost);
      checkCuda("cudaMemcpy-gen_boxcar_sum_gpu");
    #else
      gen_boxcar_sum_cpu(p2_path_sums,work,Nbox_path_sum,Nbox,n_freq,log2_max_p2,n_zp);
    #endif

    // Print out boxcar sum vector

    int start_idx;
  
    // start_idx = sig_start + n_zp - print_ofs;
    // printf("\nWork 0 Nbox=%d\n",Nbox);
    // print_Nbox_segment(&work[start_idx],print_n_pts,1.);

    // start_idx = n_freq_ext + sig_start + n_zp - print_ofs;
    // printf("\nWork 1 Nbox=%d\n",Nbox);
    // print_Nbox_segment(&work[start_idx],print_n_pts,1.);

    start_idx = sig_start - print_ofs;
    printf("\nNbox=%d, sig_start=%d %d\n",Nbox,sig_start,n_freq-sig_start);
  
    int n_pts = MIN(print_n_pts,n_freq - start_idx);
    print_Nbox_segment(&Nbox_path_sum[start_idx],n_pts,Nbox);
  }

  #if TEST_GPU
    cudaFree(gpu_DD_sums_line);
    cudaFreeHost(DD_sums_line);
    cudaFree(gpu_p2_path_sums);
    cudaFreeHost(p2_path_sums);
    cudaFree(gpu_work);
    cudaFreeHost(work);
    cudaFree(gpu_Nbox_path_sum);
    cudaFreeHost(Nbox_path_sum);
  #else
    free(DD_sums_line);
    free(p2_path_sums);
    free(work);
    free(Nbox_path_sum);
  #endif

  return 0;
}