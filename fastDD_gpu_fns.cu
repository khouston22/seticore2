
#include "fastDD_cpu.h"


__global__ void findTopPathSums2(const float* path_sums, int Nr, int num_freqs,
                                int drift_block, float* top_path_sums,
                                int* top_drift_blocks, int* top_path_offsets) {
  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }

  for (int path_offset = 0; path_offset < Nr-1; ++path_offset) {
    // // Check if the last frequency in this path is out of bounds
    // int last_freq = (Nr - 1) * drift_block + path_offset + freq;
    // if (last_freq < 0 || last_freq >= num_freqs) {
    //   // No more of these paths can be valid, either
    //   return;
    // }

    float path_sum = path_sums[num_freqs * path_offset + freq];
    if (path_sum > top_path_sums[freq]) {
      top_path_sums[freq] = path_sum;
      top_drift_blocks[freq] = drift_block;
      top_path_offsets[freq] = path_offset;
    }
  }
}


/* Simple vector add function 1: Z = X + Y */

__global__ void gpu_vector_add_xy_z(int n, float * z, float * x, float * y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] + y[i];
}

// ...
// int N = 1<<20;
// cudaMemcpy(d_x, x, N, cudaMemcpyHostToDevice);
// cudaMemcpy(d_y, y, N, cudaMemcpyHostToDevice);

// // Perform vector_add on 1M elements
// gpu_vector_add_xy_z<<<4096,256>>>(N, d_z, d_x, d_y);

// cudaMemcpy(z, d_z, N, cudaMemcpyDeviceToHost);


/* Simple vector add function X = X + Y */

__global__ void gpu_vector_add_xy_x(int n, float * x, float * y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i];
}

/* Simple vector add function 1: Z = X + Y  with a post-copy*/
        // for (kk=0; kk<n_freq_limited; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
        // }
        // for (kk=0; kk<m1; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
        // }

__global__ void gpu_vector_add_xy_z1(int n1, int n2, float * z, float * x, float * y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n1) {
    z[i] = x[i] + y[i];
  } else if (i<n2) {
    z[i] = x[i];
  }
}

/* Simple vector add function 2: Z = X + Y  with a pre-copy*/
        // for (kk=0; kk<-m1; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
        // }
        // for (kk=0; kk<n_freq_limited; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
        // }

__global__ void gpu_vector_add_xy_z2(int n1, int n2, float * z, float * x, float * y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n1) {
    z[i] = x[i];
  } else if (i<n2) {
    z[i] = x[i] + y[i];
  }
}


// ...
// int N = 1<<20;
// cudaMemcpy(d_x, x, N, cudaMemcpyHostToDevice);
// cudaMemcpy(d_y, y, N, cudaMemcpyHostToDevice);

// // Perform vector_add on 1M elements
// gpu_vector_add_xy_x<<<4096,256>>>(N, d_x, d_y);

// cudaMemcpy(x, d_x, N, cudaMemcpyDeviceToHost);

/*  ======================================================================  */
/*  These functions perform a De-Doppler sum on a spectrogram. It assumes  */
/*  the arrangement of data stream is, all points in first spectra, all     */
/*  points in second spectra, etc...  Data are summed across time           */
/*                     Original version: K. Houston 2023                    */
/*  ======================================================================  */

void fastDD_gpu_stage1(float *det_DD1, float *xx, long m_min0, long m_max0, 
               long Nt, long N0, long n_freq)
{
  // 
  //  fastDD "Slow" first stage
  // 
  // det_DD1[(Nr0*n_group)*(n_freq)]  output array
  // xx[Nt*(n_freq)]              input spectrogram (zero padded)
  // 
  
  int ig,it1,it,m,mm,mm0,k1,t_g_ofs,m_g_ofs;
  int n_group = Nt/N0;
  int Nr0 = m_max0 - m_min0 + 1;
  int *int_bin_offset;
  int xx_ofs,dd_ofs;
  int n_freq_limited;
  int grid_size = (n_freq + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  /*  ======================================================================  */

  /* calculate bin offsets */
    
  int_bin_offset = (int *) malloc((size_t)Nr0*N0*sizeof(int));
  for (mm=0; mm<Nr0; mm++) {
    m = m_min0 + mm;
    for (it=0; it<N0; it++) {
      int_bin_offset[mm*N0+it] = (int) round((double)m*it/(double)(N0));
    }
  }

  for (ig=0; ig<n_group; ig++) {
    t_g_ofs = ig*N0;
    m_g_ofs = ig*Nr0;
    for (mm=0; mm<Nr0; mm++) {
      mm0 = m_g_ofs+mm;
      for (it=0; it<N0; it++) {
        it1 = t_g_ofs+it;
        k1 = int_bin_offset[mm*N0+it];
        if (k1>=0) {
          // input SG offset equivalent to xx[mm0][k1]
          xx_ofs = it1*n_freq + k1;
          // output DD offset equivalent to det_DD1[mm0][0]
          dd_ofs = mm0*n_freq; 
        } else { // k1<0          
          // input SG offset equivalent to xx[mm0][0]
          xx_ofs = it1*n_freq;
          // output DD offset equivalent to det_DD1[mm0][-k1]
          dd_ofs = mm0*n_freq - k1; 
        }
        
        n_freq_limited = n_freq - abs(k1);
      
        if (it==0) {
          // for (kk=0; kk<n_freq_limited; kk++) {
          //   det_DD1[dd_ofs++] = xx[xx_ofs++];
          // }
          cudaMemcpy(&det_DD1[dd_ofs], &xx[xx_ofs], n_freq_limited*sizeof(float), cudaMemcpyDeviceToDevice);
          checkCuda("fastDD_gpu_stage1-Memcpy");

        } else {
          // for (kk=0; kk<n_freq_limited; kk++) {
          //   det_DD1[dd_ofs++] += xx[xx_ofs++];
          // }
          gpu_vector_add_xy_x<<<grid_size, CUDA_MAX_THREADS>>>
                        (n_freq_limited, &det_DD1[dd_ofs], &xx[xx_ofs]);
          checkCuda("fastDD_gpu_stage1-add_xy_x");
        }
      }
    }
  }
  free(int_bin_offset);
  return;
}

void fastDD_gpu_stage_i(float *det_DD_out, float *det_DD_in, int i_stage,
                   int m_min0, int m_max0,int Nt, int N0, int n_freq)
{
  // 
  //  "Fast" DeDoppler stage
  // 
  // det_DD_out[(Nr1*n_group1)*(n_freq)]  output array - stage i_stage,    Nr_ext>=Nr1*n_group1>=Nr
  // det_DD_in[(Nr0*n_group0)*(n_freq)]   input array  - stage i_stage-1,  Nr_ext =Nr0*n_group0>=Nr
  //

  int ig,m1,i1,i2;
  int m_g_ofs1a,m_g_ofs1b,m_g_ofs2;
  int dd_ofs1a,dd_ofs1b,dd_ofs2;
  int N_out = N0<<(i_stage-1);  /*   N2 = N0* 2.^(i_stage-1); */
  int N_in = N_out/2;  
  int n_group = Nt/N_out;
  int m_min_out,m_max_out,m_min_in,m_max_in;

  /*  ======================================================================  */

  m_max_out = (int) round(m_max0*N_out/N0);
  m_min_out = (int) round(m_min0*N_out/N0);
  m_max_in = (int) round(m_max0*N_in/N0);
  m_min_in = (int) round(m_min0*N_in/N0);

  int Nr_out = m_max_out - m_min_out + 1;  // number of drift rates in current stage
  int Nr_in  = m_max_in - m_min_in + 1;    // number of drift rates in previous stage
  int n_freq_limited;

  int grid_size = (n_freq + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  for (ig=0; ig<n_group; ig++) {
    m_g_ofs1a = ig*2*Nr_in;
    m_g_ofs1b = m_g_ofs1a + Nr_in;
    m_g_ofs2  = ig*Nr_out;

    for (i1=0; i1<Nr_in; i1++) {
      m1 = i1 + m_min_in;
      n_freq_limited = n_freq - abs(m1);
      
      // "even" output points
      i2 = 2*i1;

      dd_ofs2 = (m_g_ofs2+i2)*n_freq; 
      if (m1>=0) {  // positive drifts
        // apply offset of m1 frequency bins to later lines in spectrum 
        // equivalent to det_DD_in[m_g_ofs1a+i1][0] & det_DD_in[m_g_ofs1b+i1][m1]
        dd_ofs1a = (m_g_ofs1a+i1)*n_freq; 
        dd_ofs1b = (m_g_ofs1b+i1)*n_freq + m1; 
        // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][0]
        // NEW
        // for (kk=0; kk<n_freq_limited; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
        // }
        // for (kk=0; kk<m1; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
        // }
        #define NEW_GPU 1
        #if NEW_GPU
          gpu_vector_add_xy_z1<<<grid_size, CUDA_MAX_THREADS>>>
                (n_freq_limited,n_freq_limited+m1,&det_DD_out[dd_ofs2],&det_DD_in[dd_ofs1a], &det_DD_in[dd_ofs1b]);
          checkCuda("fastDD_gpu_stage_i-addz1-1");        
        #else
          gpu_vector_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>
                (n_freq_limited,&det_DD_out[dd_ofs2],&det_DD_in[dd_ofs1a], &det_DD_in[dd_ofs1b]);
          checkCuda("fastDD_gpu_stage_i-add-1");
          cudaMemcpy(&det_DD_out[dd_ofs2+n_freq_limited], &det_DD_in[dd_ofs1a+n_freq_limited], m1*sizeof(float), cudaMemcpyDeviceToDevice);
          checkCuda("fastDD_gpu_stage_i-Memcpy-1");
        #endif

      } else { // m1<0 negative drifts
        // apply offset of m1 frequency bins to earlier lines in spectrum 
        // equivalent to det_DD_in[m_g_ofs1a+i1][-m1] & det_DD_in[m_g_ofs1b+i1][0]
        dd_ofs1a = (m_g_ofs1a+i1)*n_freq; 
        dd_ofs1b = (m_g_ofs1b+i1)*n_freq; 
        // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][-m1]
        // NEW
        // for (kk=0; kk<-m1; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
        // }
        // for (kk=0; kk<n_freq_limited; kk++) {
        //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
        // }
        #if NEW_GPU
          gpu_vector_add_xy_z2<<<grid_size, CUDA_MAX_THREADS>>>
                (-m1,n_freq,&det_DD_out[dd_ofs2],&det_DD_in[dd_ofs1a], &det_DD_in[dd_ofs1b+m1]);
         checkCuda("fastDD_gpu_stage_i-addz2-2");        
        #else
          cudaMemcpy(&det_DD_out[dd_ofs2], &det_DD_in[dd_ofs1a],-m1*sizeof(float), cudaMemcpyDeviceToDevice);
          checkCuda("fastDD_gpu_stage_i-Memcpy-2");
          gpu_vector_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>
                (n_freq_limited,&det_DD_out[dd_ofs2-m1],&det_DD_in[dd_ofs1a-m1], &det_DD_in[dd_ofs1b]);
          checkCuda("fastDD_gpu_stage_i-add-2");
        #endif
      }
      
      // "odd" output points
      i2 = 2*i1+1;
      if (i2<Nr_out) {
        dd_ofs2 = (m_g_ofs2+i2)*n_freq; 
        if (m1>=0) {  // positive drifts
          // apply offset of m1 frequency bins to later lines in spectrum 
          // equivalent to det_DD_in[m_g_ofs1a+i1][0] & det_DD_in[m_g_ofs1b+i1][m1]
          dd_ofs1a = (m_g_ofs1a+i1  )*n_freq; 
          dd_ofs1b = (m_g_ofs1b+i1+1)*n_freq + m1; 
          // NEW
          // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][0]
          // for (kk=0; kk<n_freq_limited; kk++) {
          //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
          // }
          // for (kk=0; kk<m1; kk++) {
          //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
          // }
        #if NEW_GPU
          gpu_vector_add_xy_z1<<<grid_size, CUDA_MAX_THREADS>>>
                (n_freq_limited,n_freq_limited+m1,&det_DD_out[dd_ofs2],&det_DD_in[dd_ofs1a], &det_DD_in[dd_ofs1b]);
          checkCuda("fastDD_gpu_stage_i-addz1-3");        
        #else
          gpu_vector_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>
                (n_freq_limited,&det_DD_out[dd_ofs2],&det_DD_in[dd_ofs1a], &det_DD_in[dd_ofs1b]);
          checkCuda("fastDD_gpu_stage_i-add-1");
          cudaMemcpy(&det_DD_out[dd_ofs2+n_freq_limited], &det_DD_in[dd_ofs1a+n_freq_limited], m1*sizeof(float), cudaMemcpyDeviceToDevice);
          checkCuda("fastDD_gpu_stage_i-Memcpy-3");
        #endif

        } else { // m1<0 negative drifts
          // apply offset of m1 frequency bins to earlier lines in spectrum 
          // equivalent to det_DD_in[m_g_ofs1a+i1][-m1] & det_DD_in[m_g_ofs1b+i1][0]
          dd_ofs1a = (m_g_ofs1a+i1  )*n_freq; 
          dd_ofs1b = (m_g_ofs1b+i1+1)*n_freq; 
          // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][-m1]
          // NEW
          // for (kk=0; kk<-m1; kk++) {
          //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
          // }
          // for (kk=0; kk<n_freq_limited; kk++) {
          //   det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
          // }
          #if NEW_GPU
            gpu_vector_add_xy_z2<<<grid_size, CUDA_MAX_THREADS>>>
                  (-m1,n_freq,&det_DD_out[dd_ofs2],&det_DD_in[dd_ofs1a], &det_DD_in[dd_ofs1b+m1]);
          checkCuda("fastDD_gpu_stage_i-addz2-2");        
          #else
            cudaMemcpy(&det_DD_out[dd_ofs2], &det_DD_in[dd_ofs1a],-m1*sizeof(float), cudaMemcpyDeviceToDevice);
            checkCuda("fastDD_gpu_stage_i-Memcpy-2");
            gpu_vector_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>
                  (n_freq_limited,&det_DD_out[dd_ofs2-m1],&det_DD_in[dd_ofs1a-m1], &det_DD_in[dd_ofs1b]);
            checkCuda("fastDD_gpu_stage_i-add-2");
          #endif
        }
      }
    }
  }

  return;
}


float* fastDD_gpu(float *xx_sg,           // input mag squared SG (spectrogram) values [Nt_sg][Nf_sg]
                  float *det_DD_work[2],  // preallocated work arrays [Nr_ext][Nf_sg] see instructions
                  DD_metadata *dd)        // DD metadata structure with entries filled by gen_fastDD_metadata()
                                          // det_DD = det_DD_work[0] or det_DD = det_DD_work[1], depending on
                                          // number of stages
{

  // ======================================================================  */
  // This is a function to Taylor-tree-sum a spectrogram to detect linearly-drifting
  // tones using an alternative fast algorithm fastDD
  //
  // Reference:
  // K. M. Houston, "A Deep Dive into De-Doppler Algorithms for SETI", 
  // Acta Astronautica, 2023, Volume 212, Pages 505-516,
  // https://doi.org/10.1016/j.actaastro.2023.08.009
  //
  // Notes
  //
  // Reference DD_metadata struct definition for input and output parameters
  //
  // BEFORE CALLING fastDD_gpu(), one needs to:
  // 1) Call function gen_fastDD_metadata() to generate metadata
  // 2) Allocate work arrays  
  //    float *det_DD_work[2];
  //    det_DD[0] = (float *) calloc((size_t) dd->Nr*dd->Nf_sg*sizeof(float));
  //    det_DD[1] = (float *) calloc((size_t) dd->Nr*dd->Nf_sg*sizeof(float));
  //    or appropriate calloc-like function
  //    Note that work arrays should be set to zero, hence calloc() instead of malloc().
  //
  // The algorithm assumes a spectrogram to have Nt*N0 lines to be summed with appropriate drift offsets,
  // where Nt is a power of 2, and N0 is the number of lines in the "slow" first stage, e.g. N0=8.
  // In general the accuracy of the summation is improved with higher N0, but must be traded against 
  // overall computation time.
  //
  // If the actual number of SG lines Nt_sg does not generate a power-of-two Nt = Nt_sg/N0, then Nt will be
  // rounded up to the nearest power of two and the input spectrogram will be zero-padded.  
  // The net result is that Nt_sg lines will be summed.
  //
  // Usually FFT filter banks are chosen, and Lf = Lr = 1.
  //
  // If PFBs or zero-padded FFTs are used, we may opt to use Lf = Lr > 1.  Some notes:
  //
  // Define bin_bw = bin bandwidth in each pfb bin = fs = sampling rate at
  // bin output
  //
  // Input in xx_sg will have frequency oversampling factor Lf, so 
  //    df_sg = bin_bw/Lf = freq1(2) - freq1(1)
  // Output in det_DD has a corresponding frequency oversampling factor Lr, so 
  //    df_DD = bin_bw/Lr = freq2(2) - freq2(1)
  // Note Lf >= Lr, and Lf must be an integer multiple of Lr
  //
  // Output will also have a df_dt increment d_dfdt,
  // where d_dfdt = fs/(Lr*Nt*dt_sg)
  //
  // Normally dt_sg will equal 1/fs.  Sometimes the spectrogram lines will be
  // averaged N_sti times prior to input, so that dt_sg = N_sti/fs.
  // However, this  (though accommodated here),
  // because the frequency drifts over fs^2/N_sti will suffer significant
  // attenuation in the averaging process.
  // 

  int n_freq = dd->Nf_sg;
  int Nt = dd->Nt;
  int N0 = dd->N0;
  int m_min0 = dd->m_min0;
  int m_max0 = dd->m_max0;
  int n_stage = dd->n_stage;
       
  /* allocate work array for output */
  
  // float *det_DD_work[2];
  int i_in=1;
  int i_out=0;
  int i_stage=1;

  /* first stage processing - fastDD algorithm */

  fastDD_gpu_stage1(det_DD_work[0],xx_sg,m_min0,m_max0,Nt,N0,n_freq);

  // printf("GPU stage 1\n");
  // cudaDeviceSynchronize();
  // checkCuda("fastDD_gpu_stage1");
  
  /* i-th stage processing */

  for (i_stage=2; i_stage<=n_stage; i_stage++) {
    i_in = i_stage % 2;
    i_out = (i_stage+1) % 2;
    fastDD_gpu_stage_i(det_DD_work[i_out],det_DD_work[i_in],i_stage,m_min0,m_max0,
                  Nt,N0,n_freq);
    // printf("GPU stage %d\n",i_stage);
    // cudaDeviceSynchronize();
    // checkCuda("fastDD_gpu_stage_i");
  }

  // cudaDeviceSynchronize();
  // checkCuda("fastDD_gpu_end");

  return det_DD_work[i_out];
  }
