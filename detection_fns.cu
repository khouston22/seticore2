
#include "detection_fns.h"

void calc_subband_mean_std(const float* x_sg, int Nf, int n_subband, bool do_limit, 
                  float *subband_limit, float *work, float *subband_mean, float *subband_std) 
{
  int Nf_subband = Nf/n_subband;
  
  for (int i_band=0; i_band<n_subband; i_band++) {
    int i_ofs = i_band*Nf_subband;
    if (do_limit) {
      float limit_value = subband_limit[i_band];
      for (int i=0;i<Nf_subband; i++) {
        work[i] = MIN(x_sg[i_ofs+i],limit_value);
      }
      calc_mean_std_dev(work, Nf_subband, &subband_mean[i_band], &subband_std[i_band]);
    } else {
      // do not apply shear operation
      calc_mean_std_dev(&x_sg[i_ofs], Nf_subband, &subband_mean[i_band], &subband_std[i_band]);
    }
  }
}

void multipass_subband_mean_std(const float* x_sg, int Nf, int n_subband, float shear_constant, 
                      float *work, float *subband_mean, float *subband_std, float *subband_limit) 
{
  // first pass subband stats

  bool do_limit = false;
  calc_subband_mean_std(x_sg,Nf,n_subband,do_limit,subband_limit,work,subband_mean,subband_std);

  // calculate shear threshold

  for (int i_band=0; i_band<n_subband; i_band++) {
    subband_limit[i_band] = subband_mean[i_band] + shear_constant*subband_std[i_band];
  }
  
  // second pass subband stats
  do_limit = true;
  calc_subband_mean_std(x_sg,Nf,n_subband,do_limit,subband_limit,work,subband_mean,subband_std);
  
  // calculate shear threshold again

  for (int i_band=0; i_band<n_subband; i_band++) {
    subband_limit[i_band] = subband_mean[i_band] + shear_constant*subband_std[i_band];
  }
  
  // third pass subband stats
  do_limit = true;
  calc_subband_mean_std(x_sg,Nf,n_subband,do_limit,subband_limit,work,subband_mean,subband_std);

}

/* linearly interpolate subband mean or std values to full values over all freqs */

__global__ void gpu_subband_interpolate(float* x, int n_freq, float* x_subband, int n_subband)
{
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < 0 || i_freq >= n_freq) {
    return;
  }

  int nf_subband = n_freq/n_subband;
  int i_subband = i_freq/nf_subband;
  int df = i_freq - nf_subband*i_subband - nf_subband/2;
  float scale;

  if (df>=0) {
    if (i_subband < n_subband-1) {
      scale = (x_subband[i_subband+1] - x_subband[i_subband])/nf_subband;
    } else {
      scale = 0.;   // Don't extrapolate at uppermost subband right edge
    }
  } else {
    if (i_subband > 0) {
      scale = (x_subband[i_subband] - x_subband[i_subband-1])/nf_subband;
    } else {
      scale = 0.;   // Don't extrapolate at lowermost subband left edge
    }
  }
  x[i_freq] = x_subband[i_subband] + df*scale;
}


/* 
scale a chi-square spectrum line to unit mean 
x[f] = x[f]/mu[f]
*/

__global__ void gpu_local_mean_scale(float* x, float* mu, int n_freq)
{
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < 0 || i_freq >= n_freq) {
    return;
  }

  x[i_freq] = (x[i_freq]/mu[i_freq]);
}

/* 
sigma_scale[f] = Nbox_gain/sigma[f]
*/

__global__ void gpu_compute_sigma_scale(float* sigma_scale, float* sigma, float Nbox_gain, int n_freq)
{
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < 0 || i_freq >= n_freq) {
    return;
  }

  sigma_scale[i_freq] = (Nbox_gain/sigma[i_freq]);
}

void calc_mean_std_dev(const float* x, int n, float *mean, float *std_dev) 
{
  double sum_x2 = 0.;
  double sum_x = 0.;

  for (int i = 0; i < n; i++) {
    float temp = x[i];
    sum_x += temp;
    sum_x2 += temp*temp;
  }
  *mean = sum_x/n;
  *std_dev = sqrt((sum_x2 - n*(*mean)*(*mean))/(n-1));
}

void calc_mean_std_dev2(const float* x, int n, float *mean, float *std_dev) 
{
  // reduced precision calculation adequate for some DSP applications with noisy signals
  float sum_x2 = 0.;
  float sum_x = 0.;

  for (int i = 0; i < n; i++) {
    float temp = x[i];
    sum_x += temp;
    sum_x2 += temp*temp;
  }
  *mean = sum_x/n;
  *std_dev = sqrt((sum_x2 - n*(*mean)*(*mean))/(n-1));
}

float find_max(const float* x, int n) 
{
  float x_max = x[0];
  for (int i = 1; i < n; i++) {
    x_max = MAX(x_max,x[i]);
  }
  return x_max;
}

float find_min(const float* x, int n) 
{
  float x_min = x[0];
  for (int i = 1; i < n; i++) {
    x_min = MIN(x_min,x[i]);
  }
  return x_min;
}

void DC_replace(float* x, int DC_replace_ofs, int DC_mean_pts) 
{
    // remove DC bins - replace by adjacent mean
    // x is part of a vector of PSD values pointing to DC (mid) point
    // call: DC_replace(&x[mid],DC_replace_ofs,DC_mean_pts);

    float adj_mean = 0.;
    for (int i_ofs=-DC_replace_ofs-DC_mean_pts; i_ofs<-DC_replace_ofs; i_ofs++) {
      adj_mean += x[i_ofs];
    }

    for (int i_ofs=DC_replace_ofs+1; i_ofs<=DC_replace_ofs+DC_mean_pts; i_ofs++) {
      adj_mean += x[i_ofs];
    }

    adj_mean /= (2*DC_mean_pts);

    for (int i_ofs=-DC_replace_ofs; i_ofs<=DC_replace_ofs; i_ofs++) {
      x[i_ofs]=adj_mean;
    }
} 

void print_x_lr(float* x, int max_ofs, float scale) 
{
  // view part of vector from both sides (left and right of center)
  // call: print_x_lr(&x[center],max_ofs,scale);
    
  for (int i_ofs=-max_ofs; i_ofs<max_ofs; i_ofs++) {
    if (i_ofs%10==0) printf("\n%6d   ",i_ofs);
    printf("%8.0f ",x[i_ofs]*scale);
  }
  if (max_ofs%10==0) printf("\n"); else printf("\n\n");
} 

void print_x_segment(float* x, int n_pts, float scale) 
{
  // view vector segment
  // call: print_x_segment(&x[start],n_pts,scale);
    
  for (int i_ofs=0; i_ofs<n_pts; i_ofs++) {
    if (i_ofs%10==0) printf("\n%6d   ",i_ofs);
    printf("%8.0f ",x[i_ofs]*scale);
  }
  if (n_pts%10==0) printf("\n"); else printf("\n\n");
} 

void print_x_segment_stride(float* x, int n_pts, int stride, float scale) 
{
  // view vector segment with strided index
  // call: print_x_segment_stride(&x[start],n_pts,stride,scale);
    
  for (int i_ofs=0; i_ofs<n_pts; i_ofs++) {
    if (i_ofs%10==0) printf("\n%6d   ",i_ofs*stride);
    printf("%8.0f ",x[i_ofs*stride]*scale);
  }
  if (n_pts%10==0) printf("\n"); else printf("\n\n");
} 

void print_f_x_segment(float* x, int n_pts, float scale, float f0, float df) 
{
  // view vector segment
  // call: print_x_segment(&x[start],n_pts,scale);
    
  for (int i_ofs=0; i_ofs<n_pts; i_ofs++) {
    if (i_ofs%10==0) printf("\n%6d %8.2f  ",i_ofs,f0+i_ofs*df);
    printf("%8.0f ",x[i_ofs]*scale);
  }
  if (n_pts%10==0) printf("\n"); else printf("\n\n");
} 

