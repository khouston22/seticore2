
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

/*
  Sum the columns of a two-dimensional array.
  input is a (num_timesteps x num_freqs) array, stored in row-major order.
  sums is an array of size num_freqs.
 */
void sumColumns_cpu(const float* input, float* sums, int num_timesteps, int n_freq) 
{
  int in_ofs;

  for (int time = 0; time < num_timesteps; time++) {
    if (time==0) {
      for (int freq = 0; freq < n_freq; freq++) {
        sums[freq] = input[freq];
      }
    } else {
      in_ofs = time*n_freq;
      for (int freq = 0; freq < n_freq; freq++) {
        sums[freq] += input[in_ofs++];
      }
    }
  }
}


void ncoh_avg_test(const float* x, int Nf, int Nt, int n_sti, int n_subband)
{
  float *cpu_column_sums;
  float *subband_mean,*subband_std,*subband_limit;
  float *work;
  int Nf_subband = Nf/n_subband;
  float *subband_m_std_ratio,*subband_std2,*subband_std2_ratio;

  cpu_column_sums = (float *) malloc(Nf*sizeof(float));
  subband_mean = (float *) malloc(n_subband*sizeof(float));
  subband_std  = (float *) malloc(n_subband*sizeof(float));
  subband_limit = (float *) malloc(n_subband*sizeof(float));
  work = (float *) malloc(Nf_subband*sizeof(float));
  subband_m_std_ratio  = (float *) malloc(n_subband*sizeof(float));
  subband_std2 = (float *) malloc(n_subband*sizeof(float));
  subband_std2_ratio = (float *) malloc(n_subband*sizeof(float));
  
  printf("\nNf=%d,n_subband=%d:\n",Nf,n_subband);
  for (int n_lti=2; n_lti<= Nt; n_lti*=2) {
    int n_avg = n_sti*n_lti;

    printf("n_sti=%2d, n_lti=%4d, n_avg=%4d ",n_sti,n_lti,n_avg);

    sumColumns_cpu(x, cpu_column_sums, n_lti, Nf) ;

    int mid = Nf / 2;
    cpu_column_sums[mid-1]=0.;
    cpu_column_sums[mid]=0.;
    cpu_column_sums[mid+1]=0.;

    if (n_subband==1){
      calc_mean_std_dev(cpu_column_sums,Nf,subband_mean,subband_std);

      // bool do_limit = false;
      // calc_subband_mean_std(cpu_column_sums,Nf,n_subband,do_limit,subband_limit,work,subband_mean,subband_std);
    } else {
      float shear_constant = 2.3;
      multipass_subband_mean_std(cpu_column_sums,Nf,n_subband,shear_constant,
                  work,subband_mean,subband_std,subband_limit);
    }
    float xf = 1./n_avg/2.;
    for (int i_band=0; i_band<n_subband; i_band++){
      subband_mean[i_band]  = xf*subband_mean[i_band];
      subband_std[i_band]   = xf*subband_std[i_band];
      subband_limit[i_band] = xf*subband_limit[i_band];
      subband_m_std_ratio[i_band] = subband_mean[i_band]/subband_std[i_band];
      if (n_lti==2) subband_std2[i_band] = subband_std[i_band];
      subband_std2_ratio[i_band] = subband_std2[i_band]/subband_std[i_band];
    }
    float m_std_ratio_mean,m_std_ratio_std;
    calc_mean_std_dev(subband_m_std_ratio,n_subband,&m_std_ratio_mean,&m_std_ratio_std);
    float std2_ratio_mean,std2_ratio_std;
    calc_mean_std_dev(subband_std2_ratio,n_subband,&std2_ratio_mean,&std2_ratio_std);
    float m = subband_mean[n_subband/2];
    float std_dev = subband_std[n_subband/2];
  
    printf("mean=%6.3f std_dev=%6.3f mean/std=%6.3f vs. %6.3f std2/std=%6.3f vs. %6.3f\n",
            m,std_dev,m_std_ratio_mean,sqrt(2*n_avg),std2_ratio_mean,sqrt(n_lti/2));
    
  }
  free(cpu_column_sums);
  free(subband_mean);
  free(subband_std);
  free(subband_limit);
  free(work);
  free(subband_std2);
  free(subband_std2_ratio);
  free(subband_m_std_ratio);
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
  printf("\n");
} 

void print_x_segment(float* x, int n_pts, float scale) 
{
  // view vector segment
  // call: print_x_segment(&x[start],n_pts,scale);
    
  for (int i_ofs=0; i_ofs<n_pts; i_ofs++) {
    if (i_ofs%10==0) printf("\n%6d   ",i_ofs);
    printf("%8.0f ",x[i_ofs]*scale);
  }
  printf("\n");
} 

