
#include "fastDD_cpu.h"

void gen_fastDD_metadata( DD_metadata *dd,      // DD metadata structure with entries filled 
                          double f_sg_min,      // SG start freq (bin center) Hz
                          double df_sg,         // SG freq increment Hz
                          int Nf_sg,           // SG number of frequency values
                          double dt_sg,         // SG time increment per line (time resolution, lines might be averaged)
                          int Nt_sg,           // SG number of lines (time values)
                          double dfdt_min_nom,  // desired DD minimum frequency rate Hz/sec (input)
                          double dfdt_max_nom,  // desired DD maximum frequency rate Hz/sec (input)
                          double Lf,            // PFB overlap factor (=1 for FFT filter bank) 
                          int Nt,              // total number of time samples to integrate (Nt/N0 is power of 2)
                          int N0)              // first stage number of time samples
{
  /* inputs */
  dd->f_sg_min = f_sg_min;
  dd->df_sg = df_sg;
  dd->Nf_sg = Nf_sg;
  dd->dt_sg = dt_sg;
  dd->Nt_sg = Nt_sg;
  dd->dfdt_min_nom = dfdt_min_nom;  // desired DD minimum frequency rate Hz/sec (input)
  dd->dfdt_max_nom = dfdt_max_nom;  // desired DD maximum frequency rate Hz/sec (input)
  dd->Lf = Lf;
  dd->Nt = Nt;
  dd->N0 = N0;  

  /* compute constants */

  dd->Nt = (int) pow(2,round(log2((double)dd->Nt/(double)dd->N0)))*dd->N0;  // make sure Nt/N0 is power of 2

  dd->f_DD_min = dd->f_sg_min;   
  dd->df_DD = dd->df_sg;
  dd->Nf_DD = dd->Nf_sg;

  dd->f_sg_max = dd->f_sg_min + (dd->Nf_sg-1)*dd->df_sg;
  dd->f_DD_max = dd->f_DD_min + (dd->Nf_DD-1)*dd->df_DD;

  dd->Lr = Lf;        // assume Lr=Lf as this is realistically only worthwhile case
  dd->fs = df_sg*dd->Lf;
  dd->n_stage = (int) round(log2((double)dd->Nt/(double)dd->N0))+1;
  
  /* first stage freq rate index limits */
  dd->m_min0 = (int) floor((double)dd->dfdt_min_nom*dd->dt_sg/dd->fs*(dd->Lr*dd->N0));
  dd->m_max0 = (int) ceil( (double)dd->dfdt_max_nom*dd->dt_sg/dd->fs*(dd->Lr*dd->N0));
  /* final stage freq rate index limits */
  dd->m_min = (int) round(dd->m_min0*dd->Nt/dd->N0);
  dd->m_max = (int) round(dd->m_max0*dd->Nt/dd->N0);

  dd->d_dfdt = dd->fs/dt_sg/(dd->Nt*dd->Lr);
  dd->dfdt_min = dd->m_min*dd->d_dfdt;  // actual DD minimum frequency rate Hz/sec
  dd->dfdt_max = dd->m_max*dd->d_dfdt;  // actual DD maximum frequency rate Hz/sec

  dd->Nr = dd->m_max - dd->m_min + 1; 
  dd->Nr0 = dd->m_max0 - dd->m_min0 + 1; 
  dd->N_group0 = dd->Nt/dd->N0;        // DD number of groups  - first stage
  dd->Nr_ext = MAX(dd->Nr0*dd->N_group0,dd->Nr); 

  dd->sg_buffer_bytes = dd->Nf_sg*dd->Nt_sg*sizeof(float);   // presumed spectrogram size in bytes
  dd->DD_buffer_bytes = dd->Nf_sg*dd->Nr0*dd->N_group0*sizeof(float);  // minimum DD buffer size in bytes (1 of 2)
  dd->total_buffer_bytes = dd->sg_buffer_bytes + 2*dd->DD_buffer_bytes;  // SG + 2 DD buffer size sum
}

void print_fastDD_metadata(DD_metadata *dd)   // DD metadata structure with entries filled 
{
  printf("\nfastDD metadata:\n");
  printf("Input  SG: freq %.1f to %.1f by %.2f Hz, %d points\n",dd->f_sg_min,dd->f_sg_max,dd->df_sg,dd->Nf_sg);
  printf("Input  SG: time %d lines with incr %.1f sec, total duration %.1f sec\n",dd->Nt_sg,dd->dt_sg,dd->Nt_sg*dd->dt_sg);
  printf("Output DD: freq %.1f to %.1f by %.2f Hz, %d points\n",dd->f_DD_min,dd->f_DD_max,dd->df_DD,dd->Nf_DD);
  printf("Output DD: rate %.2f to %.2f by %.3f Hz/sec, %d points\n\n",dd->dfdt_min,dd->dfdt_max,dd->d_dfdt,dd->Nr);
  printf("Lf=Lr=%.2f fs=%.2f Nt=%d N0=%d n_stage=%d\n",dd->Lf,dd->fs,dd->Nt,dd->N0,dd->n_stage);

  printf("First stage drift rate indices %d to %d, %d points\n",dd->m_min0,dd->m_max0,dd->Nr0);
  printf("Final stage drift rate indices %d to %d, %d points\n",dd->m_min,dd->m_max,dd->Nr);
  printf("Extended Rate Dimension Nr_ext=%d vs %d\n",dd->Nr_ext,dd->Nr);
  printf("Buffer Sizes SG=%.2f + DD %.2f x 2 = %.2f MB Total\n",
          dd->sg_buffer_bytes/1024./1024.,dd->DD_buffer_bytes/1024./1024.,dd->total_buffer_bytes/1024./1024.);
}

/*
  Note that path_sum is the deDoppler sum det_DD, 
  freq is the integer frequency index, 
  path_offset = drift rate index with respect to the current drift block
  Nr = number of drift rates in current drift block, which may be slightly different 
  from the number of time steps in the original spectrogram

  From seticore comments: 
  Gather information about the top hits.

  The eventual goal is for every frequency freq, we want:

  top_path_sums[freq] to contain the largest path sum that starts at freq
  top_drift_blocks[freq] to contain the drift block of that path
  top_path_offsets[freq] to contain the path offset of that path

  path_sums[path_offset][freq] contains one path sum.
  (In row-major order.)
  So we are just taking the max along a column and carrying some
  metadata along as we find it. One thread per freq.

  The function ignores data corresponding to invalid paths. See
  comments in taylor.cu for details.
*/
void findTopPathSums_cpu(const float* path_sums, int Nr, int n_freq,
                                int drift_block, float* top_path_sums,
                                int* top_drift_blocks, int* top_path_offsets)
{
  int dd_ofs;

  for (int path_offset = 0; path_offset < Nr; ++path_offset) {
    dd_ofs = n_freq * path_offset;
    for (int freq = 0; freq < n_freq; ++freq) {
      float path_sum = path_sums[dd_ofs++];
      if (path_sum > top_path_sums[freq]) {
        top_path_sums[freq] = path_sum;
        top_drift_blocks[freq] = drift_block;
        top_path_offsets[freq] = path_offset;
      }
    }
  }
}

void zeroTopPathSums_cpu( int n_freq, float* top_path_sums,
                          int* top_drift_blocks, int* top_path_offsets)
{
  for (int freq = 0; freq < n_freq; ++freq) {
    top_path_sums[freq] = 0.;
    top_drift_blocks[freq] = 0;
    top_path_offsets[freq] = 0;
  }
}


/*  ======================================================================  */
/*  These functions perform a De-Doppler sum on a spectrogram. It assumes  */
/*  the arrangement of data stream is, all points in first spectra, all     */
/*  points in second spectra, etc...  Data are summed across time           */
/*                     Original version: K. Houston 2023                    */
/*  ======================================================================  */

void fastDD_cpu_stage1(float *det_DD1, float *xx, int m_min0, int m_max0, 
               int Nt, int N0, int n_freq)
{
  // 
  //  fastDD "Slow" first stage
  // 
  // det_DD1[(Nr0*n_group)*(n_freq)]  output array
  // xx[Nt*(n_freq)]              input spectrogram
  // 
  
  int ig,it1,it,m,mm,mm0,kk,k1,t_g_ofs,m_g_ofs;
  int n_group = Nt/N0;
  int Nr0 = m_max0 - m_min0 + 1;
  int *int_bin_offset;
  int xx_ofs,dd_ofs;
  int n_freq_limited;

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
          for (kk=0; kk<n_freq_limited; kk++) {
            det_DD1[dd_ofs++] = xx[xx_ofs++];
          }
        } else {
          for (kk=0; kk<n_freq_limited; kk++) {
            det_DD1[dd_ofs++] += xx[xx_ofs++];
          }
        }
      }
    }
  }
  free(int_bin_offset);
  return;
}


void fastDD_cpu_stage_i(float *det_DD_out, float *det_DD_in, int i_stage,
                   int m_min0, int m_max0,int Nt, int N0, int n_freq)
{
  // 
  //  "Fast" DeDoppler stage
  // 
  // det_DD_out[(Nr1*n_group1)*(n_freq)]  output array - stage i_stage,    Nr_ext>=Nr1*n_group1>=Nr
  // det_DD_in[(Nr0*n_group0)*(n_freq)]   input array  - stage i_stage-1,  Nr_ext =Nr0*n_group0>=Nr
  //

  int ig,m1,kk,i1,i2;
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
        for (kk=0; kk<n_freq_limited; kk++) {
          det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
        }
        for (kk=0; kk<m1; kk++) {
          det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
        }
      } else { // m1<0 negative drifts
        // apply offset of m1 frequency bins to earlier lines in spectrum 
        // equivalent to det_DD_in[m_g_ofs1a+i1][-m1] & det_DD_in[m_g_ofs1b+i1][0]
        dd_ofs1a = (m_g_ofs1a+i1)*n_freq; 
        dd_ofs1b = (m_g_ofs1b+i1)*n_freq; 
        // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][-m1]
        for (kk=0; kk<-m1; kk++) {
          det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
        }
        for (kk=0; kk<n_freq_limited; kk++) {
          det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
        }
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
          // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][0]
          for (kk=0; kk<n_freq_limited; kk++) {
            det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
          }
          for (kk=0; kk<m1; kk++) {
            det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
          }
        } else { // m1<0 negative drifts
          // apply offset of m1 frequency bins to earlier lines in spectrum 
          // equivalent to det_DD_in[m_g_ofs1a+i1][-m1] & det_DD_in[m_g_ofs1b+i1][0]
          dd_ofs1a = (m_g_ofs1a+i1  )*n_freq; 
          dd_ofs1b = (m_g_ofs1b+i1+1)*n_freq; 
          // output DD offset equivalent to det_DD_out[m_g_ofs2+i2][-m1]
          for (kk=0; kk<-m1; kk++) {
            det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++];
          }
          for (kk=0; kk<n_freq_limited; kk++) {
            det_DD_out[dd_ofs2++] = det_DD_in[dd_ofs1a++] + det_DD_in[dd_ofs1b++];
          }
        }
      }
    }
  }

  return;
}


float* fastDD_cpu(float *xx_sg,           // input mag squared SG (spectrogram) values [Nt_sg][Nf_sg]
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
  // BEFORE CALLING fastDD_cpu(), one needs to:
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

  fastDD_cpu_stage1(det_DD_work[0],xx_sg,m_min0,m_max0,Nt,N0,n_freq);
  
  /* i-th stage processing */

  for (i_stage=2; i_stage<=n_stage; i_stage++) {
    i_in = i_stage % 2;
    i_out = (i_stage+1) % 2;
    fastDD_cpu_stage_i(det_DD_work[i_out],det_DD_work[i_in],i_stage,m_min0,m_max0,
                  Nt,N0,n_freq);
  }

  return det_DD_work[i_out];
}
