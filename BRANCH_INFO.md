# seticore2 branch information

## sc0

seticore fork with trivial changes to run on a local machine, including GPU architecture spec, executable name seticore2, etc.

## sc1

Add modifications to spectrogram normalization, threshold setting, and SNR estimates of detections.
Uses baseline Taylor-tree GPU code.  Switch NEW_NORM is defined to allow comparison between new (=1) and old (=0) normalization methods.

Note: With changes to any .h files, may need to do "meson setup --wipe" before "meson compile"

## sc2

Add modified DD files, and allow for self-managed buffers.  Uses baseline Taylor-tree GPU code. This is the baseline for the IAC-2024 paper.  

New normalization is adopted (NEW_NORM=1 code). Switch MANAGED_INPUT defined to examine Unified Memory managed input (=1, previous baseline) vs. unmanaged input (=0, new baseline) with explicit host->device and device-> host transfers for DeDoppler processing.  Unmanaged buffers run considerably faster, with only a small number additional lines of code.

Added function to remove DC offset in spectrogram column sums by replacing points near DC with surrounding average.  Simply setting these points to zero upsets the chi-squared noise standard deviation estimate, causing anomaly in SNR estimate near DC.

## sc2a

Variant of sc2 using fastDD GPU code.  This has not been optimized to the same extent as the Taylor-tree version.

## sc2b

Variant of sc2 using fastDD CPU code.  This might be useful in systems without a GPU.

## sc3

Further normalization refinements.  Instead of having a single mean & std estimate for a subband, the subband estimates are interpolated for every frequency in the coarse channel.  The entire input spectrogram is divided by the point-by-point mean.  This equalizes or "flattens" the overall spectrum. The normalization process is then repeated so that the mean & std statistics reflect deviations from a flattened spectrum, which greatly reduces the effect of slope in the original spectrum.

Boxcar filtering (aka "de-smearing") is applied to the DeDoppler output.  A "moving average" filter (FIR filter with Nbox equal coefficients) is applied to every line of the DD output. Nbox is chosen to offset the spectral "smearing" that occurs to a drifting tone when a spectrogram is averaged by Nsti lines prior to DeDoppler.  The smearing spreads energy to Nbox adjacent frequency points.  A net gain of SNR (ideally sqrt(Nbox)) is obtained by summing adjacent frequency bins together in the moving average.

DC offset excision is also updated.

## sc3a

Identical to sc3, except the model for net gain of SNR from boxcar filtering (ideally sqrt(Nbox)) is changed to pow(Nbox,.40), ie change Nbox^.50 in sc3 to Nbox^.40 in sc3a.  This model is employed in choosing between drift rates according to the best SNR.  Empirically, this significantly reduces the tendency for drift rates to go to the positive or negative drift rate limits (the drift rate "rails"), especially in RFI regions.

## sc4

Addition of broadband (BB) signal detection.  The subband statistics computed for spectrum normalization can be used to distinguish between a chi-square background (receiver noise only) and a corrupted background (RFI or correlated spectral noise) within that subband.  For chi-square noise with two combined polarizations, the subband std/mean ratio is nominally 1/sqrt(2*Navg).  With BB signals or RFI, the std/mean ratio rises considerably.  

Within a BB subband, we can change the detection criteria for narrowband (NB) signals of interest so as to avoid declaring a large number of spurious NB detections.  For sc4, the subband std/mean estimate is increased by a factor of 10 (effectively increasing the detection threshold by a factor of 10), which prevents most NB detections from occurring in BB regions.

## sc5

BB detection for a subband is based (indirectly) on Block Spectral Kurtosis (BlkSK).  The thresholds for declaring a BB detection have been tightened (BlkSK>1.16), so many more BB detections occur.

The NB detection SNR thresholds within a BB segement are now the same as for non-BB segments, allowing some hits to occur in BB segments.  However, other simple screening criteria for hits within BB segments are applied based on BlkSK and the drift rate, as follows:

    Regardless of BB detection: Reject all if local BlkSK > 3, or |drift_rate| < min_drift from command line
    If within a BB segment and 1.16 < BlkSK < 3, screen according to drift rate:
        Reject if -0.6 < drift_rate < +.1

Note that GPS/GNSS satellites produce hits with significant negative drift rates due to satellite orbits.  The drift rate bounds can be tightened for non-LEO bands.

Screening greatly reduces hits in capture files while having small effect on test signals.  This is a proof-of-concept, and works reasonably well.  Ultimately the user will probably want to turn off screening and design a custom ML screening algorithm separate from seticore2.                

## sc6

Experimental branch: additional changes TBD.


Note that after switching branches ("git checkout branch_name"), it is necessary to recompile ("meson compile").
