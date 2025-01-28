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

Further refactoring, TBD.

Note that after switching branches ("git checkout branch_name"), it is necessary to recompile ("meson compile").
