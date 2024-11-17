# seticore2 branch information

## sc0

seticore fork modified to run on a local machine

## sc1

Add modifications to spectrogram normalization, threshold setting, and SNR estimates of detections.
Uses baseline Taylor-tree GPU code.  

## sc2

Add modified DD files, and allow for self-managed buffers.  Uses baseline Taylor-tree GPU code. This is the baseline for the IAC-2024 paper.

## sc2a

Variant of sc2 using fastDD GPU code.  This has not been optimized to the same extent as the Taylor-tree version.

## sc2b

Variant of sc2 using fastDD CPU code.  This might be useful in systems without a GPU.

## sc3

Further refactoring, TBD.

Note that after switching branches ("git checkout branch_name"), it is necessary to recompile ("meson compile").
