#!/bin/csh
#Tests a rebuild of nubeam with cuda-memcheck

# Be sure to set nubeam_path to the nubeam source code directory
setenv COMPUTE_PROFILE 1
setenv COMPUTE_PROFILE_CSV 1
setenv COMPUTE_PROFILE_CONFIG ./Docs/Profiling/compute_profile.conf
setenv COMPUTE_PROFILE_LOG ./Docs/Profiling/cuda_profile_%d.csv


computeprof &
csh ./nubeam_test.csh
