#!/bin/csh
#Tests a rebuild of nubeam with cuda-memcheck

# Be sure to set nubeam_path to the nubeam source code directory
set nubeam_path = ../nubeam_sandbox/nubeam/nubeam
cd $nubeam_path
cd ..
cd nubeam_comp_exec
(csh ./d3d_test.csh -nptcls 50000)
