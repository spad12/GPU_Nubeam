#!/bin/csh
# Rebuilds and tests gpu_nubeam

# Be sure to set nubeam_path to the nubeam source code directory
set nubeam_path = ../nubeam_sandbox/nubeam/nubeam
set share_path = ../nubeam_sandbox/nubeam/share

cp -fr ./src/cuda/orbit_gpu.cu $nubeam_path/
cp -fr ./src/cuda/display_orbits.cu $nubeam_path/
cp -fr ./src/cuda/*.cuh $nubeam_path/
cp -f ./src/fortran/nubeam_new/*.* $nubeam_path/
cp -f ./src/fortran/nubeam_replacements/*.* $nubeam_path/
cp -f ./include/*.* $nubeam_path/
#cp -f ./include/radixSort/*.* $nubeam_path
cp -f ./Docs/exampleshare/Make.local $share_path/
cp -f ./src/fortran/nubeam_replacements/Make2 $nubeam_path/
mv -f $nubeam_path/nbspline_gpu.f90 $nubeam_path/nbspline.f90
mv -f $nubeam_path/nbspline_mod_gpu.f90 $nubeam_path/nbspline_mod.f90
mv -f $nubeam_path/orball_gpu.F $nubeam_path/orball.F
cd $nubeam_path
cd ..
make uninstall
cd nubeam
make -f Make2 uninstall
#make -f Make2 realclean
cd ..
make #this should build any changes
cd nubeam_comp_exec
make exec -B
cd ..
make install -B
cd nubeam_comp_exec
#(csh ./d3d_test.csh -nptcls 20000) > gpu_debug.msg 
#(cuda-memcheck csh ./d3d_test.csh -nptcls 4000) > gpu_debug.msg 
