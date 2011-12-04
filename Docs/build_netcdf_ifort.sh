#!/bin/sh
#this build works fine with gcc,g77


make distclean
#INSTALL=/usr/local/netcdf-3.6.0-p1 
    TOP=/usr/local/netcdf
    CC=/usr/bin/gcc
    export CC 
    CPPFLAGS="-DNDEBUG -DpgiFortran"
    export CPPFLAGS
    FC=ifort
    export FC 
    FFLAGS="-O -w"              # "-Nx400" allows fortran/netcdf.inc to
                                # have many EXTERNAL statements
    export FCFLAGS="-fPIC" 
    export F90FLAGS="-fPIC"
    export FFLAGS
    CXX=/usr/bin/c++
    export CXX           
    CFLAGS=-O
    export CFLAGS

./configure --prefix=$TOP --enable-shared
make
make test 


#check if user is su:

#  if [ $es -eq 0 ]; then
#     echo " error must be su to execute this script " ;
#     exit 1;
#  fi


#make install -must be root
