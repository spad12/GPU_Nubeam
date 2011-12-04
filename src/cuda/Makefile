NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= ../../include,/home/joshp/lib/cudpp,../../include/radixSort
CUDAFORTRAN_FLAGS := -L$(LD_LIBRARY_PATH) -lcuda -I$(CUDA_INCLUDE_PATH)
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:= -G -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -I$(CUDA_INCLUDE_PATH) 
RADIXSORTOBJDIR = ../../include/radixSort/
RADIXSORTOBJ = radixsort.cpp.o radixsort.cu.o

all: orbit_gpu.o

radixsort.cu.o: $(RADIXSORTOBJDIR)radixsort.cu
	$(NVCC) -c $(NVCCFLAGS) -L/home/joshp/lib/cudpp -lcudpp_x86_64 -L/home/joshp/lib -lcutil_x86_64 $(NVCCFLAGS) -o radixsort.cu.o $<

radixsort.cpp.o: $(RADIXSORTOBJDIR)radixsort.cpp
	$(NVCC) -c $(NVCCFLAGS) -L/home/joshp/lib/cudpp -lcudpp_x86_64 -L/home/joshp/lib -lcutil_x86_64 $(NVCCFLAGS) -o radixsort.cpp.o $<
	
%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) -L/home/joshp/lib/cudpp -lcudpp_x86_64 -L/home/joshp/lib -lcutil_x86_64 $(NVCCFLAGS) -o ./$*.o $<

	
	