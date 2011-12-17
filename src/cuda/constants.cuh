#define __cplusplus
#define __CUDACC__

#include <thrust/sort.h>

#include "cudamatrix_types.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <ctime>
#include <cstring>
#include "cuda.h"
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <iostream>
#include <curand_kernel.h>
#include "curand.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"
#include "builtin_types.h"
#include "cutil/cutil.h"
#include "device_functions.h"
#include "cuda_gl_interop.h"

#define BLOCK_SIZE 256
#define BLOCK_SIZE2 512
#define ORBIT_BLOCK_SIZE 128
#define Max_Splits 32 // Maximum number of neutral splits
#define Max_Track_segments 64// Maximum number of track segments
#define Max_energy_sectors 512
#define NPTCLS_PER_COLLIDE_THREAD 1
#define MAX_STEPS 5000

#define debug

//#define USE_STUPID_SORT

#define Cell_granularity 2

#define RANDOM_SEED 12348


// Uncomment this for double precision calculations
#define __double_precision

// Set this if you want to the orbits to be animated.

#define Animate_orbits
#define SPHERE_SPACING 50


#define DO_BEAMCX

#define DONT_DO_NUTRAV

#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }


#ifdef __double_precision
typedef double realkind;
typedef double2 realkind2;
typedef double3 realkind3;
typedef double4 realkind4;
#else
typedef float realkind;
typedef float2 realkind2;
typedef float3 realkind3;
typedef float4 realkind4;
#endif

typedef cudaMatrixT<unsigned int> cudaMatrixui;
typedef cudaMatrixT<realkind> cudaMatrixr;

typedef realkind (*texFunctionPtr)(realkind,realkind,realkind);

const int threadsPerBlock = 256;

__constant__ realkind pi = 3.1415926535897931; //Pi
__constant__ realkind twopi = 6.2831853071795862;
__constant__ realkind thmax = 6.2831853071795862;
__constant__ realkind thmin = 0.0f;


__constant__ realkind ZMP = 1.6726E-24;
__constant__ realkind ZC = 2.9979E10; // Speed of light
__constant__ realkind ZEL = 4.8032E-10;
__constant__ realkind ZT2G = 1.0;

__constant__ realkind I0 = 2.0e6; // Amp
__constant__ realkind mu0 =  1.2566370614359e-6; // N/Amp^2

__constant__ realkind epsilon = 1.0e-34;

__constant__ realkind orbit_error_con = 1.0e-4;
__constant__ realkind znrc_errcon = 1.04976e-3;
__constant__ realkind znrc_pshrnk = -0.333333333333;
__constant__ realkind znrc_pgrow = -0.25;

__constant__ realkind orbit_dt_min = 5.0e-9;

__constant__ int gridSetupIDs_x[16] = {-1,0,1,2,-1,0,1,2,-1,0,1,2,-1,0,1,2};
__constant__ int gridSetupIDs_y[16] = {-1,-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2};

__constant__ int bphi_sign = 1;
__constant__ int jdotb;

__device__ cudaMatrixi3* blockinfo_d;

__constant__ realkind rk4_mults[3] = {0.5,0.5,1.0};

__constant__ realkind V2TOEV = 5.219843860854e-13;

// Binary magic numbers for Z-curve mapping
static __constant__ unsigned int magicB[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
static __constant__ unsigned int magicS[] = {1, 2, 4, 8};

curandState* random_state_pool;
size_t random_state_pool_pitch;
size_t random_state_pool_xsize;
size_t random_state_pool_ysize;

__constant__ int random_state_counter_d;
char* random_state_counter_symbol = "random_state_counter_d";
int random_state_counter = 0;

cudaMatrixf xposition_matrix;
cudaMatrixf yposition_matrix;
cudaMatrixi nptcls_at_step_matrix;

unsigned int original_idx_counter;
__constant__ unsigned int original_idx_counter_d;



int next_tex2D = 0;
int next_tex1DLayered = 0;
int next_tex2DLayered = 0;

extern "C" void orbit_animate(cudaMatrixf xposition,cudaMatrixf yposition,cudaMatrixi nptcls,
												    cudaMatrixf limiter_bitmap,
												    float2 gridspacing, float2 gridorigins,int nsteps,int nptcls_max,int sphere_spacing_in);




/*
__global__
void curand_init_kernel(curandState* random_states, size_t pitch,int random_state_offset,int nptcls_max)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.y;
	unsigned int idBeam = blockIdx.y;
	unsigned int element_id = gidx + gridDim.x*blockDim.x*idBeam+random_state_offset;

	curandState* my_state;

	if(gidx < nptcls_max)
	{
		my_state = (curandState*)((char*)(random_states)+pitch*idBeam)+gidx;

		curand_init(RANDOM_SEED,element_id,0,my_state);
	}


}

__host__
void random_state_pool_init(int xsize,int ysize)
{
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&random_state_pool,&random_state_pool_pitch,
								xsize*sizeof(curandState),ysize));
	random_state_pool_xsize = xsize;
	random_state_pool_ysize = ysize;
	random_state_counter = 0;
}
*/

template <unsigned int blockSize,typename T>
__global__ void basicreduce(T *g_idata, T *g_odata, unsigned int n)
{
	 __shared__ T sdata[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;

	sdata[tid] = 0;

	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize];  i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
		if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<int blockSize,typename T>
__device__
T reduce(T* sdata)
{
	unsigned int tid = threadIdx.x;
	volatile T* s_ptr = sdata;

	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
	__syncthreads();
	if (tid < 32) {
		if (blockSize >=  64) s_ptr[tid] += s_ptr[tid + 32];
		if (blockSize >=  32) s_ptr[tid] += s_ptr[tid + 16];
		if (blockSize >=  16) s_ptr[tid] += s_ptr[tid +  8];
		if (blockSize >=   8) s_ptr[tid] += s_ptr[tid +  4];
		if (blockSize >=   4) s_ptr[tid] += s_ptr[tid +  2];
		if (blockSize >=   2) s_ptr[tid] += s_ptr[tid +  1];
	}
	__syncthreads();


	return sdata[0];
}

__device__
realkind2 device_quadratic_equation(realkind a,realkind b,realkind c)
{
	realkind radical = b*b-4*a*c;
	realkind2 result;

	if(radical < 0.0)
	{
		result.x = 0.0;
		result.y = 0.0;
	}
	else
	{
		radical = sqrt(radical);

		result.x = (-b+radical)/(2*a);
		result.y = (-b-radical)/(2*a);
	}

	return result;

}

__global__ void
doubletorealkind_kernel(realkind* dst_d,double* src_d,unsigned int nelements)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;

	__shared__ double tdata[512];

	if(gidx < nelements)
	{
		tdata[idx] = src_d[gidx];
	}
	__syncthreads();
	if(gidx < nelements)
	{
		dst_d[gidx] = (realkind)tdata[idx];
	}
}
__global__ void
doubletorealkind2D_kernel(realkind* dst_d,double* src_d,size_t dst_pitch,size_t src_pitch,int width,int height)
{
	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int gidy = idy+blockIdx.y*blockDim.y;
	unsigned int tid = idx+blockDim.x*idy;
	unsigned int idx_in = gidx+gidy*src_pitch/sizeof(double);
	unsigned int idx_out = gidx+gidy*src_pitch/sizeof(realkind);

	__shared__ __align__(64) double tdata[1024];

	if((gidx < width)&&(gidy < height))
	{
		tdata[tid] = ((double*)(((char*)src_d+gidy*src_pitch)))[gidx];
	}
	__syncthreads();
	if((gidx < width)&&(gidy < height))
	{
		((realkind*)(((char*)dst_d+gidy*dst_pitch)))[gidx] = (realkind)(tdata[tid]);
	}
}

__global__ void
doubletoMatrixf_kernel(cudaMatrixr dst_d,cudaMatrixd src_d,int3 dims)
{
	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int idz = threadIdx.z;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int gidy = idy+blockIdx.y*blockDim.y;
	unsigned int gidz = idz+blockIdx.z*blockDim.z;

	unsigned int lid = idx+blockDim.x*(idy+blockDim.y*idz);
	unsigned int gid = gidx+dims.x*(gidy+dims.y*gidz);

	__shared__ double doubletoMatrixf_kerneltdata[1024];

	if((gidx < dims.x)&&(gidy < dims.y)&&(gidz < dims.z))
	{
		doubletoMatrixf_kerneltdata[lid] = src_d(gidx,gidy,gidz);
	}
	__syncthreads();
	if((gidx < dims.x)&&(gidy < dims.y)&&(gidz < dims.z))
	{
		dst_d(gidx,gidy,gidz) = (realkind)(doubletoMatrixf_kerneltdata[lid]);
	}
}

__host__
void cudaMemcpydoubletorealkind(realkind* dst_d,double* src_d,unsigned int nelements)
{
	unsigned int GridSize = (512+nelements-1)/512;

	CUDA_SAFE_KERNEL((doubletorealkind_kernel<<<GridSize,512>>>(dst_d,src_d,nelements)));

	cudaThreadSynchronize();

}

__host__
void cudaMemcpy2Ddoubletorealkind(realkind* dst_d,size_t dst_pitch,
																double* src_d,size_t src_pitch,unsigned int width,
																unsigned int height)
{
	dim3 cudaBlockSize(16,16,1);
	dim3 cudaGridSize((cudaBlockSize.x+width-1)/cudaBlockSize.x,(cudaBlockSize.y+height-1)/cudaBlockSize.y,1);

	CUDA_SAFE_KERNEL((doubletorealkind2D_kernel<<<cudaGridSize,cudaBlockSize>>>
									 (dst_d,src_d,dst_pitch,src_pitch,width,height)));

	cudaThreadSynchronize();

}

__host__
void cudaMemcpydoubletoMatrixr(cudaMatrixr dst_d,double* src_h)
{
#ifdef __double_precision
	dst_d.cudaMatrixcpyHostToDevice(src_h);
#else
	int3 dims;
	int ndims = 0;
	int nthreads = 8;
	dim3 cudaBlockSize(1,1,1);
	dim3 cudaGridSize(1,1,1);
	cudaExtent extent = dst_d.getdims();


	dims.x = extent.width/sizeof(realkind);
	dims.y = extent.height;
	dims.z = extent.depth;

	cudaMatrixd src_d(dims.x,dims.y,dims.z);

	src_d.cudaMatrixcpyHostToDevice(src_h);



	if(dims.x > 1)
		ndims++;
	if(dims.y > 1)
		ndims++;
	if(dims.z > 1)
		ndims++;
	printf("dims = %i x %i x %i, ndims = %i\n",dims.x,dims.y,dims.z,ndims);
	switch(ndims)
	{
	case 0:
		return;
	case 1:
		nthreads = 512;
		break;
	case 2:
		nthreads = 16;
		break;
	case 3:
		nthreads = 8;
		break;
	default:
		break;
	}

	if(dims.x > 1)
		cudaBlockSize.x = nthreads;
	if(dims.y > 1)
		cudaBlockSize.y = nthreads;
	if(dims.z > 1)
		cudaBlockSize.z = nthreads;

	cudaGridSize.x = (cudaBlockSize.x+dims.x-1)/cudaBlockSize.x;
	cudaGridSize.y = (cudaBlockSize.y+dims.y-1)/cudaBlockSize.y;
	cudaGridSize.z = (cudaBlockSize.z+dims.z-1)/cudaBlockSize.z;


	cudaDeviceSynchronize();

	CUDA_SAFE_KERNEL((doubletoMatrixf_kernel<<<cudaGridSize,cudaBlockSize>>>(dst_d,src_d,dims)));

	cudaDeviceSynchronize();
	src_d.cudaMatrixFree();
#endif

}

template<typename T>
T* nbicudaMemcpy(T* dest,T* src,size_t* pitch,int width,int height)
{
	CUDA_SAFE_CALL(cudaMemcpy2D(dest,*pitch,src,*pitch,width*sizeof(T),height,cudaMemcpyHostToDevice));
	return dest;
}

__device__
realkind lerp(realkind f0,realkind f1,realkind fx)
{
	return f0+fx*(f1-f0);
}

__host__
void checkMemory(void)
{
	size_t free2 = 0;
	size_t total = 0;
	CUDA_SAFE_CALL(cudaMemGetInfo(&free2,&total));
	printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free2)/(1<<20),(int)(total-free2)/(1<<20));
}

__device__
unsigned int zmap(unsigned int x,unsigned int y)
{
	x /= Cell_granularity;
	y /= Cell_granularity;
	x = (x | (x << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;

	y = (y | (y << 8)) & 0x00FF00FF;
	y = (y | (y << 4)) & 0x0F0F0F0F;
	y = (y | (y << 2)) & 0x33333333;
	y = (y | (y << 1)) & 0x55555555;

	return x | (y << 1);


}































