
#include "cpu_anim.h"
#include "cudamatrix_types.cuh"
#include <curand_kernel.h>
#include "curand.h"

#define INF 2e10f

#define DIM 1024

#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }

__constant__ int sphere_spacing;

struct __align__(64) Sphere
{
	float r,b,g;
	float radius;
	float x,y,z;
	__device__ float hit(float ox,float oy,float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if(dx*dx + dy*dy < radius*radius)
		{
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz/sqrtf(radius*radius);
			return dz+z;
		}
		return -INF;
	}
};

typedef cudaMatrixT<Sphere> cudaMatrixSphere;

__global__
void sphere_random_state_init(cudaMatrixT<curandState> random_states,int nspheres)
{
	unsigned int gidx = threadIdx.x+blockIdx.x*blockDim.x;

	if(gidx < nspheres){
	curand_init(1235,gidx,0,&(random_states(gidx)));
	}
}

__global__
void sphere_color_init(cudaMatrixSphere spheres,
		cudaMatrixT<curandState> random_states,int nspheres)
{
	unsigned int gidx = threadIdx.x+blockIdx.x*blockDim.x;

	if(gidx < nspheres){
		spheres(gidx).r = curand_uniform(&(random_states(gidx)));
		spheres(gidx).g = curand_uniform(&(random_states(gidx)));
		spheres(gidx).b = curand_uniform(&(random_states(gidx)));
	}

}

__global__
void populate_spheres_kernel(cudaMatrixSphere spheres,
													cudaMatrixf xposition,cudaMatrixf yposition,
													cudaMatrixi nptcls,float2 gridspacing, float2 origin,
													int nptcls_max,int nspheres,int istep)
{
	unsigned int gidx = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int sidx = gidx*sphere_spacing;

	float nr = 128;
	float nz = 128;
	float dx = DIM;
	float dy = DIM;
	int nptcls_step = nptcls(istep);


	if(sidx<nptcls_step)
	{
		spheres(gidx).x = (dx*((xposition(gidx,0,istep)-origin.x)/gridspacing.x))/nr;
		spheres(gidx).y = (dy*((yposition(gidx,0,istep)-origin.y)/gridspacing.y))/nz;
		spheres(gidx).z = -400.0;
		spheres(gidx).radius = 5.0;

		//printf("sphere %i at %f,%f\n",gidx,spheres(gidx).x,spheres(gidx).y);

	}
	else if(gidx < nspheres)
	{
		spheres(gidx).z = -INF;
		spheres(gidx).radius = 10.0;

	}

}

__global__
void generate_frame_kernel( uchar4 *ptr, cudaMatrixSphere spheres,
												cudaMatrixf limiter_bitmap,cudaMatrixi nptcls,
												int nptcls_max,int nspheres,int istep)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;


    float r=0, g=0, b=0;
    float maxz = -INF;
    int nptcls_step = nptcls(istep);
    nspheres = min(nspheres,nptcls_step);

    r = 0.0*limiter_bitmap(x,y);
    g = 0.5*limiter_bitmap(x,y);

    // now calculate the value at that position

    for(int i=0; i<(nspheres); i++) {
        float   n;
        float   t = spheres(i).hit( x, y, &n );
        if (t > maxz) {
            float fscale = n;
          //  printf("sphere %i at %f,%f\n",i,spheres(i).x,spheres(i).y);
            r = spheres(i).r * fscale;
            g = spheres(i).g * fscale;
            b = spheres(i).b * fscale;
            maxz = t;
        }
    }

    if(istep > 1)
    {
		ptr[offset].x = (int)fmax((r*255),(0.99f*(int)ptr[offset].x));
		ptr[offset].y = (int)fmax((g*255),(0.99f*(int)ptr[offset].y));
		ptr[offset].z = (int)fmax((b*255),(0.99f*(int)ptr[offset].z));
		ptr[offset].w = 255;
    }
    else
    {
		ptr[offset].x = (int)(r*255);
		ptr[offset].y = (int)(g*255);
		ptr[offset].z = (int)(b*255);
		ptr[offset].w = 255;
    }
}

// globals needed by the update routine
struct DataBlock {
	cudaMatrixSphere Spheres;
	cudaMatrixf xposition;
	cudaMatrixf yposition;
	cudaMatrixf limiter_bitmap;
	cudaMatrixi nptcls;
	uchar4* outData;
	uchar4* temp_data;
	CPUAnimBitmap  *bitmap;
	float2 gridspacing;
	float2 gridorigins;
	int istep;
	int maxsteps;
	int nptcls_max;
	int iteration;
};

void anim_gpu(DataBlock* data, int ticks )
{
    dim3    grids((DIM)/16,(DIM)/16);
    dim3    threads(16,16);
    int nptcls_max = data->nptcls_max;
    int nspheres = (nptcls_max);
    int cudaGridSize = (512+nspheres-1)/512;



    CPUAnimBitmap  *bitmap = data->bitmap;

    uchar4* pixels = data->outData;

    (data->istep)+=1;

    if(data->istep >= (data->maxsteps-1))
    {
    	data->istep = 0;
    	data->iteration++;
    }

    int istep = data->istep;
   // printf( "Animation Time Step:  %i\n",istep);


    cudaDeviceSynchronize();
    CUDA_SAFE_KERNEL((populate_spheres_kernel<<<cudaGridSize,512>>>(
    		data->Spheres,data->xposition,data->yposition,
    		data->nptcls,data->gridspacing,data->gridorigins,nptcls_max,nspheres,istep)));
    cudaDeviceSynchronize();
    CUDA_SAFE_KERNEL(( generate_frame_kernel<<<grids,threads>>>(
    		pixels, data->Spheres,data->limiter_bitmap,data->nptcls,nptcls_max,nspheres,istep)));
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaMemcpy( bitmap->get_ptr(),pixels,bitmap->image_size(),cudaMemcpyDeviceToHost ));
    cudaDeviceSynchronize();
}

void anim_exit(DataBlock *d)
{
	d-> Spheres.cudaMatrixFree();
	d -> xposition.cudaMatrixFree();
	d -> yposition.cudaMatrixFree();
	d -> nptcls.cudaMatrixFree();
	d -> limiter_bitmap.cudaMatrixFree();
	cudaFree(d->outData);
	CUDA_SAFE_CALL(cudaDeviceReset());

}

extern "C" void orbit_animate(cudaMatrixf xposition,cudaMatrixf yposition,cudaMatrixi nptcls,
								cudaMatrixf limiter_bitmap,
								float2 gridspacing, float2 gridorigins,int nsteps,int nptcls_max,int sphere_spacing_in)
{

	DataBlock data;
	CPUAnimBitmap bitmap( DIM, DIM, &data );
	int imageSize = bitmap.image_size();
	int nspheres = (nptcls_max/sphere_spacing_in);
	int cudaGridSize = (512+nspheres-1)/512;

	data.bitmap = &bitmap;
	data.maxsteps = nsteps;
	data.nptcls_max = nptcls_max/sphere_spacing_in;
	data.istep = 0;
	data.iteration = 0;

	cudaMatrixSphere Spheres(nspheres);
	cudaMatrixT<curandState> random_states(nspheres);
	cudaMalloc((void**)&data.outData,imageSize);

	const char* symbol = "sphere_spacing";

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(symbol,&sphere_spacing_in,sizeof(int)));

	data.Spheres = Spheres;
	data.xposition = xposition;
	data.yposition = yposition;
	data.nptcls = nptcls;
	data.gridspacing = gridspacing;
	data.gridorigins = gridorigins;
	data.limiter_bitmap = limiter_bitmap;

	CUDA_SAFE_KERNEL((sphere_random_state_init<<<cudaGridSize,512>>>(
			random_states,nspheres)));
	CUDA_SAFE_KERNEL((sphere_color_init<<<cudaGridSize,512>>>(
			data.Spheres,random_states,nspheres)));

	random_states.cudaMatrixFree();

	bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
            (void (*)(void*))anim_exit );


}












