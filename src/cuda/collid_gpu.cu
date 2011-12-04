#include "particleclass.cuh"
#include <iostream>


__global__
void collide_gpu_kernel(XPlist particles_global,Environment* plasma_in,int max_steps)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	__shared__ XPlist particles;

	particles.shift_local(particles_global);

	int isteps;
	int flag;

	if(gidx < particles_global.nptcls)
	{
		if(particles.fpskip[idx] < 1.0)
			isteps = 1+rint(1.0/particles.fpskip[idx]);
		else
			isteps = 1;

		for(int i=0;i<isteps;i++)
		{
			flag = particles.collide(plasma_in,isteps);

			if(flag == 2)
				break;
			particles.update_gc(plasma_in);
			particles.gphase();
			particles.update_flr(plasma_in);
		}



	}

}


__host__
void collide_gpu(void)
{

}
