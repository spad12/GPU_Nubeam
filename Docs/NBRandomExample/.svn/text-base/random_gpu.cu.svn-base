/*
 	 This is the gpu implementation of the pseudo random number generator developed by
 	 Charles Karney <karney@princeton.edu>.

 	 Version 2.1 of random number routines
 	 Version 1.0 of GPU implementation
 	 Author: Joshua Payne <spad12@mit.edu>
 	 Date: July 5, 2010

 	 *this was developed for devices of compute capability 2.0 or greater*
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <cuPrintf.cu>
#include <mp1-util.h>
//#include <ctime>

__align__(8) double* random_array_pointer;
int random_array_index = -1;
int random_max;
int* gpu_seed;

__device__ void random_init_gpu(int* seed,int* ri,double* ra)
{
	int BlockSize = blockDim.x*blockDim.y;
	int b = rint(pow((double)2,14));
	double del =  pow((double)2,-14);
	double ulp = pow((double)2,-47);

	int a0 = 15661;
	int a1 = 678;
	int a2 = 724;
	int a3 = 5245;
	int a4 = 13656;
	int a5 = 11852;
	int a6 = 29;
	int c0 = 1;

	int t;
	int s[8];
	int z[8];
	int odd;


	for (int i=0;i<8;i++)
	{
		s[i]=seed[i];
	}
	odd = ((s[7]%2) != 0);
	ra[0] = (((s[7]*del+s[6])*del+s[5])*del+(int)s[4]/512)*512*del;
	for (int j = 1;j<99;j++)
	{

		z[0]=c0+a0*s[0];
		z[1]=a0*s[1]+a1*s[0];
		z[2]=a0*s[2]+a1*s[1]+a2*s[0];
		z[3]=a0*s[3]+a1*s[2]+a2*s[1]+a3*s[0];
		z[4]=a0*s[4]+a1*s[3]+a2*s[2]+a3*s[1]+a4*s[0];
		z[5]=a0*s[5]+a1*s[4]+a2*s[3]+a3*s[2]+a4*s[1]+a5*s[0];
		z[6]=a0*s[6]+a1*s[5]+a2*s[4]+a3*s[3]+a4*s[2]+a5*s[1]+a6*s[0];
		z[7]=a0*s[7]+a1*s[6]+a2*s[5]+a3*s[4]+a4*s[3]+a5*s[2]+a6*s[1];

		t = 0;
		for (int i=0;i<8;i++)
		{
			t = (int)(t/b)+z[i];
			s[i]=t%b;
		}
		odd = (odd || (s[7]%2 != 0));
		ra[j] = (((s[7]*del+s[6])*del+s[5])*del+(int)s[4]/512)*512*del;
	}
	ri[0] = 100;
	if (odd != 1)
	{
		z[0]=c0+a0*s[0];
		z[1]=a0*s[1]+a1*s[0];
		z[2]=a0*s[2]+a1*s[1]+a2*s[0];
		z[3]=a0*s[3]+a1*s[2]+a2*s[1]+a3*s[0];
		z[4]=a0*s[4]+a1*s[3]+a2*s[2]+a3*s[1]+a4*s[0];
		z[5]=a0*s[5]+a1*s[4]+a2*s[3]+a3*s[2]+a4*s[1]+a5*s[0];
		z[6]=a0*s[6]+a1*s[5]+a2*s[4]+a3*s[3]+a4*s[2]+a5*s[1]+a6*s[0];
		z[7]=a0*s[7]+a1*s[6]+a2*s[5]+a3*s[4]+a4*s[3]+a5*s[2]+a6*s[1];
	      t = 0;
	      for (int i=0;i<8;i++)
	      {
	  		t = (int)(t/b)+z[i];
	  		s[i]=(t%b);
	      }
	      int j = (int)s[7]*BlockSize/b;
	      ra[j] = ra[j]+ulp;
	}
	return;
}

__device__ void rand_axc_gpu(int* a,int* x,int* c)
{
	int b = pow(2.0,14);
	int z[8];
	int t;
	for (int i = 0;i < 8;i++)
	{
		z[i] = c[i];
	}
	for (int j = 0;j<8;j++)
	{
		for (int i=j;i<8;i++)
		{
			z[i] = z[i]+(a[j]*x[i-j]);
		}
	}
	t=0;
	for (int i=0;i<8;i++)
	{
		t = (int)(t/b)+z[i];
		x[i]= (t%b);
	}
	return;
}

__device__ void rand_next_seed_gpu(int n,int* ax,int* cx,int* y)
{
	int a[8];
	int c[8];
	int z[8] = {0,0,0,0,0,0,0,0};
	int t[8];
	int m = n;
	int i;

	if(n == 0)
	{
		return;
	}

	for (i = 0;i < 8;i++)
	{
		a[i] = ax[i];
		c[i] = ax[i];
	}

	while (m > 0)
	{
		if (m%2 > 0)
		{
			rand_axc_gpu(a,y,c);
		}
		m = (int)m/2;
		if (m == 0)
		{
			return;
		}
		for (i = 0; i < 8;i++)
		{
			t[i] = a[i];
		}
		rand_axc_gpu(t,a,z);
	}

}

__device__ void next_seed_gpu(int n0,int n1,int n2,int* seed)
{
	int af0[] = {15741,8689,9280,4732,12011,7130,6824,12302};
	int ab0[] = {9173,9894,15203,15379,7981,2280,8071,429};
	int af1[] = {8405,4808,3603,6718,13766,9243,10375,12108};
	int ab1[] = {6269,3240,9759,7130,15320,14399,3675,1380};
	int af2[] = {445,10754,1869,6593,385,12498,14501,7383};
	int ab2[] = {405,4903,2746,1477,3263,13564,8139,2362};
	int cf0[] = {16317,10266,1198,331,10769,8310,2779,13880};
	int cb0[] = {8383,3616,597,12724,15663,9639,187,4866};
	int cf1[] = {13951,7170,9039,11206,8706,14101,1864,15191};
	int cb1[] = {15357,5843,6205,16275,8838,12132,2198,10330};
	int cf2[] = {2285,8057,3864,10235,1805,10614,9615,15522};
	int cb2[] = {8463,575,5876,2220,4924,1701,9060,5639};



	if (n2 > 0)
	{
		rand_next_seed_gpu(n2,af2,cf2,seed);
	}

	if (n2 < 0)
	{
		rand_next_seed_gpu(-n2,ab2,cb2,seed);
	}

	if (n1 > 0)
	{
		rand_next_seed_gpu(n1,af1,cf1,seed);
	}

	if (n1 < 0)
	{
		rand_next_seed_gpu(-n1,ab1,cb1,seed);
	}

	if (n0 > 0)
	{
		rand_next_seed_gpu(n0,af0,cf0,seed);
	}

	if (n0 < 0)
	{
		rand_next_seed_gpu(-n0,ab0,cb0,seed);
	}
}

__device__ void rand_batch_gpu(int* ri,double* ra)
{
	double w[1009-100];
	double tmp;
	int i;

	for (i=0;i<63;i++)
	{
		tmp=ra[i]+ra[i+100-63];
		if (tmp>=1)
		{
			w[i]=tmp-1;
		}
		else
		{
			w[i]=tmp;
		}
	}

	for (i=63;i<100;i++)
	{
		tmp=ra[i]+w[i-63];
		if (tmp>=1)
		{
			w[i]=tmp-1;
		}
		else
		{
			w[i]=tmp;
		}
	}

	for (i=100;i<(1009-100);i++)
	{
		tmp=w[i-100]+w[i-63];
		if (tmp>=1)
		{
			w[i]=tmp-1;
		}
		else
		{
			w[i]=tmp;
		}
	}
	for (i=(1009-100);i<(1009-100+63);i++)
	{
		tmp=w[i-100]+w[i-63];
		if (tmp>=1)
		{
			ra[i-1009+100]=tmp-1;
		}
		else
		{
			ra[i-1009+100]=tmp;
		}
	}
	for (i=(1009-100+63);i<(1009);i++)
	{
		tmp=w[i-100]+ra[i-1009+100-63];
		if (tmp>=1)
		{
			ra[i-1009+100]=tmp-1;
		}
		else
		{
			ra[i-1009+100]=tmp;
		}
	}
	ri[0] = 0;
	return;
}

__device__ void random_array_gpu(double* y,int n,int* ri,double* ra)
{
	double ulp2=pow(2.0,-47);
	int k;

	if (n<=0)
	{
		return;
	}
	k=min(n,100-ri[0]);
	for (int i=0;i<k;i++)
	{
		y[i]=ra[i+ri[0]]+ulp2;
	}
	for (int j=k;j<n;j+=100)
	{
		rand_batch_gpu(ri,ra);
		for (int i=j;i<(min(j+100,n));i++)
		{
			y[i]=fmod(ra[i-j+ri[0]]+ulp2+y[i],y[i]);
		}
		ri[0] = ri[0]+min(100,(n-j));
	}
	return;
}

__global__ void random_gpu_kernel(double* ra_global,
							int* seed_init,int n,int n_arrays)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < n_arrays)
	{

		//rng_state curstate = rng_curstate[idx*100];
		double *ra = &ra_global[idx*100];
		int ri = 0;
		int seed[8];
		for (int i=0;i<8;i++)
		{
			seed[i]=seed_init[i];
		}
		next_seed_gpu(idx,0,0,seed);
		random_init_gpu(seed,&ri,ra);
		rand_batch_gpu(&ri,ra);
		random_array_gpu(ra,100,&ri,ra);

		if (idx == n_arrays-1)
		{
			for (int i=0;i<8;i++)
			{
				seed_init[i] = seed[i];
			}
		}

	}
}

/*
void nbi_random_gpu(rng_state rng_curstate,double nbi_srng,
						int rng_k,int rng_s,int rng_c)
*/

void random_gpu_array(double* random_numbers,int n_numbers,int num_arrays,int* h_seed)
{

	int n=100; // Number of random numbers to return per input array
	int n_arrays = num_arrays;
	int how_many_numbers = n_numbers;
	int size1 = sizeof(double)*n*n_arrays;
	cudaError status;

	// Check to see if the quantity of numbers to be generated is
	// less than the number of arrays * 64
	if (n_arrays*n < how_many_numbers)
	{
		printf("Not enough state arrays to generate all of the numbers. \n"
				"64*n_arrays must be greater than the quantity of numbers"
				"to be generated. \n");
		return;
	}

	// Allocate Device Memory
	__align__(4) int* d_seed;
	cudaMalloc((void**)&d_seed,sizeof(int)*8);
	__align__(8) double* d_ra;
	cudaMalloc((void**)&d_ra,size1);
	cudaMemset(d_ra,0,size1);

	//check_launch("gpu allocate");

	// Copy arrays from host to device
	//cudaMemcpy(d_ri,ri,sizeof(int)*n_arrays,cudaMemcpyHostToDevice);
	//cudaMemcpy(d_ra,ra,sizeof(double)*100*n_arrays,cudaMemcpyHostToDevice);
	cudaMemcpy(d_seed,h_seed,sizeof(int)*8,cudaMemcpyHostToDevice);

	//check_launch("gpu copy");
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "cudaMemcpy seed -> device: %s\n", cudaGetErrorString(status));}

		// Set Kernel Execution Parameters
		// total number of threads = GridDim*BlockDim
		int BlockDim = 256;
		int GridDim = (n_arrays+BlockDim-1)/BlockDim;

		// Execute GPU routine
		//printf("Before Kernel \n");
		random_gpu_kernel<<<GridDim,BlockDim>>>(d_ra,
										 d_seed,n,n_arrays);
		status = cudaGetLastError();
		if(status != cudaSuccess){fprintf(stderr, "random_gpu_kernel: %s\n", cudaGetErrorString(status));}

		// Make sure that all of the threads are done before moving on
		cudaThreadSynchronize();

		//check_launch("gpu launch");


	// Copy the results from the GPU memory to the host memory.
	status = cudaMemcpy(random_numbers,d_ra,sizeof(double)*how_many_numbers,cudaMemcpyDeviceToHost);

	if(status != cudaSuccess){fprintf(stderr, "cudaMemcpy random_numbers: %s\n", cudaGetErrorString(status));}
	cudaMemcpy(h_seed,d_seed,sizeof(int)*8,cudaMemcpyDeviceToHost);
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "cudaMemcpy seed -> host: %s\n", cudaGetErrorString(status));}

	//cudaMemcpy(ri,d_ri,sizeof(int)*n_arrays,cudaMemcpyDeviceToHost);
	//cudaMemcpy(ra,d_ra,sizeof(double)*100*n_arrays,cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_seed,d_seed,sizeof(int)*8,cudaMemcpyDeviceToHost);

	cudaFree(d_ra);
	cudaFree(d_seed);

	return;

}

extern "C" void random_init_(int* seed,int* ri,double* ra)
{
	if (random_array_index == -1)
	{
		//printf("Inside external init \n");

		int narrays = 2<<17;
		int nnumbers = 100*narrays;

		random_array_pointer  = (double*)malloc(nnumbers*sizeof(double));

		random_gpu_array(random_array_pointer,nnumbers,narrays,seed);

		random_array_index = 0;

		ri[0] = 0;
		for (int i=0;i<100;i++)
		{
			ra[i] = random_array_pointer[i];
		}
	}
	gpu_seed = seed;
	return;

}

void random_init(int* seed,int* ri)
{
	//printf("Inside internal init \n");

	int narrays = 2<<17;
	int nnumbers = 100*narrays;

	if (random_array_index == -1)
	{
		random_array_pointer  = (double*)malloc(nnumbers*sizeof(double));
	}

	random_gpu_array(random_array_pointer,nnumbers,narrays,seed);

	random_array_index = 0;
	gpu_seed = seed;


	ri[0] = 0;

	return;

}

double get_random(int ri)
{

	double y;

	y = 5.0;
	while ((y > 1)||(y < 0))
	{
		if (ri >  98)
		{
			ri = 0;
			random_array_index += 1;
			if (random_array_index > ((2<<17) - 2))
			{
				random_init(gpu_seed,&ri);
			}
		}

		y = fabs(random_array_pointer[random_array_index*100+ri]);
		if  ((y > 1)||(y < 0)) {printf(" bad Y = %f", y);}
		ri += 1;
	}



	//printf("y= %f @ %i %i\n",y,ri,random_array_index);


	return y;
}

extern "C" void rand_batch_(int* ri,double* ra)
{
	for (int i=0;i<100;i++)
	{
		ra[i] = get_random(i);
		//printf("y= %f \n",ra[i]);
	}

	ri[0] = 0;
}

void rand_batch(int* ri,double* ra)
{
	for (int i=0;i<100;i++)
	{
		ra[i] = get_random(i);
		//printf("y= %f \n",ra[i]);
	}

	ri[0] = 0;
}
extern "C" void random_array_(double* y,int* n1,int* ri,double* ra)
{

	int k = ri[0];
	int n = *n1;

	for (int i=0;i<n;i++)
	{
		if (k > 99)
		{
			rand_batch(ri,ra);
			k = 0;
		}
		y[i] = ra[k];
		ri[0] += 1;
		k += 1;
	}
	//printf("y = %f \n",y[2]);
	return;
}

extern "C" void srandom_array_(float* y,int* n1,int* ri,double* ra)
{
	int k = ri[0];
	int n = *n1;

	for (int i=0;i<n;i++)
	{
		if (k > 99)
		{
			rand_batch(ri,ra);
			k = 0;
		}

		y[i] = (float)ra[k];
		ri[0] += 1;
		k += 1;
	}
	//printf("y = %f \n",y[2]);
	return;
}



