#include "texture_refs.cuh"



class XPTextureSpline
{
public:
	BCspline* spline; // dummy spline
	texFunctionPtr pt2Function;
	int2 griddims;
	realkind2 gridspacing;
	realkind2 origin; //  x[0][0], y[0][0]
	size_t pitch;
	__host__ __device__
	XPTextureSpline(){;}

	__host__
	void allocate(int2 griddims_in,enum XPgridlocation location_in);
	__host__
	void setup(double* spline_in,realkind xmin,realkind xmax,realkind ymin,realkind ymax);

	__host__
	void fill2DLayered(cudaMatrixr data_in);

	__device__
	void allocate_local(BCspline* spline_local);
	// Find the cell index of a given x and y.
	__device__
	void shift_local(XPgrid grid_in,int nx,int ny);
	__device__
	int2 findindex(realkind px,realkind py);
	// Evaluate the spline at the given location
	template<enum XPgridderiv ideriv>
	__device__
	realkind BCspline_eval(realkind px,realkind py);

	__device__
	realkind get_spline(int nx,int ny,int coeff)
	{

		return pt2Function(nx,ny,coeff);

		//return *((const BCspline*)((char*)spline+ny*pitch)+nx);
	}

private:
	enum XPgridlocation location;
	char* symbol;
	cudaArray* cuArray;

};






__host__
 void XPTextureSpline::allocate(int2 griddims_in,enum XPgridlocation location)
{
	griddims = griddims_in;

	printf(" nelements in XPgrid::setup = %i \n",griddims.x*griddims.y);

	//CUDA_SAFE_KERNEL(cudaMallocPitch((void**)&spline,&pitch,(griddims.x+8)*sizeof(BCspline),(griddims.y+8)));
	//CUDA_SAFE_KERNEL(cudaMalloc((void**)&spline,(griddims.x*griddims.y)*sizeof(BCspline)));

}


__device__
void XPTextureSpline::allocate_local(BCspline* spline_local)
{
	// This function sets up a 4x4 spline array in shared memory on the gpu.
	if(threadIdx.x == 0)
	{
		griddims.x = 16;
		griddims.y = 16;
		pitch = 16*sizeof(BCspline);
	}
	__syncthreads();

	return;

}

__device__
int2 XPTextureSpline::findindex(realkind px,realkind py)
{
	int2 result;
	result.x = max(0,min((griddims.x-2),((int)(floor((px-origin.x)/gridspacing.x)))));
	result.y = max(0,min((griddims.y-2),((int)(floor((py-origin.y)/gridspacing.y)))));

	return result;
}

// This is going to be a template function to improve performance
template<enum XPgridderiv ideriv>
__device__
realkind XPTextureSpline::BCspline_eval(realkind px,realkind py)
{
	realkind sum  = 0.0f;
	realkind xp;
	realkind xpi;
	realkind yp;
	realkind ypi;
	realkind xp2;
	realkind xpi2;
	realkind yp2;
	realkind ypi2;
	realkind cx;
	realkind cxi;
	realkind hx2;
	realkind cy;
	realkind cyi;
	realkind hy2;
	realkind cxd;
	realkind cxdi;
	realkind cyd;
	realkind cydi;
	realkind sixth = 0.166666666666666667f;
	int2 index;
	unsigned int i;
	unsigned int j;


	px = fmax((origin.x),fmin(px,(origin.x+gridspacing.x*((realkind)(griddims.x)))));
	py = fmax((origin.y),fmin(py,(origin.y+gridspacing.y*((realkind)(griddims.y)))));




	index = findindex(px,py);


	i = index.x;
	j = index.y;

	realkind hxi = 1.0f/gridspacing.x;
	realkind hyi = 1.0f/gridspacing.y;

	realkind hx = gridspacing.x;
	realkind hy = gridspacing.y;

	xp = (px-origin.x)/gridspacing.x;
	yp = (py-origin.y)/gridspacing.y;

	xp = (xp-((realkind)index.x));
	yp = (yp-((realkind)index.y));

	xpi=1.0f-xp;
	xp2=xp*xp;
	xpi2=xpi*xpi;

	cx=xp*(xp2-1.0f);
	cxi=xpi*(xpi2-1.0f);
	cxd = 3.0f*xp2-1.0f;
	cxdi= -3.0f*xpi2+1.0f;
	hx2=hx*hx;

	ypi=1.0f-yp;
	yp2=yp*yp;
	ypi2=ypi*ypi;

	cy=yp*(yp2-1.0f);
	cyi=ypi*(ypi2-1.0f);
	cyd = 3.0f*yp2-1.0f;
	cydi= -3.0f*ypi2+1.0f;
	hy2=hy*hy;

	//printf("xp = %f, yp = %f \n", xp,yp);

	switch(ideriv)
	{
	case XPgridderiv_f:

		sum = xpi*(ypi*get_spline(i,j,0)+yp*get_spline(i,j+1,0))+
				   xp*(ypi*get_spline(i+1,j,0)+yp*get_spline(i+1,j+1,0));

		sum += sixth*hx2*(cxi*(ypi*get_spline(i,j,1)+yp*get_spline(i,j+1,1))+
					 cx*(ypi*get_spline(i+1,j,1)+yp*get_spline(i+1,j+1,1)));

		sum += sixth*hy2*(xpi*(cyi*get_spline(i,j,2)+cy*get_spline(i,j+1,2))+
					 xp*(cyi*get_spline(i+1,j,2)+cy*get_spline(i+1,j+1,2)));

		sum += sixth*sixth*hx2*hy2*(cxi*(cyi*get_spline(i,j,3)+cy*get_spline(i,j+1,3))+
					 cx*(cyi*get_spline(i+1,j,3)+cy*get_spline(i+1,j+1,3)));

		break;
	case XPgridderiv_dfdx:

		sum = hxi*(-1.0f*(ypi*get_spline(i,j,0)+yp*get_spline(i,j+1,0))+
					(ypi*get_spline(i+1,j,0)+yp*get_spline(i+1,j+1,0)));

		sum += sixth*hx*(cxdi*(ypi*get_spline(i,j,1)+yp*get_spline(i,j+1,1))+
						cxd*(ypi*get_spline(i+1,j,1)+yp*get_spline(i+1,j+1,1)));

		sum += sixth*hxi*hy2*(-1.0*(cyi*get_spline(i,j,2)+cy*get_spline(i,j+1,2))+
					(cyi*get_spline(i+1,j,2)+cy*get_spline(i+1,j+1,2)));

		sum += sixth*sixth*hx*hy2*(cxdi*(cyi*get_spline(i,j,3)+cy*get_spline(i,j+1,3))+
					cxd*(cyi*get_spline(i+1,j,3)+cy*get_spline(i+1,j+1,3)));

		break;
	case XPgridderiv_dfdy:
        sum = hyi*(xpi*(-1.0f*get_spline(i,j,0)+get_spline(i,j+1,0))+
        			xp*(-1.0f*get_spline(i+1,j,0)+get_spline(i+1,j+1,0)));

        sum += sixth*hx2*hyi*(cxi*(-1.0*get_spline(i,j,1)+get_spline(i,j+1,1))+
            cx*(-1.0f*get_spline(i+1,j,1)+get_spline(i+1,j+1,1)));

        sum += sixth*hy*(xpi*(cydi*get_spline(i,j,2)+cyd*get_spline(i,j+1,2))+
            xp*(cydi*get_spline(i+1,j,2)+cyd*get_spline(i+1,j+1,2)));

        sum += sixth*sixth*hx2*hy*(cxi*(cydi*get_spline(i,j,3)+cyd*get_spline(i,j+1,3))+
            cx*(cydi*get_spline(i+1,j,3)+cyd*get_spline(i+1,j+1,3)));

        break;
	case XPgridderiv_dfdxx:
        sum = (xpi*(ypi*get_spline(i,j,1)+yp*get_spline(i,j+1,1))+
            xp*(ypi*get_spline(i+1,j,1)+yp*get_spline(i+1,j+1,1)));

        sum += sixth*hy2*(
            xpi*(cyi*get_spline(i,j,3)+cy*get_spline(i,j+1,3))+
            xp*(cyi*get_spline(i+1,j,3)+cy*get_spline(i+1,j+1,3)));

        break;
	case XPgridderiv_dfdyy:
        sum=(xpi*(ypi*get_spline(i,j,2)+yp*get_spline(i,j+1,2))+
            xp*(ypi*get_spline(i+1,j,2)+yp*get_spline(i+1,j+1,2)));

        sum += sixth*hx2*(cxi*(ypi*get_spline(i,j,3)+yp*get_spline(i,j+1,3))+
            cx*(ypi*get_spline(i+1,j,3)+yp*get_spline(i+1,j+1,3)));

        break;
	case XPgridderiv_dfdxy:

        sum=hxi*hyi*(get_spline(i,j,0)-get_spline(i,j+1,0)-
        		get_spline(i+1,j,0)+get_spline(i+1,j+1,0));

        sum += sixth*hx*hyi*(cxdi*(-1.0*get_spline(i,j,1)+get_spline(i,j+1,1))+
            cxd*(-get_spline(i+1,j,1)+get_spline(i+1,j+1,1)));

        sum += sixth*hxi*hy*(-(cydi*get_spline(i,j,2)+cyd*get_spline(i,j+1,2))
            +(cydi*get_spline(i+1,j,2)+cyd*get_spline(i+1,j+1,2)));

        sum += sixth*sixth*hx*hy*(cxdi*(cydi*get_spline(i,j,2)+cyd*get_spline(i,j+1,2))+
            cxd*(cydi*get_spline(i+1,j,2)+cyd*get_spline(i+1,j+1,2)));

        break;
	default:
		break;
	}

	return sum;

}

__device__
void XPTextureSpline::shift_local(XPgrid grid_in,int nx,int ny)
{

	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gidy = blockIdx.y*blockDim.y+idy;

	realkind pr = gidx*grid_in.gridspacing.x+grid_in.origin.x;
	realkind pz = gidy*grid_in.gridspacing.y+grid_in.origin.y;

	if(idx == 0)
	{
		gridspacing = grid_in.gridspacing;
		origin.x = pr;
		origin.y = pz;
	}
	__syncthreads();

	if((gidx < nx)&&(gidy < ny))
	{
		spline[idx+(blockDim.x+1)*idy] = grid_in.spline[gidx+grid_in.griddims.x*gidy];
		if(idx == (blockDim.x-1))
		{
			spline[idx+1+(blockDim.x+1)*idy] = grid_in.spline[gidx+1+grid_in.griddims.x*gidy];
		}
		if(idy == (blockDim.y+1))
		{
			spline[idx+(blockDim.x+1)*(idy+1)] = grid_in.spline[gidx+grid_in.griddims.x*(gidy+1)];
		}
	}
	__syncthreads();

}


__global__
void XPTextureSpline_setup_kernel(cudaMatrixr grid_out,cudaMatrixd spline_in,size_t spline_pitch,int nr,int nz)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = threadIdx.x+blockDim.x*blockIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int gidy = idy + blockDim.y*blockIdx.y;

	__shared__ BCspline sdata[256];

	if((gidx<nr)&&(gidy<nz))
	{
		sdata[idx+16*idy][0] = spline_in(0,gidx,gidy);
		sdata[idx+16*idy][1] = spline_in(1,gidx,gidy);
		sdata[idx+16*idy][2] = spline_in(2,gidx,gidy);
		sdata[idx+16*idy][3] = spline_in(3,gidx,gidy);

	}
	__syncthreads();
	if((gidx<nr)&&(gidy<nz))
	{

		grid_out(gidx,gidy,0) = sdata[idx+16*idy][0];
		grid_out(gidx,gidy,1) = sdata[idx+16*idy][1];
		grid_out(gidx,gidy,2) = sdata[idx+16*idy][2];
		grid_out(gidx,gidy,3) = sdata[idx+16*idy][3];

	}




}



__host__
void XPTextureSpline::setup(double* spline_in,realkind xmin,realkind xmax,realkind ymin,realkind ymax)
{
	size_t temp_pitch;

	origin.x = xmin;
	origin.y = ymin;

	dim3 cudaBlockSize(16,16,1);
	dim3 cudaGridSize((griddims.x+16-1)/16,(griddims.y+16-1)/16,1);

	cudaMatrixd spline_temp(4,griddims.x,griddims.y);

	cudaMatrixr data_out_temp(griddims.x,griddims.y,4);


	spline_temp.cudaMatrixcpy(spline_in,cudaMemcpyHostToDevice);

	CUDA_SAFE_KERNEL((XPTextureSpline_setup_kernel<<<cudaGridSize,cudaBlockSize>>>
			(data_out_temp,spline_temp,pitch,griddims.x,griddims.y)));

	fill2DLayered(data_out_temp);

	spline_temp.cudaMatrixFree();
	data_out_temp.cudaMatrixFree();


}

__host__
void XPTextureSpline::fill2DLayered(cudaMatrixr data_in)
{

	int nx = 1;
	int ny = 1;

	char* texrefstring = (char*)malloc(sizeof(char)*30);
	char* texfetchstring = (char*)malloc(sizeof(char)*30);

	int itemp = next_tex2DLayered;
	next_tex2DLayered++;

	sprintf(texrefstring,"texref2DLayered%i",itemp);
	sprintf(texfetchstring,"fetchtexref2DLayeredPtr%i",itemp);

	symbol = texrefstring;

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&pt2Function,texfetchstring,sizeof(texFunctionPtr)));

	nx = griddims.x;
	ny = griddims.y;

	printf(" fill2DLayered nx = %i, ny = %i \n", nx,ny);

	cudaExtent extent = make_cudaExtent(nx,ny,4);

	cudaMemcpy3DParms params = {0};
	params.kind = cudaMemcpyDeviceToDevice;

	params.kind = cudaMemcpyDeviceToDevice;
#ifdef __double_precision
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindSigned);
#else
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<realkind>();
#endif

	CUDA_SAFE_CALL(cudaMalloc3DArray(&cuArray,&desc,extent,cudaArrayLayered));

	params.srcPtr = data_in.getptr();
	params.dstArray = cuArray;
	params.extent = make_cudaExtent(nx,ny,4);

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	cudaDeviceSynchronize();
	const textureReference* texRefPtr;
	CUDA_SAFE_CALL(cudaGetTextureReference(&texRefPtr, symbol));
	cudaChannelFormatDesc channelDesc;
	CUDA_SAFE_CALL(cudaGetChannelDesc(&channelDesc, cuArray));
	CUDA_SAFE_KERNEL(cudaBindTextureToArray(texRefPtr, cuArray, &channelDesc));


}


