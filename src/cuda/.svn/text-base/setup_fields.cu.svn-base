
#include "fieldclass.cuh"



__host__
void Environment::setup_parameters(int* intparams,double* dbleparams)
{
	ntransp_zones = intparams[0];
	zonebdyctr_shift_index = intparams[1];
	nspecies = intparams[2];
	max_species = intparams[3];
	nxi = intparams[4];
	nth = intparams[5];
	nbeams = intparams[6];
	ledge_transp = intparams[7];
	lcenter_transp = intparams[8];
	nzones = intparams[9];

	nthermal_species = intparams[10];
	max_particles = intparams[11];
	phi_ccw = intparams[12];



	midplane_symetry = intparams[13];
	nbeam_zones = intparams[14];

	nbeam_zones_inside = intparams[15];
	nxi_rows = intparams[16];
	nxi_rows_inside = intparams[17];
	nrow_zones = intparams[18];
	last_inside_row = intparams[19];
	n_diffusion_energy_bins = intparams[20];
	ngases = intparams[21];
	nr = intparams[22];
	nz = intparams[23];
	nint = intparams[24];
	next = intparams[25];

	sign_bphi = intparams[29];

	nbsii = intparams[27];
	nbsjj = intparams[28];

	xi_boundary = dbleparams[0];
	theta_body0 = dbleparams[1];
	theta_body1 = dbleparams[2];
	average_weight_factor = dbleparams[3];
	energy_factor = dbleparams[4];
	Rmin = dbleparams[5];
	Rmax = dbleparams[6];
	Zmin = dbleparams[7];
	Zmax = dbleparams[8];

	fppcon = dbleparams[9];
	cxpcon = dbleparams[10];

	cx_cross_sections.max_energy = dbleparams[11];

	cx_cross_sections.minz = intparams[30];
	cx_cross_sections.maxz = intparams[31];

	griddims.x = nr;
	griddims.y = nz;
	gridspacing.x = (Rmax-Rmin)/((float)nr);
	gridspacing.y = (Zmax-Zmin)/((float)nz);



}

__device__
float2 cxnsum_intrp(float my_energy,float* energies,int* last_grid_point,
								  int* nranges,int* npoints,int irange,int idBeam)
{

	int imin;
	int imax;
	int inum;
	int inc;
	float emin;
	float emax;

	int eindex;

	int my_range_sectors = nranges[idBeam]-1;
	int my_total_points = npoints[idBeam];

	float2 result;

	my_energy = min(energies[4*idBeam],max(energies[399+4*idBeam],my_energy));

	irange = max(0,min(my_range_sectors,3));

	if(irange == 0) imin = 0;
	else imin = last_grid_point[irange-1+4*idBeam]-1;

	imax = last_grid_point[irange+4*idBeam]-1;

	emin = energies[imin];
	emax = energies[imax];

	while((my_energy < emin)||(my_energy > emax))
	{
		irange = max(0,min(irange-1,3));

		if(irange == 0) imin = 0;
		else imin = last_grid_point[irange-1+4*idBeam]-1;

		imin = max(0,imin);

		imax = last_grid_point[irange+4*idBeam]-1;
		imax = min(imax,399);

		emin = energies[imin];
		emax = energies[imax];

		if(my_energy < emin) irange -= 1;
		else if(my_energy > emax) irange+=1;
		else break;
	}

	inum = imax - imin;

	inc = min((inum-1),(int)rint(inum*(my_energy-emin)/(emax-emin)));

	eindex = imin+inc;

	result.x = (my_energy-energies[eindex])/(energies[eindex+1]-energies[eindex]);

	result.y = eindex;

	return result;

}

__device__
unsigned int get_data_index(int eidx,int* dimfacts,int dimidx,int dimbidx,int dimbidy,int dimbidz)
{
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.z;
	unsigned int bidy = blockIdx.y;

	unsigned int result = 0;

	result += dimfacts[dimidx]*eidx;
	result += dimfacts[dimbidx]*bidx;
	result += dimfacts[dimbidy]*bidy;
	result += dimfacts[dimbidz]*idBeam;

	return result;
}

__global__
void setup_cross_sections_kernel(cudaMatrixf data_out,double* data_in,
											 int* npoints,int* last_grid_point,int* nranges,
											 double* energies_in,int max_energy_points,
											 float max_energy,
											 int* dimfacts,int dimidx,int dimbidx,int dimbidy,int dimbidz)
{
	unsigned int idx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.z;
	unsigned int bidy = blockIdx.y;
	unsigned int tid = 0;

	float my_energy = ((float)idx)/((float)Max_energy_sectors);

	int irange = idx / (Max_energy_sectors/4);

	unsigned int data_index;
	unsigned int data_index1;

	float2 tempfactor;
	float intrp_factor;
	int eindex;


	float temp_data;


	my_energy = max_energy*(exp10(4*my_energy-4)-0.0001); // Using an exponential function that closely matches the range method

	__shared__ float energies[400];

	while(idx+tid < 400)
	{
		energies[idx+tid]  = energies_in[idx+tid+max_energy_points*idBeam];
		tid += blockDim.x;
	}
	__syncthreads();

	tempfactor  = cxnsum_intrp(my_energy,energies,last_grid_point,nranges,npoints,irange,idBeam);

	intrp_factor = tempfactor.x;
	eindex = rint(tempfactor.y);

	data_index = get_data_index(eindex,dimfacts,dimidx,dimbidx,dimbidy,dimbidz);
	data_index1 = get_data_index(eindex+1,dimfacts,dimidx,dimbidx,dimbidy,dimbidz);

	temp_data = (1-intrp_factor)*data_in[data_index]+intrp_factor*data_in[data_index1];

	data_out(idx,0,bidx+gridDim.x*(bidy+gridDim.y*idBeam)) = temp_data;


}

__host__
void XPCxGrid::setup_cross_section(double* data_in,int* npoints_d,int* last_grid_point_d,int* nranges_d,
																	  double* energies_in_d,float max_energy_in,int dims[4],
																	   int dimidx,int dimbidx,int dimbidy,int dimbidz,int ndims_in)
{
	texturetype = crossSection;
	tdims = 1;
	ndims = ndims_in;


	int dimfacts[6] = {1,1,1,1,1,1};
	int esectors = Max_energy_sectors;
	int data_size = 1;
	int nbnsvmx = dims[dimidx];
	int* dimfacts_d;
	double* data_in_d;
	char* texrefstring = (char*)malloc(sizeof(char)*30);
	char* texfetchstring = (char*)malloc(sizeof(char)*30);

	max_energy = max_energy_in;
	//dims[dimidx] = 1;

	cudaMalloc((void**)&dimfacts_d,6*sizeof(int));

	gridspacing[0] = max_energy/Max_energy_sectors;

	for(int i = 1;i<(ndims+tdims-1);i++)
	{

		dimfacts[i] = dims[i-1]*dimfacts[i-1];
		printf("dimfacts[%i] = %i , dims = %i\n",i,dimfacts[i],dims[i]);
		gridspacing[i] = 1.0;
	}

	// Find a free texture reference and take it

	int itemp = next_tex1DLayered;
	next_tex1DLayered++;

	sprintf(texrefstring,"texref1DLayered%i",itemp);
	sprintf(texfetchstring,"fetchtexref1DLayeredPtr%i",itemp);

	symbol = texrefstring;
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&pt2Function,texfetchstring,sizeof(texFunctionPtr)));

	for(int i=0;i<(ndims+tdims);i++)
		data_size *= dims[i];

	float min_energy = 0.0;
	float energy_limit = max_energy;

	cudaError status;
	dim3 cudaGridSize(dims[dimbidx],dims[dimbidy],dims[dimbidz]);
	dim3 cudaBlockSize(Max_energy_sectors,1,1);

	cudaMatrixf tempdata(esectors,1,cudaGridSize.x*cudaGridSize.y*cudaGridSize.z);
	cudaExtent extent;
	cudaPitchedPtr matrixPtr;

	cudaMemcpy3DParms params = {0};
	params.kind = cudaMemcpyDeviceToDevice;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	printf(" data size = %i \n",data_size);
	cudaMalloc((void**)&data_in_d,data_size*sizeof(double));
	CUDA_SAFE_CALL(cudaMemcpy(data_in_d,data_in,data_size*sizeof(double),cudaMemcpyHostToDevice));

	matrixPtr = tempdata.getptr();

	CUDA_SAFE_CALL(cudaMemcpy(dimfacts_d,dimfacts,4*sizeof(int),cudaMemcpyHostToDevice));

	CUDA_SAFE_KERNEL((setup_cross_sections_kernel<<<cudaGridSize,cudaBlockSize>>>(tempdata,data_in_d,
												 npoints_d,last_grid_point_d,nranges_d,
												 energies_in_d,nbnsvmx,
												 energy_limit,
												 dimfacts_d,dimidx,dimbidx,dimbidy,dimbidz)));

	cudaDeviceSynchronize();

	printf("cudaGridSize = %i, %i, %i \n",cudaGridSize.x,cudaGridSize.y,cudaGridSize.z);
	extent = make_cudaExtent(esectors,0,cudaGridSize.x*cudaGridSize.y*cudaGridSize.z);
	CUDA_SAFE_CALL(cudaMalloc3DArray(&cuArray,&desc,extent,cudaArrayLayered));

	params.srcPtr = matrixPtr;
	params.dstArray = cuArray;
	params.extent = make_cudaExtent(esectors,1,cudaGridSize.x*cudaGridSize.y*cudaGridSize.z);;

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	cudaDeviceSynchronize();
	const textureReference* texRefPtr;
	CUDA_SAFE_CALL(cudaGetTextureReference(&texRefPtr, symbol));
	cudaChannelFormatDesc channelDesc;
	cudaGetChannelDesc(&channelDesc, cuArray);
	CUDA_SAFE_CALL(cudaBindTextureToArray(texRefPtr, cuArray, &channelDesc));

	cudaFree(data_in_d);
	free(texrefstring);
	free(texfetchstring);
	tempdata.cudaMatrixFree();




}

__host__
void Environment::setup_cross_sections(int* nbnsve,int* lbnsve,int* nbnsver,
										double* bnsves,
										double* bnsvtot,double* bnsvexc,
										double* bnsviif,double* bnsvief,double* bnsvizf,
										double* bnsvcxf,double* bbnsvcx,double* bbnsvii,
										double* cxn_thcx_a,double* cxn_thcx_wa,double* cxn_thii_wa,
										double* cxn_thcx_ha,double* cxn_thii_ha,double* cxn_bbcx,
										double* cxn_bbii,double* btfus_dt,double* btfus_d3,
										double* btfus_ddn,double* btfus_ddp,double* btfus_td,
										double* btfus_tt,double* btfus_3d)
{

	int ncross_sections = 22;

	int nbnsvmx = 400;
	int lep1 = ledge_transp+1;
	int nsbeam = nspecies;
	int nfbznsi = nbeam_zones_inside;
	int ng = ngases;
	int cxn_zmin = cx_cross_sections.minz;
	int cxn_zmax = cx_cross_sections.maxz;

	int ndims;
	int dimfacts[6] = {1,1,1,1,1,1};
	int dimidx;
	int dimbidx;
	int dimbidy;
	int dimbidz;
	int esectors = Max_energy_sectors;

	float min_energy = 0.0;
	float energy_limit = cx_cross_sections.max_energy;

	cudaError status;
	cudaExtent extent;
	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(Max_energy_sectors,1,1);

	cudaMatrixf tempdata;
	cudaPitchedPtr matrixPtr;

	cudaMemcpy3DParms params = {0};
	params.kind = cudaMemcpyDeviceToDevice;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	int* npoints_d;
	int* last_grid_point_d;
	int* nranges_d;
	int data_size;
	double* energies_in_d;
	double* data_in_d;

	cudaMalloc((void**)&npoints_d,(nsbeam+1)*sizeof(int));
	cudaMalloc((void**)&last_grid_point_d,4*(nsbeam+1)*sizeof(int));
	cudaMalloc((void**)&nranges_d,(nsbeam+1)*sizeof(int));
	cudaMalloc((void**)&energies_in_d,nbnsvmx*(nsbeam+1)*sizeof(double));

	CUDA_SAFE_CALL(cudaMemcpy(npoints_d,nbnsve,(nsbeam+1)*sizeof(int),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(last_grid_point_d,lbnsve,4*(nsbeam+1)*sizeof(int),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(nranges_d,nbnsver,(nsbeam+1)*sizeof(int),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(energies_in_d,bnsves,nbnsvmx*(nsbeam+1)*sizeof(double),cudaMemcpyHostToDevice));

	// Setup each grid

	// bnsvtot
	ndims = 3;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.thermal_total.griddims[0] = esectors;
	cx_cross_sections.thermal_total.griddims[1] = lep1;
	cx_cross_sections.thermal_total.griddims[2] = nsbeam;

	cx_cross_sections.thermal_total.setup_cross_section(bnsvtot,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// bnsvexc
	ndims = 3;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.excitation_estimate.griddims[0] = esectors;
	cx_cross_sections.excitation_estimate.griddims[1] = lep1;
	cx_cross_sections.excitation_estimate.griddims[2] = nsbeam;

	cx_cross_sections.excitation_estimate.setup_cross_section(bnsvexc,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// bnsviif
	ndims = 3;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.thermal_fraction.griddims[0] = esectors;
	cx_cross_sections.thermal_fraction.griddims[1] = lep1;
	cx_cross_sections.thermal_fraction.griddims[2] = nsbeam;

	cx_cross_sections.thermal_fraction.setup_cross_section(bnsviif,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// bnsvief
	ndims = 3;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.electron_fraction.griddims[0] = esectors;
	cx_cross_sections.electron_fraction.griddims[1] = lep1;
	cx_cross_sections.electron_fraction.griddims[2] = nsbeam;

	cx_cross_sections.electron_fraction.setup_cross_section(bnsvief,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// bnsvizf
	ndims = 3;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.impurity_fraction.griddims[0] = esectors;
	cx_cross_sections.impurity_fraction.griddims[1] = lep1;
	cx_cross_sections.impurity_fraction.griddims[2] = nsbeam;

	cx_cross_sections.impurity_fraction.setup_cross_section(bnsvizf,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	//bnsvcxf
	ndims = 4;
	dimfacts[0] = lep1;
	dimfacts[1] = ng;
	dimfacts[2] = nbnsvmx;
	dimfacts[3] = nsbeam;

	dimidx = 2;
	dimbidx = 0;
	dimbidy = 1;
	dimbidz =3;
	cx_cross_sections.cx_fraction.griddims[0] = esectors;
	cx_cross_sections.cx_fraction.griddims[1] = lep1;
	cx_cross_sections.cx_fraction.griddims[2] = ng;
	cx_cross_sections.cx_fraction.griddims[3] = nsbeam;

	cx_cross_sections.cx_fraction.setup_cross_section(bnsvcxf,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// bbnsvcx
	ndims = 4;
	dimfacts[0] = nfbznsi;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = nsbeam;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 2;
	dimbidz =3;
	cx_cross_sections.beam_beam_cx.griddims[0] = esectors;
	cx_cross_sections.beam_beam_cx.griddims[1] = nfbznsi;
	cx_cross_sections.beam_beam_cx.griddims[2] = nsbeam;
	cx_cross_sections.beam_beam_cx.griddims[3] = nsbeam;

	cx_cross_sections.beam_beam_cx.setup_cross_section(bbnsvcx,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// bbnsvii
	ndims = 4;
	dimfacts[0] = nfbznsi;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = nsbeam;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 2;
	dimbidz =3;
	cx_cross_sections.beam_beam_ii.griddims[0] = esectors;
	cx_cross_sections.beam_beam_ii.griddims[1] = nfbznsi;
	cx_cross_sections.beam_beam_ii.griddims[2] = nsbeam;
	cx_cross_sections.beam_beam_ii.griddims[3] = nsbeam;

	cx_cross_sections.beam_beam_ii.setup_cross_section(bbnsvii,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_thcx_a
	ndims = 3;
	dimfacts[0] = ng;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = nsbeam;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz =2;
	cx_cross_sections.cx_outside_plasma.griddims[0] = esectors;
	cx_cross_sections.cx_outside_plasma.griddims[1] = ng;
	cx_cross_sections.cx_outside_plasma.griddims[2] = nsbeam;

	cx_cross_sections.cx_outside_plasma.setup_cross_section(cxn_thcx_a,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_thcx_wa
	printf(" cxn_thcx_wa(%i,%i,%i,%i)\n",ng,lep1,nbnsvmx,nsbeam);
	ndims = 4;
	dimfacts[0] = ng;
	dimfacts[1] = lep1;
	dimfacts[2] = nbnsvmx;
	dimfacts[3] = nsbeam;

	dimidx = 2;
	dimbidx = 1;
	dimbidy = 0;
	dimbidz =3;
	cx_cross_sections.cx_thcx_wall.griddims[0] = esectors;
	cx_cross_sections.cx_thcx_wall.griddims[1] = lep1;
	cx_cross_sections.cx_thcx_wall.griddims[2] = ng;
	cx_cross_sections.cx_thcx_wall.griddims[3] = nsbeam;

	cx_cross_sections.cx_thcx_wall.setup_cross_section(cxn_thcx_wa,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_thii_wa
	printf(" cxn_thii_wa(%i,%i,%i,%i)\n",ng,lep1,nbnsvmx,nsbeam);
	ndims = 4;
	dimfacts[0] = ng;
	dimfacts[1] = lep1;
	dimfacts[2] = nbnsvmx;
	dimfacts[3] = nsbeam;

	dimidx = 2;
	dimbidx = 1;
	dimbidy = 0;
	dimbidz =3;
	cx_cross_sections.cx_thii_wall.griddims[0] = esectors;
	cx_cross_sections.cx_thii_wall.griddims[1] = lep1;
	cx_cross_sections.cx_thii_wall.griddims[2] = ng;
	cx_cross_sections.cx_thii_wall.griddims[3] = nsbeam;

	cx_cross_sections.cx_thii_wall.setup_cross_section(cxn_thii_wa,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_thcx_ha
	printf(" cxn_thcx_ha(%i,%i,%i,%i)\n",ng,nfbznsi,nbnsvmx,nsbeam);
	ndims = 4;
	dimfacts[0] = ng;
	dimfacts[1] = nfbznsi;
	dimfacts[2] = nbnsvmx;
	dimfacts[3] = nsbeam;

	dimidx = 2;
	dimbidx = 1;
	dimbidy = 0;
	dimbidz = 3;
	cx_cross_sections.cx_thcx_halo.griddims[0] = esectors;
	cx_cross_sections.cx_thcx_halo.griddims[1] = nfbznsi;
	cx_cross_sections.cx_thcx_halo.griddims[2] = ng;
	cx_cross_sections.cx_thcx_halo.griddims[3] = nsbeam;

	cx_cross_sections.cx_thcx_halo.setup_cross_section(cxn_thcx_ha,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_thii_ha
	printf(" cxn_thii_ha(%i,%i,%i,%i)\n",ng,nfbznsi,nbnsvmx,nsbeam);
	ndims = 4;
	dimfacts[0] = ng;
	dimfacts[1] = nfbznsi;
	dimfacts[2] = nbnsvmx;
	dimfacts[3] = nsbeam;

	dimidx = 2;
	dimbidx = 1;
	dimbidy = 0;
	dimbidz = 3;
	cx_cross_sections.cx_thii_halo.griddims[0] = esectors;
	cx_cross_sections.cx_thii_halo.griddims[1] = nfbznsi;
	cx_cross_sections.cx_thii_halo.griddims[2] = ng;
	cx_cross_sections.cx_thii_halo.griddims[3] = nsbeam;

	cx_cross_sections.cx_thii_halo.setup_cross_section(cxn_thii_ha,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_bbcx
	printf(" cxn_bbcx \n");
	ndims = 3;
	dimfacts[0] = cxn_zmax-cxn_zmin+1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = cxn_zmax-cxn_zmin+1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.cx_thcx_beam_beam.griddims[0] = esectors;
	cx_cross_sections.cx_thcx_beam_beam.griddims[1] = cxn_zmax-cxn_zmin+1;
	cx_cross_sections.cx_thcx_beam_beam.griddims[2] = cxn_zmax-cxn_zmin+1;

	cx_cross_sections.cx_thcx_beam_beam.origin[0] = 0;
	cx_cross_sections.cx_thcx_beam_beam.origin[1] = cxn_zmin-1;
	cx_cross_sections.cx_thcx_beam_beam.origin[2] = cxn_zmin-1;

	cx_cross_sections.cx_thcx_beam_beam.setup_cross_section(cxn_bbcx,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// cxn_bbii
	printf(" cxn_bbii \n");
	ndims = 3;
	dimfacts[0] = cxn_zmax-cxn_zmin+1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = cxn_zmax-cxn_zmin+1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.cx_thii_beam_beam.griddims[0] = esectors;
	cx_cross_sections.cx_thii_beam_beam.griddims[1] = cxn_zmax-cxn_zmin+1;
	cx_cross_sections.cx_thii_beam_beam.griddims[2] = cxn_zmax-cxn_zmin+1;

	cx_cross_sections.cx_thii_beam_beam.origin[0] = 0;
	cx_cross_sections.cx_thii_beam_beam.origin[1] = cxn_zmin-1;
	cx_cross_sections.cx_thii_beam_beam.origin[2] = cxn_zmin-1;

	cx_cross_sections.cx_thii_beam_beam.setup_cross_section(cxn_bbii,npoints_d,last_grid_point_d,nranges_d,
																					energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);
// Don't need these for now
/*
	// btfus_dt
	printf(" btfus_dt \n");
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_dt.griddims[0] = esectors;
	cx_cross_sections.btfus_dt.griddims[1] = lep1;

	cx_cross_sections.btfus_dt.setup_cross_section(btfus_dt,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// btfus_d3
	printf(" btfus_d3 \n");
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_d3.griddims[0] = esectors;
	cx_cross_sections.btfus_d3.griddims[1] = lep1;

	cx_cross_sections.btfus_d3.setup_cross_section(btfus_d3,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// btfus_ddn
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_ddn.griddims[0] = esectors;
	cx_cross_sections.btfus_ddn.griddims[1] = lep1;

	cx_cross_sections.btfus_ddn.setup_cross_section(btfus_ddn,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// btfus_ddp
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_ddp.griddims[0] = esectors;
	cx_cross_sections.btfus_ddp.griddims[1] = lep1;

	cx_cross_sections.btfus_ddp.setup_cross_section(btfus_ddp,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// btfus_td
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_td.griddims[0] = esectors;
	cx_cross_sections.btfus_td.griddims[1] = lep1;

	cx_cross_sections.btfus_td.setup_cross_section(btfus_td,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// btfus_tt
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_tt.griddims[0] = esectors;
	cx_cross_sections.btfus_tt.griddims[1] = lep1;

	cx_cross_sections.btfus_tt.setup_cross_section(btfus_tt,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);

	// btfus_3d
	ndims = 2;
	dimfacts[0] = lep1;
	dimfacts[1] = nbnsvmx;
	dimfacts[2] = 1;
	dimfacts[3] = 1;

	dimidx = 1;
	dimbidx = 0;
	dimbidy = 3;
	dimbidz = 2;
	cx_cross_sections.btfus_3d.griddims[0] = esectors;
	cx_cross_sections.btfus_3d.griddims[1] = lep1;

	cx_cross_sections.btfus_3d.setup_cross_section(btfus_3d,npoints_d,last_grid_point_d,nranges_d,
																			energies_in_d,energy_limit,dimfacts,dimidx,dimbidx,dimbidy,dimbidz,ndims);


*/
}

__host__
void XPTextureGrid::fill2DLayered(double* data_in,enum XPgridlocation location = XPgridlocation_host)
{
	int nx = 1;
	int ny = 1;
	int nz = 1;

	nx = griddims[0];

	if(ndims+tdims > 1)
		ny = griddims[1];

	if(ndims+tdims >2)
	{
		for(int i = 2;i<(ndims+tdims);i++)
		{
			nz *= griddims[i];
		}
	}

	unsigned int nelements = nx*ny*nz;

	double* data_in_d;
	float* data_temp_d;

	cudaError status;

	cudaExtent extent = make_cudaExtent(nx,ny,nz);

	cudaMemcpy3DParms params = {0};
	params.kind = cudaMemcpyDeviceToDevice;

	cudaMalloc((void**)&data_temp_d,nelements*sizeof(float));

	if(location == XPgridlocation_host)
	{
		// data not in device memory
		cudaMalloc((void**)&data_in_d,nelements*sizeof(double));

		cudaMemcpy(data_in_d,data_in,nelements*sizeof(double),cudaMemcpyHostToDevice);
	}
	else
	{
		// data already in device memory
		data_in_d = data_in;
	}


	cudaMemcpydoubletofloat(data_temp_d,data_in_d,nelements);





	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	status = cudaMalloc3DArray(&cuArray,&desc,extent,cudaArrayLayered);

	params.srcPtr.ptr = (void**)data_temp_d;
	params.srcPtr.pitch = nx*sizeof(float);
	params.srcPtr.xsize = nx;
	params.srcPtr.ysize = ny;
	params.dstArray = cuArray;
	params.extent = extent;

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	cudaDeviceSynchronize();
	const textureReference* texRefPtr;
	cudaGetTextureReference(&texRefPtr, symbol);
	cudaChannelFormatDesc channelDesc;
	cudaGetChannelDesc(&channelDesc, cuArray);
	CUDA_SAFE_CALL(cudaBindTextureToArray(texRefPtr, cuArray, &channelDesc));

	cudaFree(data_in_d);
	cudaFree(data_temp_d);
}

__host__
void XPTextureGrid::fill2D(cudaMatrixf data_in)
{

	texturetype = XPtex2D;

	int nx = 1;
	int ny = 1;

	char* texrefstring = (char*)malloc(sizeof(char)*25);
	char* texfetchstring = (char*)malloc(sizeof(char)*25);

	int itemp = next_tex2D;
	next_tex2D++;

	sprintf(texrefstring,"texref2D%i",itemp);
	sprintf(texfetchstring,"fetchtexref2DPtr%i",itemp);

	symbol = texrefstring;

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&pt2Function,texfetchstring,sizeof(texFunctionPtr)));

	nx = griddims[0];
	ny = griddims[1];

	printf(" fill2D nx = %i, ny = %i \n", nx,ny);

	cudaError status;

	cudaExtent extent = make_cudaExtent(nx,ny,0);

	cudaMemcpy3DParms params = {0};
	params.kind = cudaMemcpyDeviceToDevice;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	CUDA_SAFE_CALL(cudaMalloc3DArray(&cuArray,&desc,extent));

	params.srcPtr = data_in.getptr();
	params.dstArray = cuArray;
	params.extent = make_cudaExtent(nx,ny,1);

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	cudaDeviceSynchronize();
	const textureReference* texRefPtr;
	cudaGetTextureReference(&texRefPtr, symbol);
	cudaChannelFormatDesc channelDesc;
	cudaGetChannelDesc(&channelDesc, cuArray);
	CUDA_SAFE_CALL(cudaBindTextureToArray(texRefPtr, cuArray, &channelDesc));


}

__host__
void XPTextureGrid::fill1DLayered(double* data_in,enum XPgridlocation location = XPgridlocation_host)
{
	int nx = 1;
	int ny = 1;

	nx = griddims[0];

	if(ndims+tdims > 1)
	{
		for(int i = 1;i<(ndims+tdims);i++)
		{
			ny *= griddims[i];
		}
	}

	unsigned int nelements = nx*ny;

	double* data_in_d;
	float* data_temp_d;

	cudaError status;

	cudaExtent extent = make_cudaExtent(nx,0,ny);

	cudaMemcpy3DParms params = {0};
	params.kind = cudaMemcpyDeviceToDevice;

	cudaMalloc((void**)&data_temp_d,nelements*sizeof(float));

	if(location == XPgridlocation_host)
	{
		// data not in device memory
		cudaMalloc((void**)&data_in_d,nelements*sizeof(double));

		cudaMemcpy(data_in_d,data_in,nelements*sizeof(double),cudaMemcpyHostToDevice);
	}
	else
	{
		// data already in device memory
		data_in_d = data_in;
	}

	cudaMemcpydoubletofloat(data_temp_d,data_in_d,nelements);

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	status = cudaMalloc3DArray(&cuArray,&desc,extent,cudaArrayLayered);

	params.srcPtr.ptr = (void**)data_temp_d;
	params.srcPtr.pitch = nx*sizeof(float);
	params.srcPtr.xsize = nx;
	params.srcPtr.ysize = 1;
	params.dstArray = cuArray;
	params.extent = make_cudaExtent(nx,1,ny);

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	cudaDeviceSynchronize();
	const textureReference* texRefPtr;
	cudaGetTextureReference(&texRefPtr, symbol);
	cudaChannelFormatDesc channelDesc;
	cudaGetChannelDesc(&channelDesc, cuArray);
	CUDA_SAFE_CALL(cudaBindTextureToArray(texRefPtr, cuArray, &channelDesc));

	cudaFree(data_in_d);
	cudaFree(data_temp_d);
}



__global__
void map_transp_zones(cudaMatrixf transp_zone,XPTextureGrid xi_map,
										int lcenter,int lep1,int nzones,int nr,int nz)
{
	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gidy = blockIdx.y*blockDim.y+idy;

	float temp_transp_zone;
	float xi;

	if((gidx < nr)&&(gidy < nz))
	{
		xi = xi_map(gidx,gidy);
		temp_transp_zone = lcenter+xi*nzones;
		temp_transp_zone = fmin((float)lep1,temp_transp_zone);

		transp_zone(gidx,gidy) = temp_transp_zone;

	}
}

__global__
void map_beam_zones(cudaMatrixf beam_zone,XPTextureGrid xi_map,XPTextureGrid theta_map,
										int lcenter,int nznbmr,double xminbm,simple_XPgrid<int,1> nthzsm,
										int nlsym2b,int nrow_zones,double thbdy0,int nr,int nz)
{
	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gidy = blockIdx.y*blockDim.y+idy;

	float xi;
	float theta;
	float temp_beam_zone;
	float ibrf;
	int ibr0;
	int ibr1;
	int ibr;
	int inz0;
	int inz;
	int inz1;

	float inzf;
	float inz0f;

	if((gidx < nr)&&(gidy < nz))
	{
		xi = xi_map(gidx,gidy);
		theta = theta_map(gidx,gidy);

		ibrf = (lcenter+nznbmr*(xi/xminbm));

		ibr0 = max(1,min((int)rint(ibrf)-1,nrow_zones));
		inz0 = nthzsm(ibr-1);
		ibr = max(1,min((int)rint(ibrf),nrow_zones));
		inz = nthzsm(ibr-1);
		ibr1 = max(1,min((int)rint(ibrf)+1,nrow_zones));
		inz1 = nthzsm(ibr1-1);

		inz0f = (inz-inz0)*(ibrf-ibr0)+inz0;
		inzf = (inz1-inz)*(ibrf-ibr)+inz;

		if(nlsym2b)
			temp_beam_zone = min(inzf,(inz0f+1+(abs(theta)/pi)*(inzf-inz0f)));
		else
			temp_beam_zone = min(inzf,(inz0f+1+((theta-thbdy0)*0.5/pi)*(inzf-inz0f)));

		beam_zone(gidx,gidy) = temp_beam_zone;

	}
}


__host__
void Environment::setup_transp_zones(void)
{
	cudaError status;
	dim3 cudaGridSize((nr+16-1)/16,(nz+16-1)/16,1);
	dim3 cudaBlockSize(16,16,1);

	cudaMatrixf temp_data(nr,nz);

	int lcenter = lcenter_transp;;
	int nznbmr = nxi_rows;
	int lep1 = ledge_transp+1;
	int nlsym2b = midplane_symetry;

	double xminbm = xi_boundary;
	double thbdy0 = theta_body0;

	transp_zone.griddims[0] = nr;
	transp_zone.griddims[1] = nz;
	transp_zone.origin[0] = Rmin;
	transp_zone.origin[1] = Zmin;
	transp_zone.gridspacing[0] = (Rmax-Rmin)/nr;
	transp_zone.gridspacing[1] = (Zmax-Zmin)/nz;

	beam_zone.griddims[0] = nr;
	beam_zone.griddims[1] = nz;
	beam_zone.origin[0] = Rmin;
	beam_zone.origin[1] = Zmin;
	beam_zone.gridspacing[0] = (Rmax-Rmin)/nr;
	beam_zone.gridspacing[1] = (Zmax-Zmin)/nz;

	CUDA_SAFE_KERNEL((map_transp_zones<<<cudaGridSize,cudaBlockSize>>>(temp_data,
																				Xi_map,lcenter,lep1,nzones,nr,nz)));
	cudaDeviceSynchronize();

	transp_zone.fill2D(temp_data);

	CUDA_SAFE_KERNEL((map_beam_zones<<<cudaGridSize,cudaBlockSize>>>(temp_data,
																				Xi_map,Theta_map,lcenter,nznbmr,xminbm,
																				ntheta_row_zones,nlsym2b,nrow_zones,thbdy0,
																				nr,nz)));
	cudaDeviceSynchronize();

	beam_zone.fill2D(temp_data);

	cudaDeviceSynchronize();

	temp_data.cudaMatrixFree();

}


__host__
void Environment::allocate_Grids(void)
{

	int2 griddims_out;
	int mj = ntransp_zones;
	int mig = nthermal_species;
	int mimxbz = nbeam_zones;
	int mimxbzf = nbeam_zones_inside;
	int miz = zonebdyctr_shift_index;
	int mibs = max_species;
	int mib = nbeams;
	int ndifbe = n_diffusion_energy_bins;



	griddims_out.x = nr;
	griddims_out.y = nz;

	xigrid.cudaMatrix_allocate(nxi,1,1);
	thgrid.cudaMatrix_allocate(nth,1,1);

	Psispline.allocate(griddims_out,XPgridlocation_device);
	gspline.allocate(griddims_out,XPgridlocation_device);
	Phispline.allocate(griddims_out,XPgridlocation_device);

	rspline.allocate(nth,nint);
	rsplinex.allocate(nth,next);
	zspline.allocate(nth,nint);
	zsplinex.allocate(nth,next);

	Xi_bloated.allocate(lcenter_transp+xi_boundary*nzones+1);
	ntheta_row_zones.allocate(nrow_zones);

	background_density.allocate(mj,mig,miz);
	omega_wall_neutrals.allocate(mig,mj);
	omega_thermal_neutrals.allocate(mig,mimxbz);
	beamcx_neutral_density.allocate(mimxbz,mibs);
	beamcx_neutral_velocity.allocate(mimxbz,mibs);
	beamcx_neutral_energy.allocate(mimxbz,mibs);
	species_atomic_number.allocate(mibs);
	grid_zone_volume.allocate(mimxbz);
	beam_1stgen_neutral_density2d.allocate(mib,3,2,mimxbzf);

	injection_rate.allocate(mib);

	beam_ion_initial_velocity.allocate(3,mib);
	beam_ion_velocity_direction.allocate(mib,3,2,mimxbzf);

	toroidal_beam_velocity.allocate(mimxbz,mibs);
	average_beam_weight.allocate(mj,mibs);

	is_fusion_product.allocate(mibs);

	electron_temperature.allocate(mj,miz);
	ion_temperature.allocate(mj,miz);
	injection_energy.allocate(mibs);
	FPcoeff_arrayC.allocate(mj,mibs,4);
	FPcoeff_arrayD.allocate(mj,mibs,4);
	FPcoeff_arrayE.allocate(mj,mibs,4);

	loop_voltage.allocate(mj);
	current_shielding.allocate(mj);
	thermal_velocity.allocate(mj,mibs);

	adif_multiplier.allocate(ndifbe);
	adif_energies.allocate(ndifbe);

}

template<typename T,int dims>
__host__
void simple_XPgrid<T,dims>::copyFromDouble(double* data_in,enum cudaMemcpyKind kind)
{
	double* data_in_d;
	cudaExtent extent = data.getdims();
	int n_elements = extent.width*extent.height*extent.depth/sizeof(T);
	if(kind == cudaMemcpyHostToDevice)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&data_in_d,n_elements*sizeof(double)));
		CUDA_SAFE_CALL(cudaMemcpy(data_in_d,data_in,n_elements*sizeof(double),cudaMemcpyHostToDevice));

	}
	else if(kind == cudaMemcpyDeviceToDevice)
	{
		data_in_d = data_in;
	}
	else
	{
		return;
	}

	cudaMemcpydoubletoMatrixf(data,data_in_d);

	return;
}

__host__
void Environment::setup_fields(double** data_in,int** idata_in)
{
	double* Psispline_in = data_in[0];
	double* gspline_in = data_in[1];
	double* Phispline_in = data_in[2];
	double* Xi_map_in = data_in[3];
	double* Theta_map_in = data_in[4];
	double* omegag = data_in[5];
	double* rhob = data_in[6];
	double* owall0 = data_in[7];
	double* ovol02 = data_in[8];
	double* bn0x2p = data_in[9];
	double* bv0x2p = data_in[10];
	double* be0x2p = data_in[11];
	double* xzbeams = data_in[12];
	double* bmvol = data_in[13];
	double* bn002 = data_in[14];
	double* xninja = data_in[15];
	double* viona = data_in[16];
	double* vcxbn0 = data_in[17];
	double* vbtr2p = data_in[18];
	double* wbav = data_in[19];

	double* xiblo = data_in[20];
	double* dxi1 = data_in[21];
	double* dxi2 = dxi1+ntransp_zones;

	double* te = data_in[22];
	double* ti = data_in[23];
	double* einjs = data_in[24];
	double* cfa = data_in[25];
	double* dfa = data_in[26];
	double* efa = data_in[27];
	double* vpoh = data_in[28];
	double* xjbfac = data_in[29];
	double* vmin = data_in[30];
	double* velb_fi = data_in[31];
	double* difb_fi = data_in[32];
	double* velb_bi = data_in[33];
	double* difb_bi = data_in[34];
	double* fdifbe = data_in[35];
	double* edifbe = data_in[36];

	double* rspline_in = data_in[37];
	double* zspline_in = data_in[38];
	double* rsplinex_in = data_in[39];
	double* zsplinex_in = data_in[40];

	double* xigrid_in = data_in[41];
	double* thgrid_in = data_in[42];

	double* spacingparams = data_in[43];
	double* limiter_map_in = data_in[44];
	double* ympx_in = data_in[45];

	int* nlfprod = idata_in[0];
	int* nthzsm = idata_in[1];


	double* Xi_map_d;
	double* Theta_map_d;
	double* xigrid_in_d;
	double* thgrid_in_d;
	double* limiter_map_in_d;

	float simple_gridspacing[6] = {1,1,1,1,1,1};
	float simple_origins[6] = {0,0,0,0,0,0};

	int tex_griddims[2] = {nr,nz};
	float tex_gridspacing[2] = {(Rmax-Rmin)/nr,(Zmax-Zmin)/nz};
	float tex_origin[2] = {Rmin,Zmin};

	size_t free = 0;
	size_t total = 0;

	transp_zone.tdims = 2;
	beam_zone.tdims = 2;
	Xi_map.tdims = 2;
	Theta_map.tdims = 2;
	rotation.tdims = 2;
	fusion_anomalous_radialv.tdims = 2;
	fusion_anomalous_diffusion.tdims = 2;
	beam_anomalous_radialv.tdims = 2;
	beam_anomalous_diffusion.tdims = 2;
	dxi_spacing1.tdims = 2;
	dxi_spacing2.tdims = 2;
	limiter_map.tdims = 2;
	Ymidplane.tdims = 2;

	transp_zone.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	beam_zone.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	Xi_map.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	Theta_map.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	rotation.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	fusion_anomalous_radialv.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	fusion_anomalous_diffusion.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	beam_anomalous_radialv.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	beam_anomalous_diffusion.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	dxi_spacing1.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	dxi_spacing2.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	limiter_map.setup_dims(tex_griddims,tex_gridspacing,tex_origin);
	Ymidplane.setup_dims(tex_griddims,tex_gridspacing,tex_origin);


	cudaMalloc((void**)&limiter_map_in_d,nr*nz*sizeof(double));
	cudaMalloc((void**)&xigrid_in_d,nxi*sizeof(double));
	cudaMalloc((void**)&thgrid_in_d,nth*sizeof(double));

	cudaMalloc((void**)&Xi_map_d,nr*nz*sizeof(double));
	cudaMalloc((void**)&Theta_map_d,nr*nz*sizeof(double));

	CUDA_SAFE_CALL(cudaMemcpy(Xi_map_d,Xi_map_in,nr*nz*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(Theta_map_d,Theta_map_in,nr*nz*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(limiter_map_in_d,limiter_map_in,nr*nz*sizeof(double),cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(xigrid_in_d,xigrid_in,nxi*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(thgrid_in_d,thgrid_in,nth*sizeof(double),cudaMemcpyHostToDevice));

	cudaMatrixf Xi_map_temp(nr,nz);
	cudaMatrixf Theta_map_temp(nr,nz);
	cudaMatrixf limiter_map_temp(nr,nz);

	allocate_Grids();

	ntheta_row_zones.setupi(nthzsm,simple_gridspacing,simple_origins);

	cudaMemGetInfo(&free,&total);
	printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free)/(1<<10),(int)(total-free)/(1<<10));

	cudaMemcpydoubletoMatrixf(xigrid,xigrid_in_d);
	cudaMemcpydoubletoMatrixf(thgrid,thgrid_in_d);


	printf("setting up Psispline \n");
	Psispline.setup(Psispline_in,Rmin,Rmax,Zmin,Zmax);
	printf("setting up gspline \n");
	gspline.setup(gspline_in,Rmin,Rmax,Zmin,Zmax);
	printf("setting up phispline \n");
	Phispline.setup(Phispline_in,Rmin,Rmax,Zmin,Zmax);

	printf("setting up rspline \n");
	rspline.setup(rspline_in,xigrid,thgrid,
				nbsii,nbsjj,spacingparams[0],spacingparams[2],
	  	  	  spacingparams[1],spacingparams[3],
	  	  	  spacingparams[4],spacingparams[5]);

	printf("setting up rsplinex \n");
	rsplinex.setup(rsplinex_in,xigrid,thgrid,
				nbsii,nbsjj,spacingparams[0],spacingparams[2],
	  	  	  spacingparams[1],spacingparams[3],
	  	  	  spacingparams[4],spacingparams[5]);

	printf("setting up zspline \n");
	zspline.setup(zspline_in,xigrid,thgrid,
				nbsii,nbsjj,spacingparams[0],spacingparams[2],
	  	  	  spacingparams[1],spacingparams[3],
	  	  	  spacingparams[4],spacingparams[5]);

	printf("setting up zsplinex \n");
	zsplinex.setup(zsplinex_in,xigrid,thgrid,
				nbsii,nbsjj,spacingparams[0],spacingparams[2],
	  	  	  spacingparams[1],spacingparams[3],
	  	  	  spacingparams[4],spacingparams[5]);

	setupBfield();

	cudaMemcpydoubletoMatrixf(Xi_map_temp,Xi_map_d);
	cudaMemcpydoubletoMatrixf(Theta_map_temp,Theta_map_d);
	cudaMemcpydoubletoMatrixf(limiter_map_temp,limiter_map_in_d);

	printf("setting up Xi_map \n");
	Xi_map.fill2D(Xi_map_temp);
	printf("setting up Theta_map \n");
	Theta_map.fill2D(Theta_map_temp);
	printf("setting up limiter_map \n");
	limiter_map.fill2D(limiter_map_temp);

	Xi_map_temp.cudaMatrixFree();
	Theta_map_temp.cudaMatrixFree();
	limiter_map_temp.cudaMatrixFree();
	cudaFree(Xi_map_d);
	cudaFree(Theta_map_d);
	cudaFree(limiter_map_in_d);

	cudaFree(xigrid_in_d);
	cudaFree(thgrid_in_d);

	printf("setting up transp zones \n");
	setup_transp_zones();

	Xi_bloated.setup(xiblo,simple_gridspacing,simple_origins);

	cudaMatrixf dxi_spacing1_temp = mapTransp_data_to_RZ(dxi1);
	dxi_spacing1.fill2D(dxi_spacing1_temp);
	dxi_spacing1_temp.cudaMatrixFree();

	cudaMatrixf dxi_spacing2_temp = mapTransp_data_to_RZ(dxi2);
	dxi_spacing2.fill2D(dxi_spacing2_temp);
	dxi_spacing2_temp.cudaMatrixFree();

	cudaMatrixf rotation_temp = mapTransp_data_to_RZ(omegag);
	rotation.fill2D(rotation_temp);
	rotation_temp.cudaMatrixFree();

	cudaMatrixf velb_fi_temp = mapTransp_data_to_RZ(velb_fi);
	fusion_anomalous_radialv.fill2D(velb_fi_temp);
	velb_fi_temp.cudaMatrixFree();

	cudaMatrixf difb_fi_temp = mapTransp_data_to_RZ(difb_fi);
	fusion_anomalous_diffusion.fill2D(difb_fi_temp);
	difb_fi_temp.cudaMatrixFree();

	cudaMatrixf velb_bi_temp = mapTransp_data_to_RZ(velb_bi);
	beam_anomalous_radialv.fill2D(velb_bi_temp);
	velb_bi_temp.cudaMatrixFree();

	cudaMatrixf difb_bi_temp = mapTransp_data_to_RZ(difb_bi);
	beam_anomalous_diffusion.fill2D(difb_bi_temp);
	difb_bi_temp.cudaMatrixFree();

	cudaMatrixf ympx_temp = mapTransp_data_to_RZ(ympx_in,1,1);
	Ymidplane.fill2D(ympx_temp);
	ympx_temp.cudaMatrixFree();


	background_density.setup(rhob,simple_gridspacing,simple_origins);
	omega_wall_neutrals.setup(owall0,simple_gridspacing,simple_origins);
	beamcx_neutral_density.setup(bn0x2p,simple_gridspacing,simple_origins);
	beamcx_neutral_velocity.setup(bv0x2p,simple_gridspacing,simple_origins);
	beamcx_neutral_energy.setup(be0x2p,simple_gridspacing,simple_origins);
	species_atomic_number.setup(xzbeams,simple_gridspacing,simple_origins);
	grid_zone_volume.setup(bmvol,simple_gridspacing,simple_origins);
	beam_1stgen_neutral_density2d.setup(bn002,simple_gridspacing,simple_origins);
	injection_rate.setup(xninja,simple_gridspacing,simple_origins);
	beam_ion_initial_velocity.setup(viona,simple_gridspacing,simple_origins);
	beam_ion_velocity_direction.setup(vcxbn0,simple_gridspacing,simple_origins);

	toroidal_beam_velocity.setup(vbtr2p,simple_gridspacing,simple_origins);

	average_beam_weight.setup(wbav,simple_gridspacing,simple_origins);

	is_fusion_product.setupi(nlfprod,simple_gridspacing,simple_origins);

	electron_temperature.setup(te,simple_gridspacing,simple_origins);
	ion_temperature.setup(ti,simple_gridspacing,simple_origins);
	injection_energy.setup(einjs,simple_gridspacing,simple_origins);
	FPcoeff_arrayC.setup(cfa,simple_gridspacing,simple_origins);
	FPcoeff_arrayD.setup(dfa,simple_gridspacing,simple_origins);
	FPcoeff_arrayE.setup(efa,simple_gridspacing,simple_origins);
	loop_voltage.setup(vpoh,simple_gridspacing,simple_origins);
	current_shielding.setup(xjbfac,simple_gridspacing,simple_origins);
	thermal_velocity.setup(vmin,simple_gridspacing,simple_origins);
	adif_multiplier.setup(fdifbe,simple_gridspacing,simple_origins);
	adif_energies.setup(edifbe,simple_gridspacing,simple_origins);

}

__global__
void mapTransp_data_to_RZ_kernel(cudaMatrixf data_out,cudaMatrixf data_in,
															 XPTextureGrid Xi_map,simple_XPgrid<float,1> Xi_bloated,
															 int nr,int nz,int ledge,float xi_boundary,int nzones,int lcenter,int bloated)
{
	unsigned int idx = threadIdx.x;
	unsigned int idy = threadIdx.y;
	unsigned int idz = threadIdx.z;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int gidy = idy+blockIdx.y*blockDim.y;
	unsigned int gidz = idz+blockIdx.z*blockDim.z;

	float xi;
	int ngc;
	int ngcx;
	float xingc;
	float r = gidx*Xi_map.gridspacing[0];
	float z = gidy*Xi_map.gridspacing[1];
	float data_out_temp;

	xi = Xi_map(r,z);

	ngcx = rint(lcenter+xi*nzones);
	ngc = min(ledge+1,ngcx);

	xingc = (xi-Xi_bloated((float)ngcx))/(Xi_bloated(float(ngcx+1))-Xi_bloated(float(ngcx)));
	xingc = max(0.0,min(1.0,xingc));

	if(ngc > ledge)
	{
		xingc = 0.0;
	}

	if(!bloated)
	{
		data_out_temp = data_in(ngc,gidz)+xingc*(data_in(ngc+1,gidz)-data_in(ngc,gidz));
	}
	else
	{
		data_out_temp = data_in(ngcx,gidz)+xingc*(data_in(ngcx+1,gidz)-data_in(ngcx,gidz));
	}

	if((gidx<nr)&&(gidy<nz))
	{
		data_out(gidx,gidy,gidz) = data_out_temp;
	}


	return;

}

__host__
cudaMatrixf Environment::mapTransp_data_to_RZ(double* data_in_h,int ndim3,int bloated)
{
	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(16,16,1);
	int data_size;

	if(bloated = 0)
		data_size = ntransp_zones;
	else
		data_size = 2*ntransp_zones;

	cudaMatrixf data_in_temp(data_size,ndim3);


	cudaMemcpydoubletoMatrixf(data_in_temp,data_in_);

	cudaFree(data_in_d);

	cudaMatrixf data_out(nr,nz,ndim3);


	if(ndim3 > 1)
	{
		cudaBlockSize.x = 8;
		cudaBlockSize.y = 8;
		cudaBlockSize.z = 8;
	}

	cudaGridSize.x = (cudaBlockSize.x+nr-1)/cudaBlockSize.x;
	cudaGridSize.y = (cudaBlockSize.y+nz-1)/cudaBlockSize.y;
	cudaGridSize.z = (cudaBlockSize.z+ndim3-1)/cudaBlockSize.z;

	CUDA_SAFE_KERNEL((mapTransp_data_to_RZ_kernel<<<cudaGridSize,cudaBlockSize>>>(
								data_out,data_in_temp,Xi_map,Xi_bloated,nr,nz,ledge_transp,xi_boundary,
								nzones,lcenter_transp,bloated)));
	cudaDeviceSynchronize();

	data_in_temp.cudaMatrixFree();
	return data_out;

}

__global__
void get_extern_plasma_ptr(Environment** plasma_ptr_out)
{
	*plasma_ptr_out = &extern_plasma_d;
}


extern "C" __host__ void setup_gpu_fields_(double* orbrzv_psi,double* orbrzv_g,double* orbrzv_phi,
											  double* Xi_map,double* theta_map,double* omegag,
											  double* rhob,double* owall0,double* ovol02,double* bn0x2p,
											  double* bv0x2p,double* be0x2p,double* xzbeams,
											  double* bmvol,double* bn002,double* xninja,double* viona,
											  double* vcxbn0,double* vbtr2p,double* wbav,double* xiblo,
											  double* dxi,double* te,double* ti, double* einjs,double* cfa,
											  double* dfa,double* efa,double* vpoh,double* xjbfac,double* vmin,
											  double* velb_fi,double* difb_fi,double* velb_bi,double* difb_bi,
											  double* fdifbe,double* edifbe,double* rspl,double* zspl,double* rsplx,
											  double* zsplx,double* xi,double* theta,int* nlfprod,
											  double* limiter_map_in,double* ympx,
											  double* bnsvtot,double* bnsvexc,double* bnsviif,double* bnsvief,
											  double* bnsvizf,double* bnsvcxf,double* bbnsvcx,double* bbnsvii,
											  double* cxn_thcx_a,double* cxn_thcx_wa,double* cxn_thii_wa,
											  double* cxn_thcx_ha,double* cxn_thii_ha,double* cxn_bbcx,
											  double* cxn_bbii,double* btfus_dt,double* btfus_d3,
											  double* btfus_ddn,double* btfus_ddp,double* btfus_td,
											  double* btfus_tt,double* btfus_3d,
											  double* bnsves,int* nbnsve,int* lbnsve,int* nbnsver,int* nthzsm,
											  double* dblparams,int* intparams,double* spacingparams,int* ierr1)
{
	next_tex2D = 0;
	next_tex1DLayered = 0;
	double* double_data[46];
	int* int_data[2];

	double_data[0] = orbrzv_psi;
	double_data[1] = orbrzv_g;
	double_data[2] = orbrzv_phi;
	double_data[3] = Xi_map;
	double_data[4] = theta_map;
	double_data[5] = omegag;
	double_data[6] = rhob;
	double_data[7] = owall0;
	double_data[8] = ovol02;
	double_data[9] = bn0x2p;
	double_data[10] = bv0x2p;
	double_data[11] = be0x2p;
	double_data[12] = xzbeams;
	double_data[13] = bmvol;
	double_data[14] = bn002;
	double_data[15] = xninja;
	double_data[16] = viona;
	double_data[17] = vcxbn0;
	double_data[18] = vbtr2p;
	double_data[19] = wbav;
	double_data[20] = xiblo;
	double_data[21] = dxi;
	double_data[22] = te;
	double_data[23] = ti;
	double_data[24] = einjs;
	double_data[25] = cfa;
	double_data[26] = dfa;
	double_data[27] = efa;
	double_data[28] = vpoh;
	double_data[29] = xjbfac;
	double_data[30] = vmin;
	double_data[31] = velb_fi;
	double_data[32] = difb_fi;
	double_data[33] = velb_bi;
	double_data[34] = difb_bi;
	double_data[35] = fdifbe;
	double_data[36] = edifbe;
	double_data[37] = rspl;
	double_data[38] = zspl;
	double_data[39] = rsplx;
	double_data[40] = zsplx;
	double_data[41] = xi;
	double_data[42] = theta;
	double_data[43] = spacingparams;
	double_data[44] = limiter_map_in;
	double_data[45] = ympx;

	int_data[0] = nlfprod;
	int_data[1] = nthzsm;

	// Setup Grid Parameters
	plasma_h.setup_parameters(intparams,dblparams);
	plasma_h.setup_fields(double_data,int_data);
	plasma_h.setup_cross_sections(nbnsve,lbnsve,nbnsver,bnsves,
			bnsvtot,bnsvexc,
			bnsviif,bnsvief,bnsvizf,
			bnsvcxf,bbnsvcx,bbnsvii,
			cxn_thcx_a,cxn_thcx_wa, cxn_thii_wa,
			cxn_thcx_ha,cxn_thii_ha, cxn_bbcx,
			cxn_bbii,btfus_dt,btfus_d3,
			btfus_ddn,btfus_ddp,btfus_td,
			btfus_tt,btfus_3d);

	Environment** plasma_ptr;
	cudaMalloc((void**)&plasma_ptr,sizeof(Environment*));

	CUDA_SAFE_KERNEL((get_extern_plasma_ptr<<<1,1>>>(plasma_ptr)));

	CUDA_SAFE_CALL(cudaMemcpy(&plasma_d_ptr,plasma_ptr,sizeof(Environment*),cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaMemcpy(plasma_d_ptr,&plasma_h,sizeof(Environment),cudaMemcpyHostToDevice));

}






























