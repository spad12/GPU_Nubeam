
#include "particleclass.cuh"


XPlist particles_done_per_step[16];
int finished_particle_list_counter;

__host__
void append_finished_particle_table(XPlist finished_list)
{
	int i = finished_particle_list_counter;
	particles_done_per_step[i] = finished_list;
	i++;
}

__host__
void update_original_idx_counter(int increment)
{
	char* symbol = "original_idx_counter_d";
	original_idx_counter += increment;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(symbol,&original_idx_counter,sizeof(unsigned int)));
}


/*
__global__
void XPlist_move_shared_kernel(XPlist particles,XPlist particles_old,Environment* plasma_in,
										  cudaMatrixr orbit_error,cudaMatrixui splitting_condition,
										  cudaMatrixi3 blockInfo,int istep,cudaMatrixf px,cudaMatrixf py,
										  cudaMatrixi nptcls_at_step)
{
	unsigned int idx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int gidx = blockInfo(bidx,idBeam).y+idx;
	unsigned int block_start = blockInfo(bidx,idBeam).y;

	//if(idx == 0)
	//	printf("nptcls[%i] = %i\n",idBeam,particles.nptcls[idBeam]);

	uint2 tid;
	int cellindex_old = blockInfo(bidx,idBeam).y;
	realkind mu;
	int nx;
	int ny;
#ifdef Animate_orbits
	if(gidx == 0)
	{
		nptcls_at_step(istep) = particles.nptcls[idBeam];
	}
#endif

	realkind energy0;
	realkind momentum0;
	realkind B;


	// Results of the rk4 calculations;
	realkind3 initialvalues;
	realkind3 result[4];
	// Temporary variable for error calculation
	realkind3 temp;
	int orbflag = 0;
	int nr = plasma_in -> griddims.x;
	int nz = plasma_in -> griddims.y;

	// dt is used a lot, so we'll stick it in shared memory
	__shared__ realkind dt[BLOCK_SIZE];

	// The local particle list, some members will exist in shared memory
	__shared__ XPlist localion;

	__shared__ int2 cell; // The R and Z index of the cell
	__shared__ int2 GridInfo; // the R and Z integer dimensions of the grid
	__shared__ realkind2 origin; // The R and Z coordinates of the bottom right corner of the cell

	// Splines that describe the fields for this cell
	__shared__ XPTextureSpline Psispline;
	__shared__ XPTextureSpline gspline;
	__shared__ XPTextureSpline Phispline;
	__shared__ realkind2 gridspacing;
	__shared__ realkind2 plasma_origin;
	__shared__ BCspline Psispline_local[256];
	__shared__ BCspline gspline_local[256];
	__shared__ BCspline Phispline_local[256];
		//printf("blockstart(%i) = %i nptcls[%i] = %i\n",bidx,blockInfo(bidx,idBeam).y,bidx,blockInfo(bidx,idBeam).z);

	if((block_start)<particles.nptcls[idBeam])
	{


	// Copy data from global memory to shared memory
	localion.shift_local(&particles,block_start);
	__syncthreads();




	if((idx == 0))
	{
		localion.nptcls_max = blockInfo(bidx,idBeam).z;
		Psispline = plasma_in->Psispline;
		gspline = plasma_in -> gspline;
		Phispline = plasma_in -> Phispline;
		cell.x = max(0,(8*((int)floor(localion.nx[0][idx]/8.0))-4));
		cell.y = max(0,(8*((int)floor(localion.ny[0][idx]/8.0))-4));
		origin.x = (cell.x)*(plasma_in -> gridspacing.x)+(plasma_in -> Rmin);
		origin.y = (cell.y)*(plasma_in -> gridspacing.y)+(plasma_in -> Zmin);
		plasma_origin.x = plasma_in -> Rmin;
		plasma_origin.y = plasma_in -> Zmin;
		GridInfo.x = plasma_in -> griddims.x;
		GridInfo.y = plasma_in -> griddims.y;
		gridspacing = plasma_in -> gridspacing;

		// Allocate space in shared memory for shared copy of splines

	}
	//Psispline.allocate_local(Psispline_local);
	//gspline.allocate_local(gspline_local);
	//Phispline.allocate_local(Phispline_local);
	__syncthreads();



	if(idx == 0)
	{
		Psispline.origin = origin;
		gspline.origin = origin;
		Phispline.origin = origin;
		Psispline.gridspacing = gridspacing;
		gspline.gridspacing = gridspacing;
		Phispline.gridspacing = gridspacing;
	}

	tid.x = max(0,min((cell.x+(idx%16)),(GridInfo.x-1)));
	tid.y = max(0,min((cell.y+(idx/16)),(GridInfo.y-1)));

	// Copy Splines to shared memory
	if(idx < 256)
	{
		//printf("Getting Sline @ %i,%i\nfor cell %i,%i\nparticle %i pos %i, %i\n",
		//		tid.x,tid.y,cell.x,cell.y,idx,localion.nx[0][0],localion.ny[0][0]);
		//Psispline.spline[idx] = plasma_in -> Psispline.get_spline(tid.x,tid.y);
		//gspline.spline[idx] = plasma_in -> gspline.get_spline(tid.x,tid.y);
		//Phispline.spline[idx] = plasma_in -> Phispline.get_spline(tid.x,tid.y);
	}


	__syncthreads();
	if(idx < localion.nptcls_max)
		orbflag = localion.orbflag[idx];



	if(orbflag == 1)
	{
#ifdef Animate_orbits
		px(localion.original_idx[idx],idBeam,istep) = localion.px[0][idx];
		py(localion.original_idx[idx],idBeam,istep) = localion.py[0][idx];
#endif

		B = localion.eval_Bmod(Psispline,gspline);
		mu = localion.mu[idx];

		// Figure out what the timestep should be
		dt[idx] = localion.eval_dt(plasma_in);

		//printf("dt = %g\n",dt[idx]);
		initialvalues.x = localion.px[0][idx];
		initialvalues.y = localion.py[0][idx];
		initialvalues.z = localion.vpara[0][idx];
		energy0 = localion.energy[idx];
		momentum0 = localion.momentum[idx];

		//RK1
		for(int i=0;i<3;i++)
		{
			result[i] = localion.XPlist_derivs(Psispline,gspline,Phispline,mu,0);
			//printf("rk4 results(%i,%i) = %g, %g, %g, %g\n",gidx,idBeam,result[i].x,result[i].y,result[i].z,dt[idx]);
			localion.px[0][idx] = initialvalues.x+rk4_mults[i]*dt[idx]*result[i].x;
			localion.py[0][idx] = initialvalues.y+rk4_mults[i]*dt[idx]*result[i].y;
			localion.vpara[0][idx] = initialvalues.z+rk4_mults[i]*dt[idx]*result[i].z;

			localion.nx[0][idx] = (localion.px[0][idx]-plasma_origin.x)/(gridspacing.x);
			localion.ny[0][idx] = (localion.py[0][idx]-plasma_origin.y)/(gridspacing.y);

			if(localion.check_orbit(plasma_in)!=0)
			{
				// If the particle has left the grid, it needs to stop orbiting
				//printf("particles %i left orbit during rk4\n",gidx);
				break;
			}
		}

		if(localion.check_orbit(plasma_in) == 0)
		{
			result[3] = localion.XPlist_derivs(Psispline,gspline,Phispline,mu,0);

			localion.px[0][idx] =initialvalues.x+dt[idx]*(result[0].x+2*result[1].x+2*result[2].x+result[3].x)/6.0;
			localion.py[0][idx] =initialvalues.y+dt[idx]*(result[0].y+2*result[1].y+2*result[2].y+result[3].y)/6.0;
			localion.vpara[0][idx] =initialvalues.z+dt[idx]*(result[0].z+2*result[1].z+2*result[2].z+result[3].z)/6.0;

			localion.nx[0][idx] = (localion.px[0][idx]-plasma_origin.x)/(gridspacing.x);
			localion.ny[0][idx] = (localion.py[0][idx]-plasma_origin.y)/(gridspacing.y);

			result[3] = localion.XPlist_derivs(Psispline,gspline,Phispline,mu,0);

			temp.x = initialvalues.x+dt[idx]*(result[0].x+2*result[1].x+2*result[2].x+result[3].x)/6.0;
			temp.y = initialvalues.y+dt[idx]*(result[0].y+2*result[1].y+2*result[2].y+result[3].y)/6.0;
			temp.z = initialvalues.z+dt[idx]*(result[0].z+2*result[1].z+2*result[2].z+result[3].z)/6.0;

			orbit_error(gidx,0) = (localion.px[0][idx]-temp.x)/6.0;
			orbit_error(gidx,1) = (localion.py[0][idx]-temp.y)/6.0;
			orbit_error(gidx,2) = (localion.vpara[0][idx]-temp.z)/6.0;

			orbit_error(gidx,0) = fabs(orbit_error(gidx,0)/(fabs(initialvalues.x+0.5*dt[idx]*result[0].x))+epsilon);
			orbit_error(gidx,1) = fabs(orbit_error(gidx,1)/(fabs(initialvalues.y+0.5*dt[idx]*result[0].y))+epsilon);
			orbit_error(gidx,2) = fabs(orbit_error(gidx,2)/(fabs(initialvalues.z+0.5*dt[idx]*result[0].z))+epsilon);
		}
		else
		{
			if(idx < localion.nptcls_max)
			{// Particles that have left the grid in the orbit process
				orbit_error(gidx,0) = 0;
				orbit_error(gidx,1) = 0;
				orbit_error(gidx,2) = 0;

				localion.energy[idx] = energy0;
				localion.momentum[idx] = momentum0;

			}
		}


		nx = max(2,min(nr-3,localion.nx[0][idx]));
		ny = max(2,min(nz-3,localion.ny[0][idx]));
		localion.cellindex[0][idx] = zmap(nx,ny);

		localion.time_done[idx] += dt[idx];


		orbit_error(gidx,4) = fabs(localion.energy[idx]-energy0)/(fabs(energy0)+epsilon);
		orbit_error(gidx,5) = fabs(localion.momentum[idx]-momentum0)/(fabs(momentum0)+epsilon);

		temp.x = orbit_error(gidx,0);

		for(int i=1;i<6;i++)
		{
			temp.x = max(temp.x,orbit_error(gidx,i));
		}

		orbit_error(gidx,0) = temp.x;

		//printf("orbit_error(%i,%i) = %f\n",gidx,idBeam,temp.x*100.0);


		if(localion.check_orbit(plasma_in) == 0)
		{
				localion.vperp[0][idx] = sqrt(2*mu*B/(localion.mass[idx]*ZMP));
		}
		else
		{
			localion.orbflag[idx] = 0;
			localion.pexit[idx] = XPlistexit_limiter;
		}

		if(localion.time_done[idx] >= (plasma_in -> delt))
		{
			localion.orbflag[idx] = 0;
			localion.pexit[idx] = XPlistexit_time;
			//printf("time done = %f out of %f \n",localion.time_done[idx],plasma_in -> delt);
		}

		if(localion.pexit[idx] > 2)
		{
			localion.orbflag[idx] = 0;

		}

		if(!localion.orbflag[idx])
		{
			//printf("orbit exit = %i \n",localion.pexit[idx]);
		}



	}



	if(gidx < particles.nptcls[idBeam])
		splitting_condition(gidx,idBeam) = 1-localion.orbflag[idx];


	}
}

__host__
void move_particles_shared(Environment* plasma_d,XPlist* particles,XPlist* particles_old,XPlist* particles_done,int istep)
{

	cudaError status;

	XPlist particles_done_this_step;

	dim3 cudaGridSize(1,particles->nspecies,1);
	dim3 cudaBlockSize(BLOCK_SIZE,1,1);

	int nspecies = particles->nspecies;
	int nptcls = particles->nptcls_max;
	cudaMatrixr orbit_error(particles->nptcls_max,nspecies);
	cudaMatrixui splitting_condition(nptcls,nspecies);

	int2 Pgrid_i_dims = plasma_h.griddims;
	realkind2 Pgridspacing = plasma_h.gridspacing;


	dim3 cudaGridSize1((nptcls+threadsPerBlock-1)/threadsPerBlock,nspecies,1);

	int* nptcls_left_d;
	int* nptcls_left_h = (int*)malloc(sizeof(int));
	int* didileave;
	int gridSize = (4+Pgrid_i_dims.x-1)/4*(4+Pgrid_i_dims.y-1)/4;


	int* NptinCell = (int*)malloc(gridSize*nspecies*sizeof(realkind));
	int* nBlocks_h = (int*)malloc(nspecies*sizeof(int));
	int* nBlocks_max_d;
//	printf("CudaMalloc1 \n");
	cudaMalloc((void**)&nBlocks_max_d,sizeof(int));
	cudaMemset(nBlocks_max_d,0,sizeof(int));
	cudaThreadSynchronize();

	int2* cellInfo_h = (int2*)malloc((gridSize+1)*nspecies*sizeof(int2));

	cudaMatrixi2 cellInfo_d(gridSize+1,nspecies+1);
//	printf("CudaMalloc2 \n");

	int* nblocksperspecies_d;
//	printf("CudaMalloc4 \n");
	cudaMalloc((void**)&nblocksperspecies_d,nspecies*sizeof(int));
	cudaThreadSynchronize();
	cudaMemset(nblocksperspecies_d,0,nspecies*sizeof(int));

	// Figure out how many particles are in each cell
	cudaGridSize.x = (nptcls+cudaBlockSize.x-2)/threadsPerBlock;

	printf("Launching Count Particles Kernel \n");
	CUDA_SAFE_KERNEL((count_particles<<<cudaGridSize,cudaBlockSize>>>(*particles,cellInfo_d,gridSize)));
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "count particles %s\n", cudaGetErrorString(status));}

	 // Fix the cellinfo array for errors from cells with 0 particles
	cudaGridSize.x = (gridSize+cudaBlockSize.x-1)/threadsPerBlock;


	printf("Launching Fix Cellinfo Kernel \n");
	CUDA_SAFE_KERNEL((fix_cellinfo<<<cudaGridSize, cudaBlockSize>>>(cellInfo_d,gridSize)));
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "Fix Cell Info %s\n", cudaGetErrorString(status));}

	// Figure out how many thread blocks are needed for each cell and the total number of thread blocks

	printf("Launching Count Blocks Kernel \n");
	 CUDA_SAFE_KERNEL((count_blocks<<<cudaGridSize, cudaBlockSize>>>(cellInfo_d,nblocksperspecies_d,nBlocks_max_d,gridSize)));
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "count blocks %s\n", cudaGetErrorString(status));}
	//cudaMemcpy(cellInfo_h,cellInfo_d,(gridSize+1)*sizeof(int2),cudaMemcpyDeviceToHost);
	CUDA_SAFE_CALL(cudaMemcpy(nBlocks_h,nBlocks_max_d,sizeof(int),cudaMemcpyDeviceToHost));

	cudaThreadSynchronize();

	printf(" nblocks = %i \n",nBlocks_h[0]);

	if(nBlocks_h[0] > nptcls)
	{
		printf(" error, nBlocks way to big, returning \n");
		return;
	}




	// Populate blockInfo
	cudaGridSize1.x = nBlocks_h[0];

	cudaMatrixi3 blockinfo_d(nBlocks_h[0]+5,nspecies);// grid index, first particle index, number of particles in block
	cudaDeviceSynchronize();

	CUDA_SAFE_KERNEL((populate_blockinfo<<<1,nspecies>>>(blockinfo_d,cellInfo_d,nspecies,gridSize,nblocksperspecies_d)));

	//cudaMemcpy(blockinfo_d,blockinfo_h,nBlocks_h[0]*sizeof(int3),cudaMemcpyHostToDevice);
	cudaThreadSynchronize();


	 // Move the particles
	printf("Launching Move Kernel \n");

	CUDA_SAFE_KERNEL((XPlist_move_shared_kernel<<<cudaGridSize1,cudaBlockSize>>>(*particles,*particles_old,
																plasma_d,orbit_error,splitting_condition,
																blockinfo_d,istep,xposition_matrix,yposition_matrix,
																nptcls_at_step_matrix)));

	printf("Finished Move Kernel \n");

	printf("Splitting Particle List\n");
	particles_done_this_step = particles->split(splitting_condition);

	if(particles_done_this_step.nptcls_max > 1)
	{
		printf("Appending finished Particle List\n");
		particles_done->append(particles_done_this_step);
		particles_done_this_step.XPlistFree();
	}




//	if(nptcls_left_h[0] < nptcls)
	//	inject_new_particles(didileave,nptcls_left_h[0]);

	blockinfo_d.cudaMatrixFree();
	cellInfo_d.cudaMatrixFree();
	splitting_condition.cudaMatrixFree();

	orbit_error.cudaMatrixFree();
	CUDA_SAFE_CALL(cudaFree(nblocksperspecies_d));
	CUDA_SAFE_CALL(cudaFree(nBlocks_max_d));
	free(cellInfo_h);

//	printf("Finished freeing stuff \n");
	//cudaThreadSynchronize();


}
*/
__global__
void XPlist_move_kernel(XPlist* particles,XPlist* particles_old,Environment* plasma_in,
										  cudaMatrixr orbit_error,cudaMatrixui splitting_condition,
										  int istep,cudaMatrixf px,cudaMatrixf py,
										  cudaMatrixi nptcls_at_step)
{
	unsigned int idx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int gidx = bidx*blockDim.x+idx;
	unsigned int block_start = bidx*blockDim.x;

	//if(idx == 0)
	//	printf("nptcls[%i] = %i\n",idBeam,particles.nptcls[idBeam]);

	realkind mu;
	int nx;
	int ny;

#ifdef Animate_orbits
	if(gidx+idBeam == 0)
	{
		nptcls_at_step(istep) = particles->nptcls[idBeam];
	}
#endif
	realkind energy0;
	realkind momentum0;
	realkind potential0;
	realkind B;

	realkind orberror;


	// Results of the rk4 calculations;
	realkind3 initialvalues;
	realkind3 result[4];
	// Temporary variable for error calculation
	realkind3 temp;
	int orbflag = 0;
	int nr = plasma_in -> griddims.x;
	int nz = plasma_in -> griddims.y;
	int iorberr = 0;

	// dt is used a lot, so we'll stick it in shared memory
	__shared__ realkind dt[BLOCK_SIZE];

	// The local particle list, some members will exist in shared memory
	__shared__ XPlist localion;

	__shared__ realkind2 origin; // The R and Z coordinates of the bottom right corner of the cell

	// Splines that describe the fields for this cell
	__shared__ XPTextureSpline Psispline;
	__shared__ XPTextureSpline gspline;
	__shared__ XPTextureSpline Phispline;
	__shared__ realkind2 gridspacing;
		//printf("blockstart(%i) = %i nptcls[%i] = %i\n",bidx,blockInfo(bidx,idBeam).y,bidx,blockInfo(bidx,idBeam).z);

	if((block_start)<particles->nptcls[idBeam])
	{


	// Copy data from global memory to shared memory
	localion.shift_local(particles,block_start);




	if((idx == 0))
	{
		plasma_in->istep = istep;
		Psispline = plasma_in->Psispline;
		gspline = plasma_in -> gspline;
		Phispline = plasma_in -> Phispline;
		origin.x = (plasma_in -> Rmin);
		origin.y =(plasma_in -> Zmin);

		gridspacing = plasma_in -> gridspacing;

		// Allocate space in shared memory for shared copy of splines

	}


	__syncthreads();
	if(idx < localion.nptcls_max)
	{
		orbflag = localion.orbflag[idx];
		localion.old_idx[idx] = gidx;
	}
	else
	{
		orbflag = 0;
	}



	if(orbflag == 1)
	{
#ifdef Animate_orbits

		if((localion.original_idx[idx] % SPHERE_SPACING) == 0)
		{
			px(localion.original_idx[idx],idBeam,istep) = localion.px[0][idx];
			py(localion.original_idx[idx],idBeam,istep) = localion.py[0][idx];
		}
#endif
		mu = localion.mu[idx];

		// Figure out what the timestep should be
		dt[idx] = localion.eval_dt(plasma_in);
		localion.deltat[idx] = dt[idx];

		//printf("dt = %g\n",dt[idx]);
		initialvalues.x = localion.px[0][idx];
		initialvalues.y = localion.py[0][idx];
		initialvalues.z = localion.vpara[0][idx];
		energy0 = localion.energy[idx];
		momentum0 = localion.momentum[idx];
		potential0 = localion.potential[0][idx];

		for(int itry=0;itry<20;itry++)
		{
			dt[idx] = localion.deltat[idx];

			//RK1
			for(int i=0;i<3;i++)
			{
				result[i] = localion.XPlist_derivs(Psispline,gspline,Phispline,mu,0);
				//printf("rk4 results(%i,%i) = %g, %g, %g, %g\n",gidx,idBeam,result[i].x,result[i].y,result[i].z,dt[idx]);
				localion.px[0][idx] = initialvalues.x+rk4_mults[i]*dt[idx]*result[i].x;
				localion.py[0][idx] = initialvalues.y+rk4_mults[i]*dt[idx]*result[i].y;
				localion.vpara[0][idx] = initialvalues.z+rk4_mults[i]*dt[idx]*result[i].z;

				localion.nx[0][idx] = rint((localion.px[0][idx]-origin.x)/(gridspacing.x));
				localion.ny[0][idx] = rint((localion.py[0][idx]-origin.y)/(gridspacing.y));

				if(localion.check_orbit(plasma_in)!=0)
				{
					// If the particle has left the grid, it needs to stop orbiting

					localion.orbflag[idx] = 0;
					orbflag = 0;
					localion.pexit[idx] = XPlistexit_limiter;
					//printf("particles %i left orbit during rk4\n",gidx);

					break;
				}
			}

			if((localion.check_orbit(plasma_in) == 0)&&(orbflag == 1))
			//if(orbflag == 1)
			{
				result[3] = localion.XPlist_derivs(Psispline,gspline,Phispline,mu,0);

				localion.px[0][idx] =initialvalues.x+dt[idx]*(result[0].x+2.0*result[1].x+2.0*result[2].x+result[3].x)/6.0;
				localion.py[0][idx] =initialvalues.y+dt[idx]*(result[0].y+2.0*result[1].y+2.0*result[2].y+result[3].y)/6.0;
				localion.vpara[0][idx] =initialvalues.z+dt[idx]*(result[0].z+2.0*result[1].z+2.0*result[2].z+result[3].z)/6.0;

				localion.nx[0][idx] = (localion.px[0][idx]-origin.x)/(gridspacing.x);
				localion.ny[0][idx] = (localion.py[0][idx]-origin.y)/(gridspacing.y);

				result[3] = localion.XPlist_derivs(Psispline,gspline,Phispline,mu,0);

				temp.x = initialvalues.x+dt[idx]*(result[0].x+2.0*result[1].x+2.0*result[2].x+result[3].x)/6.0;
				temp.y = initialvalues.y+dt[idx]*(result[0].y+2.0*result[1].y+2.0*result[2].y+result[3].y)/6.0;
				temp.z = initialvalues.z+dt[idx]*(result[0].z+2.0*result[1].z+2.0*result[2].z+result[3].z)/6.0;

				orbit_error(gidx,idBeam,0) = (localion.px[0][idx]-temp.x)/6.0;
				orbit_error(gidx,idBeam,1) = (localion.py[0][idx]-temp.y)/6.0;
				orbit_error(gidx,idBeam,2) = (localion.vpara[0][idx]-temp.z)/6.0;

				orbit_error(gidx,idBeam,0) = fabs(orbit_error(gidx,idBeam,0)/((fabs(initialvalues.x+0.5*dt[idx]*result[0].x))+epsilon));
				orbit_error(gidx,idBeam,1) = fabs(orbit_error(gidx,idBeam,1)/((fabs(initialvalues.y+0.5*dt[idx]*result[0].y+origin.y))+epsilon));
				orbit_error(gidx,idBeam,2) = fabs(orbit_error(gidx,idBeam,2)/((fabs(initialvalues.z+0.5*dt[idx]*result[0].z))+epsilon));

				orbit_error(gidx,idBeam,4) = fabs(localion.energy[idx]-energy0)/(fabs(energy0)+epsilon);
				orbit_error(gidx,idBeam,5) = fabs(localion.momentum[idx]-momentum0)/(fabs(momentum0)+epsilon);

			}
			else
			{
				if(idx < localion.nptcls_max)
				{// Particles that have left the grid in the orbit process
					//orbit_error(gidx,idBeam,0) = 0;
					//orbit_error(gidx,idBeam,1) = 0;
					//orbit_error(gidx,idBeam,2) = 0;

					//orbit_error(gidx,idBeam,4) = 0;
					//orbit_error(gidx,idBeam,5) = 0;

					//localion.energy[idx] = energy0;
					//localion.momentum[idx] = momentum0;
					localion.orbflag[idx] = 0;
					orbflag = 0;
					localion.pexit[idx] = XPlistexit_limiter;


				}
				break;
			}


			if(orbflag == 1)
			{
				nx = max(0,min(nr-1,localion.nx[0][idx]));
				ny = max(0,min(nz-1,localion.ny[0][idx]));
				localion.cellindex[0][idx] = localion.eval_NGC(plasma_in,idx,0);



				temp.x = orbit_error(gidx,idBeam,0);

				for(int i=1;i<6;i++)
				{
					if(orbit_error(gidx,idBeam,i)>temp.x)
					{

					}
					temp.x = max(temp.x,orbit_error(gidx,idBeam,i));
				}

				orberror = temp.x/orbit_error_con;

				if(orberror>1.0)
				{
					// Error too big, adjust the step size and redo the timestep
				//	printf("orbit_error(%i,%i) = %14.10g (%i) @ %f, %f\n",gidx,idBeam,temp.x,iorberr,localion.px[0][idx],localion.py[0][idx]);



					if(dt[idx] <= orbit_dt_min)
					{
						localion.orbflag[idx] = 0;
						localion.pexit[idx] = XPlistexit_dtmin;
						break;
					}
					else
					{
						dt[idx] = 0.9*dt[idx]*pow(orberror,znrc_pshrnk);
						localion.px[0][idx] = initialvalues.x;
						localion.py[0][idx] = initialvalues.y;
						localion.vpara[0][idx] = initialvalues.z;
						localion.energy[idx] = energy0;
						localion.momentum[idx] = momentum0;
						localion.potential[0][idx] = potential0;
					}

					localion.deltat[idx] = dt[idx];

				}
				else
				{

					// Increase the error if it is too small
					if(orberror > znrc_errcon)
					{
						dt[idx] *= (0.9*pow(orberror,znrc_pgrow));
					}
					else
					{
						dt[idx] *= 5.0;
					}

					localion.deltat[idx] = dt[idx];

					break;
				}



			}
			else
			{
				localion.cellindex[0][idx] = 2*plasma_in->ntransp_zones;
				break;
			}



			iorberr++;


		}


		localion.cxdt_goosed[idx] = dt[idx];
		localion.fpdt_goosed[idx] = dt[idx];


		localion.time_done[idx] += dt[idx];

		if(((localion.check_orbit(plasma_in) == 0)&&(orbflag == 1)))
		{
				B = localion.eval_Bmod(Psispline,gspline);
				localion.vperp[0][idx] = sqrt(2.0*mu*B/(localion.mass[idx]*ZMP));
/*
				if(iorberr > 0)
				{
					printf("orbit_error(%i,%i) = %14.10g (%i)\n",gidx,idBeam,orberror,iorberr);
				}
*/

				localion.update_gc(plasma_in);

				localion.gphase();

				localion.update_flr(plasma_in);

				localion.steps_midplanecx[idx]++;
		}
		else
		{
			localion.orbflag[idx] = 0;
			localion.pexit[idx] = XPlistexit_limiter;
		}

		if(localion.time_done[idx] >= (plasma_in -> delt-orbit_dt_min))
		{
			localion.orbflag[idx] = 0;
			localion.pexit[idx] = XPlistexit_time;
			//printf("time done = %f out of %f \n",localion.time_done[idx],plasma_in -> delt);
		}

		orbflag = localion.orbflag[idx];







	}

	}

	if(gidx < particles->nptcls[idBeam])
	{
		orbflag = localion.orbflag[idx];

		if(orbflag == 1)
		{
			splitting_condition(gidx,idBeam) = 0;
		}
		else
		{
			splitting_condition(gidx,idBeam) = 1;
		//	printf("orbit exit = %i = %i \n",localion.pexit[idx],localion.orbflag[idx]);
		}
	}









}

__host__
void move_particles(Environment* plasma_d,XPlist* particles,XPlist* particles_old,XPlist* particles_done,int istep)
{

	int nspecies = particles->nspecies;
	int nptcls = particles->nptcls_max;

	dim3 cudaGridSize(1,particles->nspecies,1);
	dim3 cudaBlockSize(BLOCK_SIZE,1,1);

	cudaMatrixr orbit_error(particles->nptcls_max+1,nspecies,6);
	cudaMatrixui splitting_condition(nptcls,nspecies);

	XPlist* particles_d;
	XPlist* particles_old_d;
	XPlist particles_done_this_step;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,sizeof(XPlist)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_old_d,sizeof(XPlist)));

	CUDA_SAFE_CALL(cudaMemcpy(particles_d,particles,sizeof(XPlist),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_old_d,particles_old,sizeof(XPlist),cudaMemcpyHostToDevice));

	cudaGridSize.x = (particles->nptcls_max+cudaBlockSize.x-1)/cudaBlockSize.x;


	 // Move the particles
#ifdef debug
	printf("Launching Move Kernel \n");
#endif

	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_KERNEL((XPlist_move_kernel<<<cudaGridSize,cudaBlockSize>>>(particles_d,particles_old_d,
																plasma_d,orbit_error,splitting_condition,
																istep,xposition_matrix,yposition_matrix,
																nptcls_at_step_matrix)));


	//CUDA_SAFE_CALL(cudaThreadSetLimit(cudaLimitStackSize, ((1<<11)*sizeof(char))));

#ifdef debug
	printf("Splitting Particle List\n");
#endif
	particles_done_this_step = particles->split(splitting_condition,0);

	if(particles_done_this_step.nptcls_max > 0)
	{
#ifdef debug
		printf("Appending finished Particle List\n");
#endif
		//particles_done->append(particles_done_this_step);
		//append_finished_particle_table(particles_done_this_step);

	}

	particles_done_this_step.XPlistFree();

#ifdef debug
	checkMemory();
	printf("Finished Move Kernel \n");
#endif








	splitting_condition.cudaMatrixFree();

	orbit_error.cudaMatrixFree();

	cudaFree(particles_d);
	cudaFree(particles_old_d);
	//checkMemory();

//	printf("Finished freeing stuff \n");
	//cudaThreadSynchronize();


}




  __host__
void charge_exchange(Environment* plasma_d,XPlist &particles,XPlist &particles_old,int istep)
{


	int nspecies = particles.nspecies;

	dim3 cudaGridSize(1,nspecies,1);
	dim3 cudaBlockSize(BLOCK_SIZE2,1,1);

	XPlist* particles_d;
	XPlist* particles_old_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,sizeof(XPlist)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_old_d,sizeof(XPlist)));

	CUDA_SAFE_CALL(cudaMemcpy(particles_d,&particles,sizeof(XPlist),cudaMemcpyHostToDevice));



	int cudaGridSizex = (BLOCK_SIZE2+particles.nptcls_max-1)/BLOCK_SIZE2;

	int nptcls_cx_h;
	int nblocks_nutrav_h;

	cudaGridSize.x = cudaGridSizex;

	cudaMatrixui splittinglist(particles.nptcls_max,particles.nspecies);

	XPlist cx_particles;
	XPlist nutrav_parents;
	XPlist dead_neutrals;

	// Check for a charge exchange collision
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	CUDA_SAFE_KERNEL((XPlist_check_CX<<<cudaGridSize,cudaBlockSize>>>(particles_d,splittinglist,istep)));

	// Split off the particles that are going to undergo a cx event.
	cx_particles = particles.split(splittinglist,1);



	nptcls_cx_h = cx_particles.nptcls_max;
	printf("nptcls_cx = %i\n",nptcls_cx_h);
	if(nptcls_cx_h == 0)
	{
		cx_particles.XPlistFree();
		splittinglist.cudaMatrixFree();
		cudaFree(particles_d);
		cudaFree(particles_old_d);
		return;
	}


	cudaBlockSize.x = BLOCK_SIZE;
	cudaGridSize.x =  (BLOCK_SIZE+nptcls_cx_h-1)/BLOCK_SIZE;


	// Setup arrays to store temporary CX data
	cudaMatrixr nutrav_weight_in(nptcls_cx_h,nspecies);
	cudaMatrixi ievent(nptcls_cx_h,nspecies);
	cudaMatrixui nsplits(nptcls_cx_h,nspecies);
	cudaMatrixi parent_ids(nptcls_cx_h,nspecies);

	cudaMatrixui old_ids0;
	cudaMatrixui old_ids1;


	// Figure out which particles undergo cx and spawn neutrals

	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	CUDA_SAFE_KERNEL((beamcx_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,cx_particles,nutrav_weight_in,ievent,
																					 nsplits,splittinglist)));
	nutrav_parents = cx_particles.split(splittinglist,old_ids0,old_ids1,1,1);



	nptcls_cx_h = cx_particles.nptcls_max;

	if(cx_particles.nptcls_max > 0)
	{
		printf("Appending main Particle list (%i ptcls) with nptcls %i \n",particles.nptcls_max,cx_particles.nptcls_max);
		if(particles.nptcls_max > 0)
		{
			particles.append(cx_particles);

		}
		else
		{
			particles = cx_particles;
		}
	}

	ievent.cudaMatrixFree();
	parent_ids.cudaMatrixFree();

	if(nutrav_parents.nptcls_max <= 0)
	{
		splittinglist.cudaMatrixFree();
		nutrav_weight_in.cudaMatrixFree();
		nsplits.cudaMatrixFree();
		old_ids1.cudaMatrixFree();
		old_ids0.cudaMatrixFree();
		return;
	}



	splittinglist.cudaMatrixFree();


	nblocks_nutrav_h = nutrav_parents.nptcls_max;
	printf("nptcls_nutrav = %i \n",nblocks_nutrav_h);

//	cudaExtent extent = nsplits.getdims();
//	printf("nsplits = %i x %i \n", extent.width/sizeof(realkind),extent.height);


	// Launch 1 block per parent

	XPlist cx_neutrals(nblocks_nutrav_h*Max_Splits,nspecies,XPlistlocation_device);
	splittinglist.cudaMatrix_allocate(nblocks_nutrav_h*Max_Splits,nspecies,1);

	cudaMatrixT<realkind3> recaptured_velocity_vector(Max_Splits*nblocks_nutrav_h,nspecies);

	cudaBlockSize.x = Max_Splits;
	cudaGridSize.x =  nblocks_nutrav_h;

	CUDA_SAFE_KERNEL((setup_nutrav<<<cudaGridSize,cudaBlockSize>>>(plasma_d,nutrav_parents,cx_neutrals,nutrav_weight_in,
																				   nsplits,old_ids1,nptcls_cx_h)));

	cudaDeviceSynchronize();
	nutrav_parents.XPlistFree();
	old_ids1.cudaMatrixFree();
	old_ids0.cudaMatrixFree();



	cudaBlockSize.x = Max_Track_segments;
	cudaGridSize.x =  Max_Splits*nblocks_nutrav_h;

	CUDA_SAFE_KERNEL((nutrav_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,cx_neutrals,recaptured_velocity_vector,splittinglist)));

	// Split off the neutrals that are going to be deposited

	dead_neutrals = cx_neutrals.split(splittinglist,old_ids0,old_ids1,1,1);

	printf("number of new particles = %i \n",cx_neutrals.nptcls_max);
	if(cx_neutrals.nptcls_max > 0)
	{


	// Setup the new ions that came from the recpture
	cudaBlockSize.x = BLOCK_SIZE2;
	cudaGridSize.x =  (BLOCK_SIZE2+cx_neutrals.nptcls_max-1)/BLOCK_SIZE2;
	CUDA_SAFE_KERNEL((recapture_neutrals<<<cudaGridSize,cudaBlockSize>>>(plasma_d,cx_neutrals,
																						recaptured_velocity_vector,old_ids0,dead_neutrals.nptcls_max)));

	printf("Appending main Particle list (%i ptcls) with nptcls %i \n",particles.nptcls_max,cx_neutrals.nptcls_max);

	particles.append(cx_neutrals);

	printf("main_list.nptcls_max = %i \n",particles.nptcls_max);

	update_original_idx_counter(cx_neutrals.nptcls_max);




	//append_finished_particle_table(dead_neutrals);



	}
#ifdef debug
	printf("Finishing Charge Exchange\n");
#endif

	recaptured_velocity_vector.cudaMatrixFree();
	splittinglist.cudaMatrixFree();
	nsplits.cudaMatrixFree();
	dead_neutrals.XPlistFree();
	nutrav_weight_in.cudaMatrixFree();

	old_ids0.cudaMatrixFree();
	old_ids1.cudaMatrixFree();

	CUDA_SAFE_CALL(cudaFree(particles_d));
	CUDA_SAFE_CALL(cudaFree(particles_old_d));




	return;

}



/*
__host__
void charge_exchange(Environment* plasma_d,XPlist &particles,XPlist &particles_old,int istep)
{

	int nspecies = particles.nspecies;

	dim3 cudaGridSize(1,nspecies,1);
	dim3 cudaBlockSize(BLOCK_SIZE2,1,1);

	XPlist* particles_d;
	XPlist* particles_old_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,sizeof(XPlist)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_old_d,sizeof(XPlist)));

	CUDA_SAFE_CALL(cudaMemcpy(particles_d,&particles,sizeof(XPlist),cudaMemcpyHostToDevice));



	int cudaGridSizex = (BLOCK_SIZE2+particles.nptcls_max-1)/BLOCK_SIZE2;

	int nptcls_cx_h;
	int nblocks_nutrav_h;

	cudaGridSize.x = cudaGridSizex;

	cudaMatrixui splittinglist(particles.nptcls_max,particles.nspecies);

	XPlist cx_particles;
	XPlist nutrav_parents;
	XPlist dead_neutrals;

	// Check for a charge exchange collision
	CUDA_SAFE_KERNEL((XPlist_check_CX<<<cudaGridSize,cudaBlockSize>>>(particles_d,splittinglist,istep)));

	// Split off the particles that are going to undergo a cx event.
	cx_particles = particles.split(splittinglist,1);





	nptcls_cx_h = cx_particles.nptcls_max;
	printf("nptcls_cx = %i\n",nptcls_cx_h);
	if(nptcls_cx_h == 0)
		return;

	XPlist cx_neutrals(nptcls_cx_h/100,nspecies,XPlistlocation_device);


	cudaBlockSize.x = BLOCK_SIZE;
	cudaGridSize.x =  (BLOCK_SIZE+nptcls_cx_h-1)/BLOCK_SIZE;


	// Setup arrays to store temporary CX data
	cudaMatrixr nutrav_weight_in(nptcls_cx_h,nspecies);
	cudaMatrixi ievent(nptcls_cx_h,nspecies);
	cudaMatrixui nsplits(nptcls_cx_h,nspecies);
	cudaMatrixi parent_ids(nptcls_cx_h,nspecies);

	cudaMatrixui old_ids0;
	cudaMatrixui old_ids1;


	// Figure out which particles undergo cx and spawn neutrals

	CUDA_SAFE_KERNEL((beamcx_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,cx_particles,nutrav_weight_in,ievent,
																					 nsplits,splittinglist)));
	nutrav_parents = cx_particles.split(splittinglist,old_ids0,old_ids1,1,1);



	nptcls_cx_h = cx_particles.nptcls_max;

	if(cx_particles.nptcls_max > 0)
	{
		printf("Appending main Particle list (%i ptcls) with nptcls %i \n",particles.nptcls_max,cx_particles.nptcls_max);
		if(particles.nptcls_max > 0)
		{
			particles.append(cx_particles);

		}
		else
		{
			particles = cx_particles;
		}
	}

	ievent.cudaMatrixFree();
	parent_ids.cudaMatrixFree();

	if(nutrav_parents.nptcls_max <= 0)
	{
		splittinglist.cudaMatrixFree();
		nutrav_weight_in.cudaMatrixFree();
		nsplits.cudaMatrixFree();
		old_ids1.cudaMatrixFree();
		old_ids0.cudaMatrixFree();
		return;
	}



	splittinglist.cudaMatrixFree();


	nblocks_nutrav_h = nutrav_parents.nptcls_max;
	printf("nptcls_nutrav = %i \n",nblocks_nutrav_h);

//	cudaExtent extent = nsplits.getdims();
//	printf("nsplits = %i x %i \n", extent.width/sizeof(realkind),extent.height);


	// Launch 1 block per parent


	splittinglist.cudaMatrix_allocate(nblocks_nutrav_h*Max_Splits,nspecies,1);

	cudaMatrixT<realkind3> recaptured_velocity_vector(Max_Splits*nblocks_nutrav_h,nspecies);

	cudaBlockSize.x = Max_Splits;
	cudaGridSize.x =  nblocks_nutrav_h;

	CUDA_SAFE_CALL(cudaThreadSetLimit(cudaLimitStackSize, ((1<<14)*sizeof(char))));
	CUDA_SAFE_KERNEL((setup_nutrav<<<cudaGridSize,cudaBlockSize>>>(plasma_d,nutrav_parents,cx_neutrals,nutrav_weight_in,
																				   nsplits,old_ids1,nptcls_cx_h)));

	cudaDeviceSynchronize();
	nutrav_parents.XPlistFree();
	old_ids1.cudaMatrixFree();
	old_ids0.cudaMatrixFree();


	CUDA_SAFE_CALL(cudaThreadSetLimit(cudaLimitStackSize, ((1<<11)*sizeof(char))));


	cudaBlockSize.x = Max_Track_segments;
	cudaGridSize.x =  Max_Splits*nblocks_nutrav_h;

	CUDA_SAFE_KERNEL((nutrav_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,cx_neutrals,recaptured_velocity_vector,splittinglist)));

	// Split off the neutrals that are going to be deposited

	dead_neutrals = cx_neutrals.split(splittinglist,old_ids0,old_ids1,1,1);

	printf("number of new particles = %i \n",cx_neutrals.nptcls_max);
	if(cx_neutrals.nptcls_max > 0)
	{


	// Setup the new ions that came from the recpture
	cudaBlockSize.x = BLOCK_SIZE2;
	cudaGridSize.x =  (BLOCK_SIZE2+cx_neutrals.nptcls_max-1)/BLOCK_SIZE2;
	CUDA_SAFE_KERNEL((recapture_neutrals<<<cudaGridSize,cudaBlockSize>>>(plasma_d,cx_neutrals,
																						recaptured_velocity_vector,old_ids0,dead_neutrals.nptcls_max)));

	printf("Appending main Particle list (%i ptcls) with nptcls %i \n",particles.nptcls_max,cx_neutrals.nptcls_max);
	particles.append(cx_neutrals);
	printf("main_list.nptcls_max = %i \n",particles.nptcls_max);

	update_original_idx_counter(cx_neutrals.nptcls_max);




	//append_finished_particle_table(dead_neutrals);



	}

	printf("Finishing Charge Exchange\n");

	recaptured_velocity_vector.cudaMatrixFree();
	splittinglist.cudaMatrixFree();
	nsplits.cudaMatrixFree();
	dead_neutrals.XPlistFree();
	nutrav_weight_in.cudaMatrixFree();

	old_ids0.cudaMatrixFree();
	old_ids1.cudaMatrixFree();

	CUDA_SAFE_CALL(cudaFree(particles_d));
	CUDA_SAFE_CALL(cudaFree(particles_old_d));




	return;

}
*/



__global__
void collide_gpu_kernel(XPlist particles_global,Environment* plasma_in,int max_steps)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	__shared__ XPlist particles;

	particles.shift_local(&particles_global);

	int isteps;
	int flag;

	__syncthreads();

	if(idx < particles.nptcls_max)
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


__global__
void step_finish_kernel(Environment* plasma_in,XPlist* particles_global,XPlist* particles_global_old,
										cudaMatrixui splitting_condition)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int block_start = blockIdx.x*blockDim.x;
	unsigned int orbflag = 0;

	__shared__ XPlist particles;
	realkind velocity;
	realkind transp_index;

if(block_start < particles_global->nptcls_max)
{
	particles.shift_local(particles_global);
	__syncthreads();

	if(idx < particles.nptcls_max)
	{
		if(particles.check_orbit(plasma_in) == 0)
		{
			orbflag = particles.orbflag[idx];


			velocity = pow(particles.vpara[0][idx],2)+pow(particles.vperp[0][idx],2);
			velocity = sqrt(velocity);
			transp_index = plasma_in->transp_zone(particles.px[0][idx],particles.py[0][idx]);

			if(velocity <= plasma_in ->thermal_velocity(transp_index,idBeam))
			{
				particles.orbflag[idx] = 0;
				orbflag = 0;
				particles.pexit[idx] = XPlistexit_thermalized;
			}


		}
		else
		{
			orbflag = 0;
			particles.orbflag[idx] = 0;
			particles.pexit[idx] = XPlistexit_limiter;
		}






		splitting_condition(gidx,idBeam) = (orbflag!=1);

	}

	if(orbflag == 1)
	{
		// Initialize the bounce and collision counters for new particles, and set those particles as regular particles
		if(particles.pexit[idx] == XPlistexit_newparticle)
		{
			particles.bounce_init(1,plasma_in);
			particles.pexit[idx] = XPlistexit_stillorbiting;
		}
		else
		{
			// Update the timing factors for those particles that have crossed the midplane.
			if(particles.check_midplane_cx(plasma_in,particles_global_old) == 1)
			{
				particles.update_timing_factors(plasma_in);
			}

		}


	}


}
}

__host__
void step_finish(Environment* plasma_d,XPlist &particles,XPlist &particles_old,int istep)
{

	dim3 cudaGridSize(1,particles.nspecies,1);
	dim3 cudaBlockSize(BLOCK_SIZE2,1,1);

	cudaGridSize.x = (particles.nptcls_max+cudaBlockSize.x-1)/cudaBlockSize.x;

	XPlist* particles_d;
	XPlist* particles_old_d;
	XPlist particles_done_this_step;

	cudaMatrixui splitting_condition(particles.nptcls_max,particles.nspecies);
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,sizeof(XPlist)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_old_d,sizeof(XPlist)));

	CUDA_SAFE_CALL(cudaMemcpy(particles_d,&particles,sizeof(XPlist),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_old_d,&particles_old,sizeof(XPlist),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	CUDA_SAFE_KERNEL((step_finish_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,particles_d,particles_old_d,splitting_condition)));

	printf("Splitting Particle List\n");
	particles_done_this_step = particles.split(splitting_condition,1);

	if(particles_done_this_step.nptcls_max > 0)
	{
		printf("Appending finished Particle List\n");
		//particles_done->append(particles_done_this_step);
		//append_finished_particle_table(particles_done_this_step);

	}

	particles_done_this_step.XPlistFree();

	cudaFree(particles_d);
	cudaFree(particles_old_d);

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
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaDeviceSynchronize();

	checkMemory();

	next_tex2D = 0;
	next_tex1DLayered = 0;
	double* double_data[46];
	int* int_data[2];


	size_t free2 = 0;
	size_t total = 0;

	cudaMemGetInfo(&free2,&total);
	printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free2)/(1<<20),(int)(total-free2)/(1<<20));

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

	plasma_h.delt = 1.0e-2;
	plasma_h.max_species = 1;

	//plasma_h.check_environment();


}





extern "C" void orbit_gpu_(int* NBNDEX,int* NBSCAY,int* NBIENAY,double* PINJAY,
											 double* EINJAY,double* XIAY,double* THAY,double* VAY,
											 double* XKSIDY,double* WGHTAY,double* XZBEAMA,
											 double* ABEAMA,double* RMJIONAY,double* XZIONAY,
											 double* CXPRAY,double* FPPRAY,double nbdelt,int* ierr1)
{


	double* double_data_in_h[10];
	int* int_data_in_h[2];

	double_data_in_h[0] = XKSIDY;
	double_data_in_h[1] = VAY;
	double_data_in_h[2] = WGHTAY;
	double_data_in_h[3] = XIAY;
	double_data_in_h[4] = THAY;
	double_data_in_h[5] = XZIONAY;
	double_data_in_h[6] = CXPRAY;
	double_data_in_h[7] = FPPRAY;
	double_data_in_h[8] = XZBEAMA;
	double_data_in_h[9] = ABEAMA;

	int_data_in_h[0] = NBSCAY;
	int_data_in_h[1] = NBNDEX;

	int minb = plasma_h.max_particles;
	int mibs = plasma_h.max_species;
	int nptcls_left;
	int nptcls_left0;
	int istep = 0;

	checkMemory();


	printf("minb = %i, mibs = %i \n",minb,mibs);

	Environment* plasma_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&plasma_d,sizeof(Environment)));
	CUDA_SAFE_CALL(cudaMemcpy(plasma_d,&plasma_h,sizeof(Environment),cudaMemcpyHostToDevice));

	//CUDA_SAFE_CALL(cudaThreadSetLimit(cudaLimitStackSize, ((1<<14)*sizeof(char))));

//	plasma_h.check_environment();
	// Setup particle list

	XPlist particles(minb,mibs,XPlistlocation_device);

	XPlist particles_done;
	particles_done.nptcls_max = 0;

	particles.setup(plasma_d,double_data_in_h,int_data_in_h,minb,mibs);

	//particles.check_sort();
#ifdef Animate_orbits
	cudaMatrixf xposition(minb/SPHERE_SPACING,mibs,MAX_STEPS+10);
	cudaMatrixf yposition(minb/SPHERE_SPACING,mibs,MAX_STEPS+10);
	cudaMatrixi nptcls_at_step(MAX_STEPS+10);

	xposition_matrix = xposition;
	yposition_matrix = yposition;
	nptcls_at_step_matrix = nptcls_at_step;
#endif

	particles.sort(plasma_h.gridspacing,plasma_h.griddims);

	//particles.check_sort();

	nptcls_left = particles.nptcls_max;
	nptcls_left0 = particles.nptcls_max;

	update_original_idx_counter(nptcls_left0);

	XPlist particles_old(nptcls_left,mibs,XPlistlocation_device);
	checkMemory();

	 // Top of the time loop

	istep = 0;
	while(nptcls_left > 10)
	{
#ifdef debug
		printf("\n---------------------------------------\nStarting Step %i...\n",istep);
		printf("nptcls_left = %i at step %i\n",nptcls_left,istep);
#endif
		// Initialize the counter for the table of finished particle lists
		finished_particle_list_counter = 0;



		//checkMemory();
		XPlistCopy(particles_old,particles,nptcls_left,mibs,cudaMemcpyDeviceToDevice);
		 // Move the particles
		move_particles(plasma_d,&particles,&particles_old,&particles_done,istep);


#ifdef debug
		printf("nptcls_left = %i at step %i\n",nptcls_left,istep);
#endif

		 // BeamCX
	//	checkMemory();
		charge_exchange(plasma_d,particles,particles_old,istep);

		 // Fokker Plank Collisions

		 // Anomalous diffusion

		 // Combine all particle lists

		 // Cleanup time step
	//	checkMemory();
		step_finish(plasma_d,particles,particles_old,istep);


		nptcls_left = particles.nptcls_max;



		particles.sort(plasma_h.gridspacing,plasma_h.griddims);

	//	checkMemory();
#ifdef debug
		printf("Finished Step %i...\n---------------------------------------\n",istep);
#endif


		if(nptcls_left < BLOCK_SIZE/4)
			break;


		istep++;

		if(istep > MAX_STEPS)
			break;


	}

	checkMemory();




#ifdef Animate_orbits
	cudaDeviceSynchronize();

	cudaMatrixf limiter_map_out(1024,1024);

	dim3 cudaBlockSize(16,16,1);
	dim3 cudaGridSize(1024/16,1024/16,1);

	CUDA_SAFE_KERNEL((generate_limiter_bitmap<<<cudaGridSize,cudaBlockSize>>>(
									plasma_d,limiter_map_out)));

	float2 gridorigins;
	float2 gridspacing_out;
	gridspacing_out.x = plasma_h.gridspacing.x;
	gridspacing_out.y = plasma_h.gridspacing.y;
	gridorigins.x = plasma_h.Rmin;
	gridorigins.y = plasma_h.Zmin;

	orbit_animate(xposition,yposition,nptcls_at_step,
						limiter_map_out,gridspacing_out,gridorigins,istep+1,minb,SPHERE_SPACING);
#endif
	cudaDeviceSynchronize();
	particles.XPlistFree();
	particles_old.XPlistFree();
	cudaFree(plasma_d);
	plasma_h.EnvironmentFree();




}


extern "C" void nubeam_gpu_cleanup_(int* idum)
{
	printf("Cleaning up GPU stuff\n");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaDeviceReset());
}

extern "C" void nubeam_gpu_init_(int* idum)
{
	printf("Initializing GPU stuff\n");
	CUDA_SAFE_CALL(cudaDeviceReset());
	CUDA_SAFE_CALL(cudaSetDevice(1));
	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
}






















