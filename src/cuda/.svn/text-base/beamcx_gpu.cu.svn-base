/*
 * This code handles
 */

#include "particleclass.cuh"
#include <iostream>



__gobal__
void XPlist_check_CX(XPlist particles_global,int* nptcls_cx,int* grid_offsets,int* block_offsets, int istep)
{
	/*
	 * This kernel checks to see which particles need to check for charge exchange events
	 * Then this kernel totals up the number of particles that need to undergo events.
	 * This kernel also sets up block_info_d for later calls
	 */

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	__shared__ int sdata[BLOCK_SIZE2];

	__shared__ int shift_param;

	__shared__ XPlist particles;

	particles.shift_local(particles_global);

	sdata[idx] = 1;

	if(idx < particles.nptcls)
	{
		if(istep >= particles.istep_next_cx[idx])
		{
			sdata[idx] = 0;
		}

	}
	__syncthreads();

	for(int k=BLOCK_SIZE2/2;k>0;k>>=1)
	{
		if((idx+k)<BLOCK_SIZE2)
		{
			sdata[idx]+=sdata[idx+k];
		}
		__syncthreads();
	}
	__syncthreads();

	if(idx == 0)
	{
		shift_param = sdata[0];
		grid_offsets[blockIdx.x] = sdata[0];
		nptcls_cx[blockIdx.x] = blockDim.x-sdata[0];
	}
	__syncthreads();

	block_offsets[gidx] = shift_param - sdata[idx];

}

__global__
void reduce_CX_offsets(int* grid_offsets,int* nptcls_cx,int n)
{
	unsigned int blockSize = BLOCK_SIZE2;
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int gridSize = blockSize*gridDim.x;
	unsigned int i = gidx;

	__shared__ int sdata[8192]; // Allocate all the shared memory

	__shared__ int shift_param;

	for(int j=0;j<7680;j+=blockSize){sdata[idx+j] = 0;}

	if(n <= BLOCK_SIZE2)
	{ // Do a reduce in shared memory

		if(gidx < n) sdata[idx] = grid_offsets[gidx];
		__syncthreads();
		for(int k=BLOCK_SIZE2/2;k>0;k>>=1)
		{
			if((idx+k)<n)
			{
				sdata[idx]+=sdata[idx+k];
			}
			__syncthreads();
		}

	}
	else if(n <= 8192)
	{// n > BLOCK_SIZE2, this means that we have to do a bigger reduction, we can still do it in shared memory though

		// First populate sdata
		while(i<n)
		{
			sdata[i] = grid_offsets[i];
			i+=gridSize;
		}
		__syncthreads();

		for(int k=4092;k>0;k>>=1)
		{
			i = gidx;
			while((i+k)<n)
			{
				sdata[i] += sdata[i+k];
				i+=gridSize;
			}
			__syncthreads();
		}
	}
	else
	{
		printf("!!Error!! more particles than I can handle, bigger reduce NYI \n");
		return;
	}

	i = gidx;

	__syncthreads();
	while (i < n)
	{
		grid_offsets[i] = sdata[0]-sdata[i];
		i += gridSize;
	}
	__syncthreads();

	// Now we reduce the nptcls_cx, and get our total number of CX particles

	sdata[idx] = 0;
	i = gidx;

	while (i < n) { sdata[idx] += nptcls_cx[i] + nptcls_cx[i+blockSize];  i += gridSize; }
	__syncthreads();
	if (idx < 256) { sdata[idx] += sdata[idx + 256]; } __syncthreads();
	if (idx < 128) { sdata[idx] += sdata[idx + 128]; } __syncthreads();
	if (idx <  64) { sdata[idx] += sdata[idx +  64]; } __syncthreads();
	if (idx < 32) {
		if (blockSize >=  64) sdata[idx] += sdata[idx + 32];
		if (blockSize >=  32) sdata[idx] += sdata[idx + 16];
		if (blockSize >=  16) sdata[idx] += sdata[idx +  8];
		if (blockSize >=   8) sdata[idx] += sdata[idx +  4];
		if (blockSize >=   4) sdata[idx] += sdata[idx +  2];
		if (blockSize >=   2) sdata[idx] += sdata[idx +  1];
	}
	if (idx == 0) nptcls_cx[0] = sdata[0];

}

__global__
void pop_out_for_CX(XPlist particles,XPlist particles_old,XPlist cx_particles,XPlist cx_particles_old,
									XPlist condensed_particles,XPlist condensed_particles_old,
									int* block_offsets,int* grid_offsets,int* old_ids,int istep)
{
	/*
	 * This kernel copies particle data for particles undergoing CX events to a temporary smaller list.
	 * The old_ids array is to keep track of where the CX particles came from in the main array so that they can be replaced
	 */

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	int iinc;

	int new_id;

	__shared__ int block_grid_offset;

	if(idx==0)block_grid_offset = grid_offsets[blockIdx.x];

	__syncthreads();

	if(gidx < particles.nptcls)
	{
		if(istep>particles.istep_next_cx[gidx])
		{
			iinc = rint( particles.cxskip[gidx]);
			if((particles.cxskip[gidx]-iinc)>curand_uniform(&particles.random_state[gidx]))
				{
					iinc+=1;
				}
			particles.istep_next_cx[gidx]+= iinc;



			new_id = gidx-(block_grid_offset+block_offsets[gidx]);
			old_ids[new_id] = gidx;

			particles.pop(cx_particles,new_id);
			particles_old.pop(cx_particles_old,new_id);
		}
		else
		{
			new_id = gidx-(block_grid_offset+block_offsets[gidx]);
			particles.pop(condensed_particles,new_id);
			particles_old.pop(condensed_particles_old,new_id);
		}
	}


}

template<int test_inside_plasma>
__global__
void beamcx_kernel(XPlist particles_global,cudaMatrixf nutrav_weight_in,cudaMatrixi ievent,
								cudaMatrixi nsplit,cudaMatrixi block_offsets,cudaMatrixi grid_offsets,int* nblocks)
{

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;

	realkind cxnsum;
	realkind splitting_factor;
	int isplit;
	realkind time_since_cx_old;
	realkind dtime_since_cx;


	realkind time_step;
	realkind dtime_step;

	realkind time_next;
	realkind time_left;



	int neutrals_remaining;
	int imul;
	int ireto;

	int jsplit;

	realkind weight_sum = 0.0;

	__shared__ XPlist particles;

	__shared__ realkind* nutrav_weight;

	__shared__ int sdata[BLOCK_SIZE];

	if(idx == 0) particles.shift_local(particles_global);
	if(idx == 0) nutrav_weight = &nutrav_weight_in(gidx,idBeam);

	__syncthreads();

	// First Get the probability that we have a charge exchange

	switch(test_inside_plasma)
	{
	case 0: // Particles are outside the plasma
		cxnsum = particles.cxnsum_outside(plasma);
		break;
	case 1: // Particles are inside the plasma
		cxnsum = particles.cxnsum_inside(plasma);
		break;
	default:
		break;
	}

	splitting_factor = particles.weight[idx]/plasma.average_beam_weight(plasma.lcenter_transp,idBeam)/plasma.average_weight_factor;

	splitting_factor = fmin(splitting_factor,0.25*plasma.max_particles);

	isplit = floor(splitting_factor);
	if(particles.get_random() < splitting_factor-isplit) isplit+=1;
	isplit = max(1,isplit);

	time_since_cx_old = particles.time_since_cx[idx];

	dtime_since_cx = isplit*cxnsum;

	particles.time_since_cx[idx] += dtime_since_cx;

	if(particles.time_since_cx[idx] <= particles.time_till_next_cx[idx])
	{
		ievent(gidx,idBeam) = 0;
	}
	else
	{
		// A cx event occurs
		ievent(gidx,idBeam) = 1;

		dtime_step = (particles.time_till_next_cx[idx]-time_since_cx_old)/dtime_since_cx*particles.cxdt_goosed[idx];
		time_step = dtime_step;

		neutrals_remaining = isplit-1;

		while(neutrals_remaining != 0)
		{
			time_next = -log(particles.get_random());
			time_left = (particles.cxdt_goosed[idx]-time_step)/particles.cxdt_goosed[idx]*(isplit-1)*dtime_since_cx;

			if(time_next > time_left)
			{
				particles.time_till_next_cx[idx] = time_next-time_left;
				weight_sum +=  (particles.cxdt_goosed[idx]-time_step)*neutrals_remaining*particles.weight[idx]/isplit;
				break;
			}
			else if(neutrals_remaining == 0)
			{
				ireto = 0;
				ievent(gidx,idBeam) = 2;
				break;
			}
			else
			{
				imul++;
				dtime_step = time_next/time_left*(particles.cxdt_goosed[idx]-time_step);
				weight_sum += dtime_step*neutrals_remaining*particles.weight[idx]/isplit;
				neutrals_remaining -= 1;
				time_step += dtime_step;
			}
		}

		switch(test_inside_plasma)
		{
		case 0: // Particles are outside the plasma
			cxnsum = particles.cxnsum_outside(plasma);
			break;
		case 1: // Particles are inside the plasma
			cxnsum = particles.cxnsum_inside(plasma);
			break;
		default:
			break;
		}

		// Total weight to be followed as fast cx neutrals in nutrav
		nutrav_weight[idx] = imul*particles.weight[idx]/isplit;

		// If orbiting particle survives, reduce its weight
		if(ireto == 1)
		{
			particles.weight[idx] -= imul*particles.weight[idx]/isplit;
		}

		jsplit = min(Max_Splits,max(rint(imul*plasma.cxsplit),1));

		if(jsplit < Max_Splits)
		{
			neutrals_remaining = plasma.cxsplit -jsplit;
			if(particles.get_random() < neutrals_remaining) jsplit += 1;
		}

		if(cxnsum == 0.0) jsplit -=1;

		nsplit(gidx,idBeam) = jsplit;
	}

	sdata[idx] = !(ievent(gidx,idBeam)==0);

	__syncthreads();

	for(int k=BLOCK_SIZE/2;k>0;k>>=1)
	{
		if((idx+k)<BLOCK_SIZE)
		{
			sdata[idx]+=sdata[idx+k];
		}
		__syncthreads();
	}
	__syncthreads();

	if(idx == 0)
	{
		grid_offsets(blockIdx.x,idBeam) = sdata[0];
		atomicAdd(nblocks,BLOCK_SIZE-sdata[0]);
	}
	__syncthreads();

	block_offsets(gidx,idBeam) = sdata[0] - sdata[idx];


}

__global__
void reduce_splitting_offsets(cudaMatrixi parent_ids,
												  cudaMatrixi block_offsets,
												  cudaMatrixi grid_offsets,
												  cudaMatrixi ievent,
												  int* lock,int nblocks,int nptcls)
{
	/*	This kernel sets up for the nutrav kernel. The
	 *
	 */

	unsigned int idx = threadIdx.x;
	unsigned int gidx = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;

	unsigned int newidx;

	__shared__ int sdata[BLOCK_SIZE2]; // Allocate all the shared memory

	sdata[idx] = 0;
	if(blockIdx.x == 0)
	{
		if(idx == 0) atomicExch(&lock[idBeam],1);
		if(gidx < nblocks) sdata[idx] = grid_offsets(gidx,idBeam);
		__syncthreads();
		for(int k=BLOCK_SIZE2/2;k>0;k>>=1)
		{
			if((idx+k)<n)
			{
				sdata[idx]+=sdata[idx+k];
			}
			__syncthreads();
		}

		__syncthreads();
		grid_offsets(gidx,idBeam) = sdata[0]-sdata[idx];

		__threadfence();

		if(idx == 0) atomicExch(&lock[idBeam],0);
	}
	__syncthreads();

	while(lock[idBeam])
	{
		__syncthreads();
	}

	if(gidx < nptcls)
	{
		newidx = gidx -(block_offsets(gidx,idBeam)+grid_offsets(blockIdx.x,idBeam));

		if(ievent(gidx,idBeam)) parent_ids(newidx,idBeam) = gidx;
		else parent_ids(newidx+nblocks,idBeam) = gidx;
	}
}

__global__
void pop_back_no_neutrals(XPlist particles,XPlist particles_old,
											   XPlist cx_particles,XPlist cx_particles_old,
											   cudaMatrixi parent_ids,int nblocks,int nparticles_cx)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx+nblocks;
	unsigned int idBeam = blockIdx.y;
	unsigned int idout;

	if(gidx < nparticles_cx)
	{
		idout = parent_ids(gidx,idBeam)+(particles.nptcls-nparticles_cx);
		cx_particles.pop(particles,idout);
		cx_particles_old.pop(particles_old,idout);

	}


}

__global__
void setup_nutrav(XPlist cx_particles, XPlist neutrals_global,cudaMatrixf nutrav_weight_in,
								 cudaMatrixi parent_ids,cudaMatrixi nsplit_global)
{

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;


	realkind split_weight = 0.0;
	int nsplit = nsplit_global(bidx,idBeam);
	realkind dphase_angle = 2*pi/nsplit;
	realkind normalized_weight;

	__shared__ XPlist neutrals;
	__shared__ XPlist parents;
	__shared__ realkind weight_sum[Max_Splits];


	parents.shared_parent(cx_particles,parent_ids(bidx,idBeam));
	neutrals.shift_local(neutrals_global);
	__syncthreads();

	if(idx < nsplit)
	{
		neutrals.copyfromparent(parents);

		if(neutrals.phase_angle[idx]+dphase_angle*idx < 2*pi)
		{
			neutrals.phase_angle[idx] += dphase_angle*idx;

		}
		else
		{
			neutrals.phase_angle[idx] +=dphase_angle*idx-2*pi;
		}

		neutrals.update_flr(plasma);

		switch(neutrals.check_outside)
		{
		case 0:
			split_weight = neutrals.cxnsum_outside(plasma);
			break;
		case 1:
			split_weight = neutrals.cxnsum_inside(plasma);
			break;
		default:
			break;
		}

		neutrals.orbflag[idx] = 1;
	}
	else
	{
		neutrals.orbflag[idx] = 0;
	}

	weight_sum[idx] = split_weight;

	normalized_weight = nutrav_weight_in(parent_ids(bidx,idBeam),idBeam)/(reduce<64>(weight_sum));

	if(idx < nsplit)
	{
		neutrals.weight[idx] *= normalized_weight;

	}

}

__global__
void nutrav_kernel(XPlist* neutrals_global,cudaMatrixT<realkind3> velocity_vector_out)
{
	unsigned int idx = threadIdx.x;
	unsigned int parent_idx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;

	if(neutrals_global[0].orbflag[parent_idx+girdDim.x*idBeam] == 0) return;

	// This kernel launches a separate thread for each track segment, each block is 1 neutral

	__shared__ XPlist neutral;
	__shared__ realkind p_recapture[Max_Track_segments];
	__shared__ realkind probability_integral[Max_Track_segments];
	__shared__ int recapture_point[Max_Track_segments];
	__shared__ realkind b_vector;

	neutral.shared_parent(*neutrals_global,parent_idx);

	__shared__ realkind3 v_vector;
	__shared__ realkind track_length;
	__shared__ realkind velocity;
	__shared__ realkind random_number;
	__shared__ realkind russian_roulette_num;

	realkind velocity_relative;
	realkind velocityR;
	realkind velocityZ;
	realkind velocityPhi;
	realkind plasma_velocity;
	realkind dt;

	realkind transp_zone;
	realkind beam_zone;

	realkind plasma_ionization_frac;
	realkind beam2_ionization_frac;
	realkind rr_factor;

	realkind dexp;

	realkind inverse_mfp;
	realkind temp_exp;

	__shared__ realkind3 position[Max_Track_segments];
	__shared__ realkind3 RZPhiposition[Max_Track_segments];


	// First thread sets up everthing
	if(idx == 0)
	{
		// Figure out what direction the Bfield is in
		v_vector = neutral.eval_Bvector(plasma.Psispline,plasma.gspline,0);

		// Figure out velocity vector

		v_vector = neutral.eval_velocity_vector(v_vector.x,v_vector.y,v_vector.z,1);

		velocity = sqrt(v_vector.x*v_vector.x+v_vector.y*v_vector.y+v_vector.z*v_vector.z);

		// Figure out the track length to the boundary of the grid

		track_length = neutral.find_neutral_track_length(v_vector,plasma.Rmin,plasma.Rmax,plasma.Zmin,plasma.Zmax);
		random_number = neutral.get_random();
		russian_roulette_num = neutral.get_random();
	}

	__syncthreads();
	// Find what position this thread is going to evaluate
	dt = (idx+1)*track_length/Max_Track_segments;
	position[idx].x = neutral.px[1][0]+dt*v_vector.x;
	position[idx].y = dt*v_vector.z;
	position[idx].z = neutral.py[1][0]+dt*v_vector.y;

	RZPhiposition[idx].x = sqrt(position[idx].x*position[idx].x+position[idx].y*position[idx].y);
	RZPhiposition[idx].y = position[idx].z;
	RZPhiposition[idx].z = atan(position[idx].y/position[idx].x);

	// Find relative energy in the plasma frame

	plasma_velocity = plasma.rotation(RZPhiposition[idx].x,RZPhiposition[idx].y)*RZPhiposition[idx].z;

	velocityR = position[idx].x*v_vector.x/RZPhiposition[idx].x+position[idx].y*v_vector.z/RZPhiposition[idx].x;
	velocityZ = v_vector.y;
	velocityPhi = -position[idx].z*v_vector.x/RZPhiposition[idx].x+position[idx].x*v_vector.z/RZPhiposition[idx].x;

	velocity_relative = velocityR*velocityR+velocityZ*velocityZ+pow(plasma_velocity-velocityPhi,2.0);

	transp_zone = plasma.transp_zone(RZPhiposition[idx].x,RZPhiposition[idx].y);
	beam_zone = plasma.beam_zone(RZPhiposition[idx].x,RZPhiposition[idx].y);

	// Interpolate table of plasma interactions
	plasma_ionization_frac = plasma.cx_cross_sections.thermal_total(velocity_relative*V2TOEV/neutral.mass[0],transp_zone,idBeam);

	// Interpolate table of Beam-Beam interactions

	beam2_ionization_frac = 0.0;

	for(int isb=0;isb<plasma.nbeams;isb++)
	{
		velocity_relative = velocityR*velocityR+
									 velocityZ*velocityZ+
									 pow(velocityPhi-plasma.toroidal_beam_velocity(beam_zone,isb),2.0);
		beam2_ionization_frac += plasma.cx_cross_sections.beam_beam_cx(
												velocity_relative*V2TOEV/neutral.mass[0],beam_zone,isb,idBeam);
		beam2_ionization_frac += plasma.cx_cross_sections.beam_beam_ii(
												velocity_relative*V2TOEV/neutral.mass[0],beam_zone,isb,idBeam);
	}

	inverse_mfp = (beam2_ionization_frac+plasma_ionization_frac)/velocity;

	probability_integral[idx] = inverse_mfp*velocity*track_length/Max_Track_segments;
	__syncthreads();

	// Sum up the probability integral
	for(int i=1;i<Max_Track_segments;i<<=1)
	{
		if(idx > (i-1))temp_exp = probability_integral[idx-i]+probability_integral[idx];
		__syncthreads();
		if(idx > (i-1))probability_integral[idx] = temp_exp;
		__syncthreads();
	}

	// Calculate the exponential
	if(probability_integral[idx] > 80.0)
	{
		p_recapture[idx] = 0.0;
	}
	else
	{
		p_recapture[idx] = exp(-probability_integral[idx]);
	}
	__syncthreads();
	if(idx > 0){dexp = p_recapture[idx-1]-p_recapture[idx];}
	else{dexp = p_recapture[idx];}
	__syncthreads();
	p_recapture[idx] = dexp;
	__syncthreads();

	for(int i=1;i<Max_Track_segments;i<<=1)
	{
		if(idx > (i-1))temp_exp = p_recapture[idx-i]+p_recapture[idx];
		__syncthreads();
		if(idx > (i-1))p_recapture[idx] = temp_exp;
		__syncthreads();
	}

	if(random_number > p_recapture[idx])
	{
		recapture_point[idx] = Max_Track_segments;
	}
	else
	{
		recapture_point[idx] = idx;
	}
	__syncthreads();

	if (idx <  64) { recapture_point[idx] = min(recapture_point[idx+64],recapture_point[idx]); }
	__syncthreads();
	if (idx < 32)
	{
		recapture_point[idx] = min(recapture_point[idx+32],recapture_point[idx]);
		recapture_point[idx] = min(recapture_point[idx+16],recapture_point[idx]);
		recapture_point[idx] = min(recapture_point[idx+8],recapture_point[idx]);
		recapture_point[idx] = min(recapture_point[idx+4],recapture_point[idx]);
		recapture_point[idx] = min(recapture_point[idx+2],recapture_point[idx]);
		recapture_point[idx] = min(recapture_point[idx+1],recapture_point[idx]);
	}

	__syncthreads();

	if(recapture_point[0] < Max_Track_segments)
	{
		// Recapture the neutral
		if(idx == recapture_point[0])
		{
			rr_factor = min(1.0,neutral.weight[0]/plasma.average_beam_weight(transp_zone,idBeam)/plasma.average_weight_factor);
			// Russian Roulette
			if(russian_roulette_num > rr_factor)
			{
				neutral.exit[0] = XPlistexit_russian_roulette;
			}
			else
			{
				neutral.px[1][0] = RZPhiposition[idx].x;
				neutral.py[1][0] = RZPhiposition[idx].y;
				v_vector.x = velocityR[idx];
				v_vector.y = velocityZ[idx];
				v_vector.z = velocityPhi[idx];

				velocity_vector_out(parent_idx,idBeam) = v_vector;
			}
		}
	}
	else
	{
		neutral.exit[0] = XPlistexit_neutralWallLoss;
	}
	return;
}

__global__
void recapture_neutrals(XPlist neutrals_global,cudaMatrixT<realkind3> velocityRZPhi,int nneutrals)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	__shared__ XPlist neutrals;

	neutrals.shift_local(neutrals_global);
	__syncthreads();

	if(gidx < nneutrals)
	{
		if(!(neutrals.exit[idx]))
		{

			neutrals.depsub(velocityRZPhi(gidx,idBeam),plasma.Phispline,plasma.Psispline,plasma.gspline);
		}

	}

}





__host__
void charge_exchange(XPlist &particles,XPlist &particles_old,XPlist* recaptured_neutrals,double dt,int nptcls,int nspecies,int istep)
{
	/*
	 * TODO
	 * 1. check for a charge exchange, count the number of particles to undergo CX, and prepare to pop them out
	 * 2. Second reduce on nptcls_CX, will also fully populate grid_offsets.
	 * 3. Populate the XPlist for charge exchange particles,
	 * 		remember their old indexes so they can be put back where they belong.
	 * 	4. Take reduced particle list, send it to the charge exchange kernel.
	 * 	5. Charge Exchange kernel figures out if the particle splits, also does some sums.
	 * 		- Also sums up the number of split particles
	 * 	6. setup_nutrav kernel sets up each of the split neutrals
	 * 	7. nutrav_kernel tracks each of the split neutrals till they are recaptured or lost
	 * 	8. deposit_neutrals kernel sets up the recaptured neutrals for re-entry into orbiting
	 */

	cudaError status;

	dim3 cudaGridSize(1,nspecies,1);
	dim3 cudaBlockSize(BLOCK_SIZE2,1,1);

	int cudaGridSizex = (BLOCK_SIZE2+nptcls-1)/BLOCK_SIZE2;

	int nptcls_cx_h;
	int nblocks_nutrav_h;

	cudaGridSize.x = cudaGridSizex;

	int* nptcls_cx_d;
	int* grid_offsets_d;
	int* block_offsets_d;
	int* old_pids_d;
	int* nblocks_nutrav_d;
	int* lock;


	cudaMalloc((void**)&nptcls_cx_d,cudaGridSizex*sizeof(int));
	cudaMalloc((void**)&grid_offsets_d,cudaGridSizex*sizeof(int));
	cudaMalloc((void**)&block_offsets_d,cudaGridSizex*sizeof(int));
	cudaMalloc((void**)&nblocks_nutrav_d,sizeof(int));
	cudaMalloc((void**)&lock,sizeof(int)*nspecies);

	XPlist_check_CX<<<cudaGridSize,cudaBlockSize>>>(particles,nptcls_cx_d,grid_offsets_d,block_offsets_d,istep);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "XPlist_check_CX %s\n", cudaGetErrorString(status));}


	 reduce_CX_offsets<<<1,512>>>(grid_offsets_d,nptcls_cx_d,cudaGridSizex);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "reduce_CX_offsets %s\n", cudaGetErrorString(status));}

	cudaMemcpy(&nptcls_cx_h,nptcls_cx_d,sizeof(int),cudaMemcpyDeviceToHost);

	XPlist cx_particles(nptcls_cx_h,device);
	XPlist cx_particles_old(nptcls_cx_h,device);
	XPlist condensed_particles(particles.nptcls,device);
	XPlist condensed_particles_old(particles.nptcls,device);

	cudaMalloc((void**)&old_pids_d,nptcls_cx_h*sizeof(int));

	cudaBlockSize.x = BLOCK_SIZE;
	cudaGridSize.x =  (BLOCK_SIZE+nptcls_cx_h-1)/BLOCK_SIZE;

	cudaMatrixf nutrav_weight_in(nptcls_cx_h,nspecies);
	cudaMatrixi ievent(nptcls_cx_h,nspecies);
	cudaMatrixi block_offsets_nutrav(nptcls_cx_h,nspecies);
	cudaMatrixi grid_offsets_nutrav(cudaGridSize.x,nspecies);
	cudaMatrixi nsplits(nptcls_cx_h,nspecies);
	cudaMatrixi parent_ids(nptcls_cx_h,nspecies);

	pop_out_for_CX<<<cudaGridSize,cudaBlockSize>>>(particles,particles_old,cx_particles,cx_particles_old,
																					  block_offsets_d,grid_offsets_d,old_pids_d,istep);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "pop_out_for_CX %s\n", cudaGetErrorString(status));}

	// Figure out which particles undergo cx and spawn neutrals

	beamcx_kernel<<<cudaGridSize,cudaBlockSize>>>(cx_particles,nutrav_weight_in,ievent,
																					 nsplits,block_offsets_nutrav,grid_offsets_nutrav,
																					 nblocks_nutrav_d);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "beamcx_kernel %s\n", cudaGetErrorString(status));}

	// Get how many neutrals each split spawns, total those up, generate new particle list for the split neutrals
	cudaMemcpy(&nblocks_nutrav_h,nblocks_nutrav_d,sizeof(int),cudaMemcpyDeviceToHost);

	cudaBlockSize.x = BLOCK_SIZE2;
	cudaGridSize.x =  (BLOCK_SIZE2+nptcls_cx_h-1)/BLOCK_SIZE2;

	reduce_splitting_offsets<<<cudaGridSize,cudaBlockSize>>>(parent_ids,block_offsets,grid_offsets,
																								  ievent,lock,nblocks_nutrav_h,nptcls_cx_h);

	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "reduce_splitting_offsets %s\n", cudaGetErrorString(status));}

	cudaBlockSize.x = BLOCK_SIZE2;
	cudaGridSize.x =  (BLOCK_SIZE2+(nptcls_cx_h-nblocks_nutrav_h)-1)/BLOCK_SIZE2;

	pop_back_no_neutrals<<<cudaGridSize,cudaBlockSize>>>(condensed_particles,condensed_particles.old,
																								cx_particles,cx_particles_old,parent_ids,
																								nblocks,nparticles_cx);

	// Launch 1 block per parent

	XPlist cx_neutrals(nblocks_nutrav_h*Max_Splits,nspecies,device);
	cudaMatrix<realkind3> recaptured_velocity_vector(Max_Splits*nblocks_nutrav_h,nspecies);

	cudaBlockSize.x = Max_Splits;
	cudaGridSize.x =  nblocks_nutrav_h;

	setup_nutrav<<<cudaGridSize,cudaBlockSize>>>(cx_particles,cx_neutrals,nutrav_weight_in,
																				   parent_ids,nsplits);

	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "setup_nutrav %s\n", cudaGetErrorString(status));}

	cudaBlockSize.x = Max_Track_segments;
	cudaGridSize.x =  Max_Splits*nblocks_nutrav_h;

	nutrav_kernel<<<cudaGridSize,cudaBlockSize>>>(cx_neutrals,recaptured_velocity_vector);

	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "setup_nutrav %s\n", cudaGetErrorString(status));}

	// Setup the new ions that came from the recpture
	cudaBlockSize.x = BLOCK_SIZE2;
	cudaGridSize.x =  (BLOCK_SIZE2+Max_Splits*nblocks_nutrav_h-1)/BLOCK_SIZE2;
	recapture_neutrals<<<cudaGridSize,cudaBlockSize>>>(cx_neutrals,recaptured_velocity_vector,
																						   Max_Splits*nblocks_nutrav);

	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "recapture neutrals %s\n", cudaGetErrorString(status));}

	*recaptured_neutrals = cx_neutrals;
	particles = condensed_particles;
	particles_old = condensed_particles_old;

	return;

}




