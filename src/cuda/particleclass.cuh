/*
 *  Particleclass header file
 */

#include "fieldclass.cuh"

enum coordinatesystem
{
	rz = 0,
	xy = 1,
	xipsi = 2

};


// Define Allocation functor classes

class XPMallocHost
{
public:
	__host__ void operator() (void** Ptr,size_t* pitch,int width,int height)
	{
		CUDA_SAFE_CALL(cudaHostAlloc(Ptr,width*height,4));
	}
};

class XPMallocDevice
{
public:
	__host__ void operator() (void** Ptr,size_t* pitch,int width,int height)
	{
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMallocPitch((void**)Ptr,pitch,width,height));
		//CUDA_SAFE_CALL(cudaMemset2D(*Ptr,*pitch,0,width,height));
	}
};

enum XPlistlocation
{
	XPlistlocation_host = 0,
	XPlistlocation_device = 1,
	XPlistlocation_shared = 2
};

enum XPlistexits
{
	XPlistexit_stillorbiting = 0,
	XPlistexit_newparticle = 1,
	XPlistexit_neutralWallLoss = 2,
	XPlistexit_outsideplasma = 3,
	XPlistexit_russian_roulette = 4,
	XPlistexit_neverorbiting = 5,
	XPlistexit_limiter = 6,
	XPlistexit_leftgrid = 7,
	XPlistexit_time = 8,
	XPlistexit_dtmin = 9,
	XPlistexit_neutralized = 10,
	XPlistexit_thermalized = 11
};


class XPlist
{
public:
	realkind* px[2];
	realkind* py[2];

	realkind* vperp[2];
	realkind* vpara[2];
	realkind* pitch_angle[2];
	realkind* phase_angle;

	realkind* energy;
	realkind* potential[2];
	realkind* momentum;
	realkind* rlarmor;
	realkind* mu;

	realkind* weight; // MC particle weight

	int* mass;
	int* charge;
	int* atomic_number;

	int* nx[2];
	int* ny[2];
	int* cellindex[2];

	int* beam_source; // beam source index

	realkind* cxskip; // Skip factor on Beam Ion CX calculation
	realkind* time_till_next_cx; // Time till next Charge Exchange collision
	realkind* time_since_cx; // Charge Exchange Probability integral
	realkind* time_since_cx_pass;
	realkind* cxdt_goosed; // Goosed dt
	int* istep_next_cx; // Time Steps till next Charge Exchange
	int* cx_count;

	realkind* fpskip;
	realkind* time_till_next_fp;
	realkind* time_since_fp;
	realkind* fpdt_goosed;
	int* istep_next_fp;
	int* fp_count;

	realkind* time_done;
	realkind* deltat;

	int* steps_midplanecx;



	int* old_idx;
	int* original_idx;
	int* nbndex;

	int* orbflag; // particle still orbiting

	enum XPlistexits* pexit;

	curandState* random_state; // random number generator state

	int* nptcls; // nptcls for each species
	int nspecies;
	int nptcls_max; // maximum number of particles for any species
	int nptcls_allocated;
	int* species;

	int blockwidth;
	size_t nbi_fpitch;
	size_t nbi_ipitch;
	size_t nbi_curpitch;
	size_t nbi_epitch;

	static const int nints = 19;
	static const int nrealkinds = 29;

	__host__ __device__ XPlist(){;}

	__host__ XPlist(int nptcls_in,int nspecies_in, enum XPlistlocation location_in)
	{XPlist_init(nptcls_in,nspecies_in,location_in);}


	// Allocate particle list in specified memory
	__host__ void XPlist_init(int nptcls_in,int nspecies_in, enum XPlistlocation location_in)
	{
		nptcls_max = nptcls_in;
		nptcls_allocated =nptcls_in;
		nspecies = nspecies_in;
		location = location_in;
		if(nptcls_allocated > 0)
		{
			if(!location_in)
			{
				printf("allocating particle list on the host \n");
				XPlist_allocate(XPMallocHost());
			}
			else
			{
				printf("allocating particle list on the device \n");
				XPlist_allocate(XPMallocDevice());
			}
		}
	}
	template<class O>
	__host__ void XPlist_allocate(O op);
	// Populate the list with data from the host
	__host__
	void setup(Environment* plasma,double** dbl_data_in_h,
						int** int_data_in_h,int minb,int mibs);
	__host__ void copyFromHost(double* data_in_h);
	// Method to calculate the space and Vpara derivitives for the rk4 solver
	__device__
	realkind3 XPlist_derivs(XPTextureSpline Psispline,XPTextureSpline gspline,XPTextureSpline Phispline,realkind mu,int ilocation);
	__host__ __device__
	realkind** get_realkind_pointer(int iptr);
	__host__ __device__
	int** get_int_pointer(int iptr);
	// Method to sort the particle list (list must reside in device memory)
	__host__ void sort(realkind2 Pgridspacing, int2 Pgrid_i_dims);
	// Wrapper method for find_index_kernel()
	__host__ void find_cell_index(int* cellindex_temp,size_t ipitch,int cell_max);
	// Create a particle list in shared memory
	__device__ void allocate_shared(void);
	// Allocate space for 1 parent particle in shared memory
	__device__ void shared_parent(XPlist* parent_global,int blockstart);
	// Copy particle data from a parent particle
	__device__ void copyfromparent(XPlist* parent);
	// Copy a particle in global memory to shared
	__device__ void copyfromglobal(XPlist listin);
	// Copy a particle in shared memory to global
	__device__ void copyfromshared(XPlist listin);
	__device__ void calc_binid(Environment* plasma_in,int icall,int idx);
	// Check to see if a particle is still orbiting
	__device__ int check_orbit(Environment* plasma,int icall=0);
	// Check to see if the particle is still inside the plasma 1 for inside, 0 for outside
	__device__ int check_outside(Environment* plasma);

	// Evaluate Bmod;
	__device__ realkind eval_Bmod(XPTextureSpline Psispline,XPTextureSpline gspline);
	// Evaluate Mu
	__device__ realkind eval_mu(XPTextureSpline Psispline,XPTextureSpline gspline);
	// Evaluate a timestep
	__device__ realkind eval_dt(Environment* plasma_in);

	// Shift the pointers so that the pointer starts with the first particle in that block
	__device__ void shift_local(XPlist* global_particle,unsigned int block_start);

	template<int direction>
	__device__ void shift_shared(XPlist* local_particles,realkind* sdata,int* ireals,int nreals);

	// Check to see if a particle is going to go through charge exchange

	// Pop out my particle, and put it into the output list at new_idx
	__device__ void pop(XPlist* listout,int new_idx);

	__host__
	void append(XPlist listin);

	// Method to evaluate B
	__device__ realkind3 Bvector(realkind r,realkind dPsidR,realkind dPsidZ,realkind g,int ilocation);
	__device__ realkind3 eval_Bvector(XPTextureSpline Psispline,XPTextureSpline gspline,int ilocation);

	// n0*sigmav for particles outside the plasma.
	__device__ realkind cxnsum_outside(Environment* plasma);

	// n0*sigmav for particles inside the plasma
	__device__ realkind cxnsum_inside(Environment* plasma);

	// Evaluate the 3 components of vperp at the finite larmor radius
	__device__ realkind3 eval_vperp_vector(realkind Br, realkind Bz, realkind Bphi,int ilocation);

	__device__ realkind3 eval_velocity_vector(realkind Br,realkind Bz, realkind Bphi, int ilocation);

	// Evalute the larmor radius vector
	__device__ realkind3 eval_larmor_vector(realkind Br,realkind Bz, realkind Bphi);
	// update the values at the finite larmor radius
	__device__ void update_flr(Environment* plasma);

	// Figure out the neutral track length to the grid boundaries
	__device__ realkind find_neutral_track_length(realkind3 velocity_vector,
																		   realkind rmin,realkind rmax,
																		   realkind zmin,realkind zmax);
	__device__
	void depsub(realkind3 velocityRZPhi,XPTextureSpline Phispline,
						  XPTextureSpline Psispline,XPTextureSpline gspline,
						  unsigned int idx);

	__device__
	realkind3 eval_plasma_frame_velocity(Environment* plasma_in);

	__device__
	int collide(Environment* plasma_in,int isteps);

	__device__
	void anomalous_diffusion(Environment* plasma_in);

	__device__
	void bounce_init(int jcall,Environment* plasma_in);

	__device__
	void gphase(void);

	__device__
	void update_gc(Environment* plasma_in);

	__device__
	realkind4 find_jacobian(XPTextureGrid* Xi_map,XPTextureGrid* Theta_map);

	__device__
	realkind3 eval_rlarmor_vector(realkind Br, realkind Bz, realkind Bphi);

	__device__
	int check_midplane_cx(Environment* plasma_in,XPlist* particles_old);

	__device__
	void update_timing_factors(Environment* plasma_in);

	__device__
	int eval_NGC(Environment* plasma_in,int idx,int location = 0);

	__host__
	XPlist split(cudaMatrixui split_condition_d,
					cudaMatrixui &old_ids0,cudaMatrixui &old_ids1,const int old_ids_out,int forceflag);

	__host__
	XPlist split(cudaMatrixui split_condition1,int forceflag1=0)
	{
		XPlist result;
		cudaMatrixui old_ids0;
		cudaMatrixui old_ids1;
		result =  split(split_condition1,old_ids0,old_ids1,0,forceflag1);

		old_ids0.cudaMatrixFree();
		old_ids1.cudaMatrixFree();

		return result;

	}

	__device__
	void print_members(int idx = threadIdx.x,
			int gidx = threadIdx.x+blockIdx.x*blockDim.x,
			int idBeam = blockIdx.y);
	__host__
	void check_sort(void);


	// Get a random number
	__device__ realkind get_random(void);


	// Destructor
	__host__ void XPlistFree(void)
	{
		if((nptcls_allocated > 0))
		{
			void* ptr;

			for(int i=0;i<nrealkinds;i++)
			{
				ptr = *(get_realkind_pointer(i));
				CUDA_SAFE_CALL(cudaFree(ptr));

			}

			for(int i=0;i<nints;i++)
			{
				ptr = *(get_int_pointer(i));
				CUDA_SAFE_CALL(cudaFree(ptr));
			}

			CUDA_SAFE_CALL(cudaFree(pexit));
			CUDA_SAFE_CALL(cudaFree(random_state));
			CUDA_SAFE_CALL(cudaFree(nptcls));
			CUDA_SAFE_CALL(cudaFree(species));
			nptcls_allocated = 0;
		}

	}



private:
	enum XPlistlocation location;
	enum coordinatesystem* cs; //coordinate system indicator

};



/*
__device__
double& XPlist::get_px(void)
{
	if(location == shared)
	{
		return &px[threadIdx.x];
	}
	else
	{
		return &px[blockinfo_d[blockIdx.x].y+threadIdx.x];
	}
}
__device__
const double& XPlist::get_px(void)
const
{
	if(location == shared)
	{
		return &px[threadIdx.x];
	}
	else
	{
		return &px[blockinfo_d[blockIdx.x].y+threadIdx.x];
	}
}
__device__
double & XPlist::get_py(void)
{
	if(location == shared)
	{
		return py+threadIdx.x;
	}
	else
	{
		return py+blockinfo_d[blockIdx.x].y+threadIdx.x;
	}
}
__device__
const double & XPlist::get_py(void)
const
{
	if(location == shared)
	{
		return py+threadIdx.x;
	}
	else
	{
		return py+blockinfo_d[blockIdx.x].y+threadIdx.x;
	}
}
__device__
double & XPlist::get_vpara(void)
{
	if(location == shared)
	{
		return vpara+threadIdx.x;
	}
	else
	{
		return vpara+blockinfo_d[blockIdx.x].y+threadIdx.x;
	}
}
__device__
const double & XPlist::get_vpara(void)
const
{
	if(location == shared)
	{
		return &vpara[threadIdx.x];
	}
	else
	{
		return &vpara[blockinfo_d[blockIdx.x].y+threadIdx.x];
	}
}
__device__
double & XPlist::get_vperp(void)
{
	if(location == shared)
	{
		return &vperp[threadIdx.x];
	}
	else
	{
		return &vperp[blockinfo_d[blockIdx.x].y+threadIdx.x];
	}
}
__device__
const double & XPlist::get_vperp(void)
const
{
	if(location == shared)
	{
		return &vperp[threadIdx.x];
	}
	else
	{
		return &vperp[blockinfo_d[blockIdx.x].y+threadIdx.x];
	}
}
__device__
int & XPlist::get_nx(void)
{
	return &nx[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
int & XPlist::get_ny(void)
{
	return &ny[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
int & XPlist::get_cellindex(void)
{
	return &cellindex[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
int & XPlist::get_mass(void)
{
	return &mass[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
int & XPlist::get_charge(void)
{
	return &charge[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
int & XPlist::get_atomic_number(void)
{
	return &atomic_number[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
int & XPlist::get_source(void)
{
	return &source[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_nx(void)
const
{
	return &nx[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_ny(void)
const
{
	return &ny[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_charge(void)
const
{
	return &charge[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_cellindex(void)
const
{
	return &cellindex[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_mass(void)
const
{
	return &mass[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_atomic_number(void)
const
{
	return &atomic_number[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
__device__
const int & XPlist::get_source(void)
const
{
	return &source[blockinfo_d[blockIdx.x].y+threadIdx.x];
}
*/


__host__ void XPlistCopy(XPlist dst, XPlist src,int nptcls_in,int nspecies_in, enum cudaMemcpyKind kind)
{
	int nrealkinds = src.nrealkinds;
	int nints = src.nints;
	int* intptr_in;
	int* intptr_out;
	realkind* fltptr_in;
	realkind* fltptr_out;

	int width = nptcls_in;
	int height = nspecies_in;

	if(nptcls_in > dst.nptcls_allocated)
	{
		printf("Error, trying to copy to a particle list that is too small \n");
		return;
	}

	cudaDeviceSynchronize();

	for(int i=0;i<nrealkinds;i++)
	{
		//printf("getting pointer %i\n",i);
		fltptr_in = *(src.get_realkind_pointer(i));
		fltptr_out = *(dst.get_realkind_pointer(i));
		CUDA_SAFE_CALL(cudaMemcpy2D(fltptr_out,dst.nbi_fpitch,fltptr_in,src.nbi_fpitch,width*sizeof(realkind),height,kind));
	}

	for(int i=0;i<nints;i++)
	{
		intptr_in = *(src.get_int_pointer(i));
		intptr_out = *(dst.get_int_pointer(i));
		CUDA_SAFE_CALL(cudaMemcpy2D(intptr_out,dst.nbi_ipitch,intptr_in,src.nbi_ipitch,width*sizeof(int),height,kind));
	}

	CUDA_SAFE_CALL(cudaMemcpy2D(dst.random_state,dst.nbi_curpitch,src.random_state,src.nbi_curpitch,width*sizeof(curandState),height,kind));

	CUDA_SAFE_CALL(cudaMemcpy2D(dst.pexit,dst.nbi_epitch,src.pexit,src.nbi_epitch,width*sizeof(enum XPlistexits),height,kind));

	CUDA_SAFE_CALL(cudaMemcpy(dst.nptcls,src.nptcls,src.nspecies*sizeof(int),cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dst.species,src.species,src.nspecies*sizeof(int),cudaMemcpyDeviceToDevice));


}

__host__ __device__
realkind** XPlist::get_realkind_pointer(int iptr)
{
	realkind** fltptrs;

	switch(iptr)
	{
	case 0:
		fltptrs = &px[0];
		break;
	case 1:
		fltptrs = &px[1];
		break;
	case 2:
		fltptrs = &py[0];
		break;
	case 3:
		fltptrs = &py[1];
		break;
	case 4:
		fltptrs = &vperp[0];
		break;
	case 5:
		fltptrs = &vperp[1];
		break;
	case 6:
		fltptrs = &vpara[0];
		break;
	case 7:
		fltptrs = &vpara[1];
		break;
	case 8:
		fltptrs = &pitch_angle[0];
		break;
	case 9:
		fltptrs = &pitch_angle[1];
		break;
	case 10:
		fltptrs = &phase_angle;
		break;
	case 11:
		fltptrs = &energy;
		break;
	case 12:
		fltptrs = &potential[0];
		break;
	case 13:
		fltptrs = &potential[1];
		break;
	case 14:
		fltptrs = &momentum;
		break;
	case 15:
		fltptrs = &rlarmor;
		break;
	case 16:
		fltptrs = &mu;
		break;
	case 17:
		fltptrs = &weight;
		break;
	case 18:
		fltptrs = &cxskip;
		break;
	case 19:
		fltptrs = &time_till_next_cx;
		break;
	case 20:
		fltptrs = &time_since_cx;
		break;
	case 21:
		fltptrs = &time_since_cx_pass;
		break;
	case 22:
		fltptrs = &cxdt_goosed;
		break;
	case 23:
		fltptrs = &fpskip;
		break;
	case 24:
		fltptrs = &time_till_next_fp;
		break;
	case 25:
		fltptrs = &time_since_fp;
		break;
	case 26:
		fltptrs = &fpdt_goosed;
		break;
	case 27:
		fltptrs = &time_done;
		break;
	case 28:
		fltptrs = &deltat;
		break;
	default:
		break;
	}

	return fltptrs;


}
__host__ __device__
int** XPlist::get_int_pointer(int iptr)
{
	int** intptrs;

	switch(iptr)
	{
	case 0:
		intptrs = &mass;
		break;
	case 1:
		intptrs = &charge;
		break;
	case 2:
		intptrs = &atomic_number;
		break;
	case 3:
		intptrs = &nx[0];
		break;
	case 4:
		intptrs = &nx[1];
		break;
	case 5:
		intptrs = &ny[0];
		break;
	case 6:
		intptrs = &ny[1];
		break;
	case 7:
		intptrs = &cellindex[0];
		break;
	case 8:
		intptrs = &cellindex[1];
		break;
	case 9:
		intptrs = &beam_source;
		break;
	case 10:
		intptrs = &istep_next_cx;
		break;
	case 11:
		intptrs = &cx_count;
		break;
	case 12:
		intptrs = &istep_next_fp;
		break;
	case 13:
		intptrs = &fp_count;
		break;
	case 14:
		intptrs = &steps_midplanecx;
		break;
	case 15:
		intptrs = &old_idx;
		break;
	case 16:
		intptrs = &orbflag;
		break;
	case 17:
		intptrs = &original_idx;
		break;
	case 18:
		intptrs = &nbndex;
	default:
		break;
	}

	return intptrs;
}

__device__
void XPlist::shift_local(XPlist* global_particle,unsigned int block_start=blockIdx.x*blockDim.x)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int block_size = blockDim.x;

	unsigned int ibidx = (idBeam*(global_particle -> nbi_ipitch));
	unsigned int fbidx = (idBeam*(global_particle -> nbi_fpitch));
	unsigned int enbidx = (idBeam*(global_particle -> nbi_epitch));
	unsigned int cubidx = (idBeam*(global_particle -> nbi_curpitch));

	realkind** fltptr_out;
	realkind* fltptr_in;
	int** intptr_out;
	int* intptr_in;

	while(idx < (nrealkinds+nints+3))
	{
		if(idx < nrealkinds)
		{
			fltptr_out = get_realkind_pointer(idx);
			fltptr_in = *(global_particle -> get_realkind_pointer(idx));
			*fltptr_out = ((realkind*)((char*)fltptr_in+fbidx)+block_start);
		}
		else if((idx-nrealkinds) < nints)
		{
			intptr_out = get_int_pointer(idx-nrealkinds);
			intptr_in =*(global_particle -> get_int_pointer(idx-nrealkinds));
			*intptr_out = ((int*)((char*)intptr_in+ibidx)+block_start);
		}
		else if(idx == (nrealkinds+nints))
		{
			pexit = (enum XPlistexits*)((char*)(global_particle->pexit)+enbidx)+block_start;
		}
		else if(idx == (nrealkinds+nints+1))
		{
			random_state = (curandState*)((char*)(global_particle -> random_state)+cubidx)+block_start;
		}
		else if(idx == (nrealkinds+nints+2))
		{
			nspecies = 1;

			if((blockIdx.x+1)*blockDim.x >=(global_particle -> nptcls[idBeam]))
				nptcls_max = (global_particle->nptcls[idBeam])-blockIdx.x*blockDim.x;
			else
				nptcls_max = blockDim.x;

			nptcls = global_particle->nptcls;
			species = global_particle->species;
		}
		idx+= block_size;
	}


}


__device__
void XPlist::shared_parent(XPlist* parent_global,int blockstart)
{
	__shared__ realkind realdata[nrealkinds];
	__shared__ int intdata[nints];
	__shared__ enum XPlistexits s_pexit;
	__shared__ curandState s_random_state;

	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int block_size = blockDim.x;

	unsigned int ibidx = (idBeam*(parent_global -> nbi_ipitch));
	unsigned int fbidx = (idBeam*(parent_global -> nbi_fpitch));
	unsigned int enbidx = (idBeam*(parent_global -> nbi_epitch));
	unsigned int cubidx = (idBeam*(parent_global -> nbi_curpitch));

	realkind** fltptr_out;
	realkind* fltptr_in;
	int** intptr_out;
	int* intptr_in;

	while(idx < (nrealkinds+nints+3))
	{
		if(idx < nrealkinds)
		{
			fltptr_out = get_realkind_pointer(idx);
			*fltptr_out = &realdata[idx];
			fltptr_in = *(parent_global -> get_realkind_pointer(idx));
			**fltptr_out = *((realkind*)((char*)fltptr_in+fbidx)+blockstart);
		}
		else if((idx-nrealkinds) < nints)
		{
			intptr_out = get_int_pointer(idx-nrealkinds);
			*intptr_out = &intdata[idx-nrealkinds];
			intptr_in =*(parent_global -> get_int_pointer(idx-nrealkinds));
			**intptr_out = *((int*)((char*)intptr_in+ibidx)+blockstart);
		}
		else if(idx == (nrealkinds+nints))
		{
			pexit = &s_pexit;
			*pexit = *((enum XPlistexits*)((char*)(parent_global->pexit)+enbidx)+blockstart);
		}
		else if(idx == (nrealkinds+nints+1))
		{
			random_state = &s_random_state;
			*random_state = *((curandState*)((char*)(parent_global -> random_state)+cubidx)+blockstart);
		}
		else if(idx == (nrealkinds+nints+2))
		{
			nspecies = 1;
			nptcls_max = 1;
			nptcls = parent_global->nptcls;
			species = parent_global->species;
		}

		idx+= block_size;

	}

}

template<int direction>
__device__
void XPlist::shift_shared(XPlist* local_particles,realkind* sdata,int* ireals,int nreals)
{
	uint idx = threadIdx.x;
	uint thid = idx;
	int offset = blockDim.x;
	if(direction == 0)
	{
		while(thid < nreals)
		{
			*get_realkind_pointer(ireals[thid]) = sdata+thid*offset;
			thid+= blockDim.x;
		}

		__syncthreads();

		if(idx < local_particles->nptcls_max)
		{
			for(int i=0;i<nreals;i++)
			{
				int ireal = ireals[i];
				(*get_realkind_pointer(ireal))[idx] = (*local_particles->get_realkind_pointer(ireal))[idx];
			}
		}
	}
	else
	{
		if(idx < local_particles->nptcls_max)
		{
			for(int i=0;i<nreals;i++)
			{
				int ireal = ireals[i];
				(*local_particles->get_realkind_pointer(ireal))[idx] = (*get_realkind_pointer(ireal))[idx];
			}
		}
	}


}

__device__
void XPlist::copyfromparent(XPlist* parent)
{
	unsigned int idx = threadIdx.x;
	int element_offset = idx+blockIdx.x*blockDim.x+original_idx_counter_d;
	int skipahead_amount = element_offset*curand_uniform(parent->random_state);
	element_offset += Max_Splits*curand_uniform(parent->random_state);


	int* intptr_in;
	int* intptr_out;
	realkind* fltptr_in;
	realkind* fltptr_out;

	for(int i=0;i<nrealkinds;i++)
	{
		fltptr_in = *(parent->get_realkind_pointer(i));
		fltptr_out = *(get_realkind_pointer(i));
		fltptr_out[idx] = fltptr_in[0];
	}

	for(int i=0;i<nints;i++)
	{
		intptr_in = *(parent->get_int_pointer(i));
		intptr_out = *(get_int_pointer(i));
		intptr_out[idx] = intptr_in[0];
	}

	pexit[idx] = parent->pexit[0];
	random_state[idx] = parent->random_state[0];

	skipahead_sequence(element_offset,&(random_state[idx]));
	skipahead(skipahead_amount,&(random_state[idx]));

}

__device__
void XPlist::pop(XPlist* listout,int new_idx)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;

	unsigned int iidxin = (idBeam*(nbi_ipitch));
	unsigned int fidxin = (idBeam*(nbi_fpitch));
	unsigned int enidxin = (idBeam*(nbi_epitch));
	unsigned int cuidxin = (idBeam*(nbi_curpitch));

	unsigned int iidxout = (idBeam*(listout -> nbi_ipitch));
	unsigned int fidxout = (idBeam*(listout -> nbi_fpitch));
	unsigned int enidxout = (idBeam*(listout -> nbi_epitch));
	unsigned int cuidxout = (idBeam*(listout -> nbi_curpitch));

	int* intptr_in;
	int* intptr_out;
	realkind* fltptr_in;
	realkind* fltptr_out;

	for(int i=0;i<nrealkinds;i++)
	{
		fltptr_in = *(get_realkind_pointer(i));
		fltptr_out = *(listout -> get_realkind_pointer(i));
		*((realkind*)((char*)fltptr_out+fidxout)+new_idx) = *((realkind*)((char*)fltptr_in+fidxin)+gidx);
	}

	for(int i=0;i<nints;i++)
	{
		intptr_in = *(get_int_pointer(i));
		intptr_out = *(listout -> get_int_pointer(i));
		*((int*)((char*)intptr_out+iidxout)+new_idx) = *((int*)((char*)intptr_in+iidxin)+gidx);
	}

	*((enum XPlistexits*)((char*)listout->pexit+enidxout)+new_idx) = *((enum XPlistexits*)((char*)pexit+enidxin)+gidx);
	*((curandState*)((char*)(listout -> random_state)+cuidxout)+new_idx) = *((curandState*)((char*)(random_state)+cuidxin)+gidx);

}

__global__
void reduce_split_blockoffsets(cudaMatrixui grid_offsets,cudaMatrixui block_offsets,
										   cudaMatrixui split_condition,int* nptcls_in)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;

	__shared__ unsigned int nptcls;

	if(idx == 0) nptcls = nptcls_in[idBeam];
	__syncthreads();

	if(nptcls <= 0){return;}

	__shared__ unsigned int sdata[BLOCK_SIZE2];

	unsigned int tdata;

	unsigned int* sptr = sdata;

	sdata[idx] = 0;
	__syncthreads();

	if(gidx<nptcls)
		sdata[idx] = min(1,max(0,split_condition(gidx,idBeam)));
	else
		sdata[idx] = 0;

	__syncthreads();

	for(int k=BLOCK_SIZE2/2;k>0;k>>=1)
	{
		if((idx+k)<BLOCK_SIZE2)
		{
			tdata = sptr[idx]+sptr[idx+k];
		}
		__syncthreads();
		if((idx+k)<BLOCK_SIZE2)
		{
			sptr[idx] = tdata;
		}
		__syncthreads();
	}
	__syncthreads();

	if(gidx<nptcls)
		block_offsets(gidx,idBeam) = sdata[0]-sdata[idx];

	if(idx == 0)
		grid_offsets(bidx,idBeam) = sdata[0];


}

__global__
void reduce_split_gridoffsets(cudaMatrixui grid_offsets,cudaMatrixui block_offsets,
												  int* nptcls_0,int* nptcls_1,int* nptcls_in,
												  int* nptcls_0_max,int* nptcls_1_max,int nblocks)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;
	int i = idx;

	__shared__ unsigned int sdata[1024];

	unsigned int tdata[4];

	unsigned int* sptr = sdata;

	while(i<1024)
	{
		sdata[i] = 0;
		i+=BLOCK_SIZE2;
	}
	i = idx;

	__syncthreads();

	if(nblocks > 1024)
	{
		//printf("!!Error!! more particles than splitting can handle, implement a bigger reduce \n");
		return;
	}

	while(i<nblocks)
	{
		sdata[i] = grid_offsets(i,idBeam);
		i+=BLOCK_SIZE2;
	}
	__syncthreads();

	for(int k=512;k>0;k>>=1)
	{
		i = idx;
		while((i+k)<nblocks)
		{
			tdata[i] = sptr[i]+sptr[i+k];
			i+=BLOCK_SIZE2;
		}
		__syncthreads();
		i = idx;
		while((i+k)<nblocks)
		{
			sptr[i] = tdata[i];
			i+=BLOCK_SIZE2;
		}
		__syncthreads();
	}
	__syncthreads();

	if(gidx == 0)
	{
		nptcls_1[idBeam] = sdata[0];
		nptcls_0[idBeam] = nptcls_in[idBeam] - sdata[0];

#ifdef debug
		printf("for idBeam %i, nptcls0 = %i, nptcls1 = %i, nptcls_in = %i\n",
				idBeam,nptcls_0[idBeam],nptcls_1[idBeam],nptcls_in[idBeam]);
#endif


		atomicMax(&(nptcls_0_max[0]),nptcls_0[idBeam]);
		atomicMax(&(nptcls_1_max[0]),nptcls_1[idBeam]);

	}

	i = idx;
	while(i < nblocks)
	{
		grid_offsets(i,idBeam) = sdata[0]-sdata[i];
		i+=BLOCK_SIZE2;
	}

}

__global__
void split_list_kernel(XPlist* parent_list,XPlist* particles0,XPlist* particles1,
									cudaMatrixui old_ids0,cudaMatrixui old_ids1,
									cudaMatrixui grid_offsets,cudaMatrixui block_offsets,
									cudaMatrixui splitting_condition,const int old_ids_out)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;

	unsigned int new_id;
	unsigned int split_condition;

	if(gidx < parent_list->nptcls[idBeam])
		block_offsets(gidx,idBeam) += grid_offsets(bidx,idBeam);



	if(gidx < parent_list->nptcls[idBeam])
	{
		split_condition = splitting_condition(gidx,idBeam);
			if(split_condition == 0)
			{
				new_id = gidx-block_offsets(gidx,idBeam);
				if(new_id < particles0->nptcls[idBeam])
				{
					parent_list->pop(particles0,new_id);

					if(old_ids_out > 0)
					{
						old_ids0(new_id,idBeam) = gidx;
					}
				}
				else
				{
#ifdef debug
					printf("Error new_id %i >= nptcls0 %i\n",new_id,particles0->nptcls[idBeam]);
#endif
				}


			}
			else if(split_condition == 1)
			{
				new_id = block_offsets(gidx,idBeam);
				if(new_id < particles1->nptcls[idBeam])
				{
					parent_list->pop(particles1,new_id);

					if(old_ids_out > 0)
					{
						old_ids1(new_id,idBeam) = gidx;
					}
				}
				else
				{
#ifdef debug
					printf("Error new_id %i >= nptcls1 %i\n",new_id,particles1->nptcls[idBeam]);
#endif
				}
			}
	}

	if(gidx == 0)
	{
	//	printf("particles_0 = %i, %i\nparticles_1 = %i, %i\n",
	//			particles0.nptcls[idBeam],particles0.nptcls_max,
	//			particles1.nptcls[idBeam],particles1.nptcls_max);
	}

}

__host__
XPlist XPlist::split(cudaMatrixui split_condition_d,
		cudaMatrixui &old_ids0,cudaMatrixui &old_ids1,const int old_ids_out,int forceflag)
{
	/*
	 * This method splits returns a particle list containing particles with split_condition_d = 1
	 * This list becomes the particle list containing particles with split_condition_d = 0
	 *
	 */


	printf("Splitting Particle List\n");
	int* nptcls_0 = (int*)malloc(sizeof(int));
	int* nptcls_1 = (int*)malloc(sizeof(int));
	int nblocks = (nptcls_max+BLOCK_SIZE2-1)/BLOCK_SIZE2;

	nspecies = max(1,nspecies);
/*

	cudaMatrixr extra_data_parent;
	cudaMatrixr extra_data0_out;
	cudaMatrixr extra_data0_d;
	cudaMatrixr*extra_data1_d;
	cudaMatrixr* extra_data_parent_d;

	if(nextra > 0)
	{
		extra_data_parent = (cudaMatrixr*)malloc(nextra*sizeof(cudaMatrixr));
		extra_data0_out = (cudaMatrixr*)malloc(nextra*sizeof(cudaMatrixr));
		for(int i=0;i<nextra;i++)
		{
			extra_data_parent[i] = extra_data0[i];
		}

		cudaMalloc((void**)&extra_data0_d,nextra*sizeof(cudaMatrixr));
		cudaMalloc((void**)&extra_data1_d,nextra*sizeof(cudaMatrixr));
		cudaMalloc((void**)&extra_data_parent_d,nextra*sizeof(cudaMatrixr));

		CUDA_SAFE_CALL(cudaMemcpy(extra_data_parent_d,extra_data_parent,sizeof(cudaMatrixr)*nextra,cudaMemcpyHostToDevice));


	}
*/
	int* nptcls_0_max;
	int* nptcls_1_max;
	cudaMalloc((void**)&nptcls_0_max,nspecies*sizeof(int));
	cudaMalloc((void**)&nptcls_1_max,nspecies*sizeof(int));

	CUDA_SAFE_CALL(cudaMemset(nptcls_0_max,0,nspecies*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(nptcls_1_max,0,nspecies*sizeof(int)));

	int* nptcls_0_d;
	int* nptcls_1_d;
	cudaMalloc((void**)&nptcls_0_d,sizeof(int)*nspecies);
	cudaMalloc((void**)&nptcls_1_d,sizeof(int)*nspecies);
	CUDA_SAFE_CALL(cudaMemset(nptcls_0_d,0,nspecies*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(nptcls_1_d,0,nspecies*sizeof(int)));

	cudaMatrixui blockoffsets(nptcls_max,nspecies);
	cudaMatrixui gridoffsets(nblocks,nspecies);

	XPlist particles1;
	XPlist particles0;
	XPlist parent;

	XPlist* particles1_d;
	XPlist* particles0_d;
	XPlist* parent_d;

	CUDA_SAFE_CALL(cudaMalloc((void**)&particles1_d,sizeof(XPlist)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles0_d,sizeof(XPlist)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&parent_d,sizeof(XPlist)));


	dim3 cudaBlockSize(BLOCK_SIZE2,1,1);
	dim3 cudaGridSize(nblocks,nspecies,1);

	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	CUDA_SAFE_KERNEL((reduce_split_blockoffsets<<<cudaGridSize,cudaBlockSize>>>
													(gridoffsets,blockoffsets,split_condition_d,nptcls)));
	cudaDeviceSynchronize();

	cudaGridSize.x = 1;

	CUDA_SAFE_KERNEL((reduce_split_gridoffsets<<<cudaGridSize,cudaBlockSize>>>
			(gridoffsets,blockoffsets,nptcls_0_d,nptcls_1_d,nptcls,nptcls_0_max,nptcls_1_max,nblocks)));

	cudaDeviceSynchronize();

	CUDA_SAFE_KERNEL(cudaMemcpy(nptcls_0,nptcls_0_max,sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_KERNEL(cudaMemcpy(nptcls_1,nptcls_1_max,sizeof(int),cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	cudaGridSize.x = nblocks;

	if(forceflag == 2)
	{
		// Just set the max number of particles to the number of orbiting particles
		nptcls_max = nptcls_0[0];

		particles1.XPlist_init(1,nspecies,XPlistlocation_device);
		particles1.nptcls_max = 0;
	}
	else if((nptcls_1[0] >= BLOCK_SIZE/4)||(forceflag!=0))
	{
		printf("nptcls_0 = %i, nptcls_1 = %i \n",nptcls_0[0],nptcls_1[0]);
		if((nptcls_1[0] > 0))
		{
			if(nptcls_0[0] >0)
			{

				old_ids0.cudaMatrix_allocate(nptcls_0[0],nspecies,1);
				old_ids1.cudaMatrix_allocate(nptcls_1[0],nspecies,1);
				particles1.XPlist_init(nptcls_1[0],nspecies,XPlistlocation_device);
				particles0.XPlist_init(nptcls_0[0],nspecies,XPlistlocation_device);

				parent = *this;

				CUDA_SAFE_CALL(cudaMemcpy(particles1_d,&particles1,sizeof(XPlist),cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL(cudaMemcpy(particles0_d,&particles0,sizeof(XPlist),cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL(cudaMemcpy(parent_d,&parent,sizeof(XPlist),cudaMemcpyHostToDevice));
/*
				if(nextra > 0)
				{
					for(int i=0;i<nextra;i++)
					{
						extra_data1[i].cudaMatrix_allocate(nptcls_1[0],nspecies,1);
						extra_data0_out[i].cudaMatrix_allocate(nptcls_0[0],nspecies,1);
					}
					CUDA_SAFE_CALL(cudaMemcpy(extra_data0_d,extra_data0_out,sizeof(cudaMatrixr)*nextra,cudaMemcpyHostToDevice));
					CUDA_SAFE_CALL(cudaMemcpy(extra_data1_d,extra_data1,sizeof(cudaMatrixr)*nextra,cudaMemcpyHostToDevice));
				}
*/

				CUDA_SAFE_CALL(cudaMemcpy(particles0.nptcls,nptcls_0_d,nspecies*sizeof(int),cudaMemcpyDeviceToDevice));
				CUDA_SAFE_CALL(cudaMemcpy(particles1.nptcls,nptcls_1_d,nspecies*sizeof(int),cudaMemcpyDeviceToDevice));
				CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
				CUDA_SAFE_KERNEL((split_list_kernel<<<cudaGridSize,cudaBlockSize>>>
							(parent_d,particles0_d,particles1_d,old_ids0,old_ids1,
							gridoffsets,blockoffsets,split_condition_d,old_ids_out)));

				cudaDeviceSynchronize();

				*this = particles0;

				parent.XPlistFree();
/*
				if(nextra > 0)
				{
					for(int i=0;i<nextra;i++)
					{
						extra_data_parent[i].cudaMatrixFree();

						extra_data0[i] = extra_data0_out[i];
					}
				}
				*/
			}
			else
			{
				particles0.XPlist_init(1,nspecies,XPlistlocation_device);
				particles0.nptcls_max = 0;
				old_ids0.cudaMatrix_allocate(1,1,1);
				old_ids1.cudaMatrix_allocate(1,1,1);
				particles1 = *this;
				*this = particles0;



				/*
				if(nextra > 0)
				{
					for(int i=0;i<nextra;i++)
					{
						extra_data1[i] = extra_data0[i];
					}
				}
				*/
			}

		}
		else
		{

			particles1.XPlist_init(1,nspecies,XPlistlocation_device);
			particles1.nptcls_max = 0;
			old_ids1.cudaMatrix_allocate(1,1,1);
			old_ids0.cudaMatrix_allocate(1,1,1);
		}
	}
	else
	{
		particles1.XPlist_init(1,nspecies,XPlistlocation_device);
		particles1.nptcls_max = 0;
	}

	if(old_ids_out == 0)
	{
		old_ids0.cudaMatrix_allocate(1,1,1);
		old_ids1.cudaMatrix_allocate(1,1,1);
	}

	blockoffsets.cudaMatrixFree();
	gridoffsets.cudaMatrixFree();
/*
	if(nextra > 0)
	{
		cudaFree(extra_data0_d);
		cudaFree(extra_data1_d);
		cudaFree(extra_data_parent_d);

	}
*/

	CUDA_SAFE_CALL(cudaFree(nptcls_0_max));
	CUDA_SAFE_CALL(cudaFree(nptcls_1_max));
	CUDA_SAFE_CALL(cudaFree(nptcls_0_d));
	CUDA_SAFE_CALL(cudaFree(nptcls_1_d));
	CUDA_SAFE_CALL(cudaFree(particles1_d));
	CUDA_SAFE_CALL(cudaFree(particles0_d));
	CUDA_SAFE_CALL(cudaFree(parent_d));

	return particles1;


}


__global__
void XPlist_append_kernel(XPlist main_list,XPlist second_list,
											  int* nptcls,int nptcls_allocated)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int new_idx;

	__shared__ int nptcls_species;
	__shared__ int new_start;


	if(idx == 0)
	{
		nptcls_species = second_list.nptcls[idBeam];
		new_start = nptcls[idBeam];

		main_list.nptcls[idBeam] = nptcls_species+new_start;
	}
	__syncthreads();

	if(gidx < nptcls_species)
	{
		new_idx = gidx+new_start;

		if(new_idx < nptcls_allocated)
		{
			second_list.pop(&main_list,new_idx);
		}
		else
		{
		//	printf("!!!! Error New ID is too big to fit into the list!!!!\n");
		}
	}



}

__host__
void XPlist::append(XPlist listin)
{
	int nptcls_max_in = listin.nptcls_max;
	int nspecies_in = listin.nspecies;
	int nptcls_allocated_out;
	int* nptcls_main = (int*)malloc(nspecies*sizeof(int));
	int* nptcls_in = (int*)malloc(nspecies*sizeof(int));
	int nptcls_max_out=0;

	CUDA_SAFE_CALL(cudaMemcpy(nptcls_main,nptcls,nspecies*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(nptcls_in,listin.nptcls,nspecies*sizeof(int),cudaMemcpyDeviceToHost));

	for(int i=0;i<nspecies_in;i++)
	{
		nptcls_max_out = max(nptcls_max_out,(nptcls_main[i]+nptcls_in[i]));
		//printf("nptcls_main = %i, nptcls_in = %i\n",nptcls_main[i],nptcls_in[i]);
	}
	nptcls_max_out = min(nptcls_max_out,(nptcls_max_in+nptcls_max));

	XPlist temp_list;
	XPlist temp_list2 = *this;

	dim3 cudaBlockSize(BLOCK_SIZE2,1,1);
	dim3 cudaGridSize(1,nspecies_in,1);
	cudaGridSize.x = (nptcls_max_in+cudaBlockSize.x-1)/cudaBlockSize.x;




	// Check to make sure that the list has enough space to accomadate the new list
	if(nptcls_allocated >= nptcls_max_out)
	{
		nptcls_allocated_out = nptcls_allocated;
		temp_list = *this;
	}
	else
	{
		// Need to expand the list
		nptcls_allocated_out = nptcls_max_in+nptcls_max;
		temp_list.XPlist_init(nptcls_allocated_out,nspecies,XPlistlocation_device);
		XPlistCopy(temp_list,*this,nptcls_max,nspecies,cudaMemcpyDeviceToDevice);
		CUDA_SAFE_CALL(cudaMemcpy(temp_list.nptcls,nptcls,nspecies*sizeof(int),cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(temp_list.species,species,nspecies*sizeof(int),cudaMemcpyDeviceToDevice));
	}


	CUDA_SAFE_KERNEL((XPlist_append_kernel<<<cudaGridSize,cudaBlockSize>>>
									(temp_list,listin,nptcls,nptcls_allocated_out)));

	if(nptcls_allocated < nptcls_allocated_out)
	{
		// Need to expand the list
		temp_list2.XPlistFree();

		*this = temp_list;
	}


	listin.XPlistFree();


	return;

}


__host__
realkind box_muller(realkind width, realkind offset)
{
	realkind result;
	realkind pi = 3.14159265358979323846264338327950288419716939937510f;
	realkind u1 = ((realkind)(rand() % 100000)+1.0f)/100000.0f;
	realkind u2 = ((realkind)(rand() % 100000)+1.0f)/100000.0f;

	result = width/4.0*sqrt(-2*log(u1))*cos(2*pi*u2)+offset;

	//printf(" result = %f \n",result);

	return result;
}

__device__
realkind randGauss(curandState* state,realkind mean,realkind std)
{
	return curand_normal(state)*2.0*std+mean;
}


__global__
void curand_init_kernel(curandState* random_states, size_t pitch,int random_state_offset,int nptcls_max)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.y;
	unsigned int idBeam = blockIdx.y;
	unsigned int element_id = gidx + gridDim.x*blockDim.x*idBeam;

	curandState* my_state;

	if(gidx < nptcls_max)
	{
		my_state = (curandState*)((char*)(random_states)+pitch*idBeam)+gidx;

		curand_init(RANDOM_SEED,element_id,0,my_state);
	}


}


template<class O>
__host__
void XPlist::XPlist_allocate(O op)
{
	dim3 cudaBlockSize(512,1,1);
	dim3 cudaGridSize(1,nspecies,1);

	cudaGridSize.x = (nptcls_max+cudaBlockSize.x-1)/cudaBlockSize.x;

	int intsize = sizeof(int)*nptcls_max;
	int realkindsize = sizeof(double)*nptcls_max;

	nptcls_allocated = nptcls_max;


	realkind** fltptr;
	int** intptr;

	for(int i=0;i<nrealkinds;i++)
	{
		fltptr = get_realkind_pointer(i);
		op((void**)fltptr,&nbi_fpitch,realkindsize,nspecies);
	}

	for(int i=0;i<nints;i++)
	{
		intptr = get_int_pointer(i);
		op((void**)intptr,&nbi_ipitch,intsize,nspecies);
	}

	op((void**)&random_state,&nbi_curpitch,sizeof(curandState)*nptcls_max,nspecies);
	op((void**)&pexit,&nbi_epitch,sizeof(enum XPlistexits)*nptcls_max,nspecies);
	CUDA_SAFE_CALL(cudaMalloc((void**)&nptcls,sizeof(int)*nspecies));
	CUDA_SAFE_CALL(cudaMalloc((void**)&species,sizeof(int)*nspecies));
	CUDA_SAFE_CALL(cudaMemset(nptcls,0,sizeof(int)*nspecies));
/*
	CUDA_SAFE_KERNEL((curand_init_kernel<<<cudaGridSize,cudaBlockSize>>>(
										random_state,nbi_curpitch,random_state_counter,nptcls_max)));

	random_state_counter += nptcls_max*nspecies;
	printf("random_state_counter = %i \n",random_state_counter);
*/

	return;

}


__global__
void find_cell_index_kernel(XPlist particles,
						int* cellindex_out,size_t ipitch,int cell_max)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;
	unsigned int gid = gidx+ipitch*idBeam/sizeof(int);

	if(gidx < particles.nptcls[idBeam])
	{
		cellindex_out[gid] = particles.cellindex[0][gidx+particles.nbi_ipitch*idBeam/sizeof(int)];
		//printf("particle %i, at (%f,%f,%f) index (%i,%i,%i) cellindex %i \n",gidx,x[gidx],y[gidx],z[gidx],nx[gidx],ny[gidx],nz[gidx],cellindex[gidx]);

	}
	else if(gidx >= particles.nptcls[idBeam])
	{
		if(gidx < particles.nptcls_max)
			cellindex_out[gid] = cell_max*2;
	}
	return;

}

__host__
void XPlist::find_cell_index(int* cellindex_temp,size_t ipitch,int cell_max)
{
	if(!location)
	{
		printf("Error, XPlist::find_index() can only be called for a particle list residing on the device. \n");
		return;
	}
		dim3 cudaBlockSize(BLOCK_SIZE,1,1);
		dim3 cudaGridSize((nptcls_max+BLOCK_SIZE-1)/BLOCK_SIZE,max(1,nspecies),1);
		CUDA_SAFE_KERNEL((find_cell_index_kernel<<<cudaGridSize,cudaBlockSize>>>(*this,cellindex_temp,ipitch,cell_max)));
}

__global__
void write_xpindex_array(unsigned int* index_array,size_t ipitch,int nptcls)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	if(gidx < nptcls)
	{
		*((unsigned int*)((char*)index_array+ipitch*idBeam)+gidx) = gidx;
	}
}

__global__
void sort_remaining(XPlist particles_global, XPlist particles_global_temp, unsigned int* index_array,size_t ipitch)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	__shared__ XPlist particles;
	__shared__ XPlist particles_temp;
	__shared__ int nptcls_max;
	int ogidx;

	if(idx == 0)
	{
		particles = particles_global;
		particles_temp = particles_global_temp;
		nptcls_max = particles_global.nptcls_max;
		if(gidx == 0)
			particles_global_temp.nptcls[idBeam] = particles_global.nptcls[idBeam];
	}
	__syncthreads();

	if(gidx < particles_global.nptcls[idBeam])
	{

		 ogidx = *((unsigned int*)((char*)index_array+ipitch*idBeam)+gidx); // What particle is this thread relocating


	}
	unsigned int iidxin = ogidx+(idBeam*(particles.nbi_ipitch))/sizeof(int);
	unsigned int fidxin = ogidx+(idBeam*(particles.nbi_fpitch))/sizeof(realkind);
	unsigned int enidxin = ogidx+(idBeam*(particles.nbi_epitch))/sizeof(enum XPlistexits);
	unsigned int cuidxin = ogidx+(idBeam*(particles.nbi_curpitch))/sizeof(curandState);

	unsigned int iidxout = gidx+(idBeam*(particles_temp.nbi_ipitch))/sizeof(int);
	unsigned int fidxout = gidx+(idBeam*(particles_temp.nbi_fpitch))/sizeof(realkind);
	unsigned int enidxout = gidx+(idBeam*(particles_temp.nbi_epitch))/sizeof(enum XPlistexits);
	unsigned int cuidxout = gidx+(idBeam*(particles_temp.nbi_curpitch))/sizeof(curandState);

	int* intptr_in;
	int* intptr_out;
	realkind* fltptr_in;
	realkind* fltptr_out;

	if(gidx < particles.nptcls[idBeam])
	{

		for(int i=0;i<particles.nrealkinds;i++)
		{
			fltptr_in = *(particles.get_realkind_pointer(i));
			fltptr_out = *(particles_temp.get_realkind_pointer(i));
			fltptr_out[fidxout] = fltptr_in[fidxin];
		}

		for(int i=0;i<particles.nints;i++)
		{
			intptr_in = *(particles.get_int_pointer(i));
			intptr_out = *(particles_temp.get_int_pointer(i));
			intptr_out[iidxout] = intptr_in[iidxin];
		}

		particles_temp.pexit[enidxout] = particles.pexit[enidxin];
		particles_temp.random_state[cuidxout] = particles.random_state[cuidxin];
	}


}


int compare(const void* a, const void* b)
{
	return (((int2*)a)->x < ((int2*)b)->x);
}

void stupid_sort(int* keys_d, int* values_d, int nelements)
{

	int* keys_h = (int*)malloc(nelements*sizeof(int));
	int* values_h = (int*)malloc(nelements*sizeof(int));

	int2* dict = (int2*)malloc(nelements*sizeof(int2));

	CUDA_SAFE_CALL(cudaMemcpy(keys_h,keys_d,nelements*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(values_h,values_d,nelements*sizeof(int),cudaMemcpyDeviceToHost));

	for(int i=0;i<nelements;i++)
	{
		dict[i].x = keys_h[i];
		dict[i].y = values_h[i];
	}

	qsort(dict,nelements,sizeof(int2),compare);

	for(int i=0;i<nelements;i++)
	{
		keys_h[i] = dict[i].x;
		values_h[i] = dict[i].y;
	}

	CUDA_SAFE_CALL(cudaMemcpy(keys_d,keys_h,nelements*sizeof(int),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(values_d,values_h,nelements*sizeof(int),cudaMemcpyHostToDevice));

	free(keys_h);
	free(values_h);
	free(dict);
}

__host__
void XPlist::sort(realkind2 Pgridspacing, int2 Pgrid_i_dims)
{
	/* TODO
	 * 1) figure out the cell index of each particle
	 * 2) write a particle index array for values array
	 * 3) create a radix object
	 * 4) use radix.sort(d_keys,d_values) keys = cell index, values = particle index
	 * 5) launch a kernel to move the particles to their new array index based on the sorted key/value pairs
	 *
	 */

	cudaError status;
	dim3 cudaBlockSize(BLOCK_SIZE,1,1);
	dim3 cudaGridSize((nptcls_max+BLOCK_SIZE)/BLOCK_SIZE,nspecies,1);
	size_t ipitch;

	if(!location)
	{
		printf("Error, XPlist::sort() can only be called for a particle list residing on the device. \n");
		return;
	}

	int nptcls_h[nspecies];
	int cell_max = Pgrid_i_dims.x*Pgrid_i_dims.y;

	CUDA_SAFE_KERNEL(cudaMemcpy(nptcls_h,nptcls,nspecies*sizeof(int),cudaMemcpyDeviceToHost));

	// Setup temporary particle index array
	unsigned int* XP_index_array;
	CUDA_SAFE_KERNEL(cudaMallocPitch((void**)&XP_index_array,&ipitch,nptcls_max*sizeof(int),nspecies));

	int* cellindex_temp;
	CUDA_SAFE_KERNEL(cudaMallocPitch((void**)&cellindex_temp,&ipitch,nptcls_max*sizeof(int),nspecies));

	unsigned int* d_keys = (unsigned int*)cellindex_temp;
	unsigned int* d_values = XP_index_array;

	cudaThreadSynchronize();

	// Populate the cellindex array so that it can be used to sort the particles
	find_cell_index(cellindex_temp,ipitch,cell_max);

	// Write a particle index array for values array
	CUDA_SAFE_KERNEL((write_xpindex_array<<<cudaGridSize,cudaBlockSize>>>(XP_index_array,ipitch,nptcls_max)));
	cudaDeviceSynchronize();

#ifndef USE_STUPID_SORT
	d_keys = (unsigned int*)((char*)cellindex_temp);
	d_values = (unsigned int*)((char*)XP_index_array);

	thrust::device_ptr<unsigned int> thrust_keys(d_keys);
	thrust::device_ptr<unsigned int> thrust_values(d_values);


	// Create the RadixSort object
	//nvRadixSort::RadixSort radixsort(nptcls_max+1);
	for(int i=0;i<nspecies;i++)
	{
		printf("nptcls_h[%i] = %i\n",i,nptcls_h[i]);


		// Sort the key / value pairs
		if(nptcls_h[i] > 0)
		{
			thrust::sort_by_key( thrust_keys+i*ipitch/sizeof(int), thrust_keys+i*ipitch/sizeof(int)+nptcls_h[i], thrust_values);
			//CUDA_SAFE_KERNEL((radixsort.sort(d_keys,d_values,nptcls_h[i],32)));
		}
	}
#else
	d_keys = (uint*)((char*)cellindex_temp);
	d_values = (uint*)((char*)XP_index_array);


	// Create the RadixSort object
	//nvRadixSort::RadixSort radixsort(nptcls_max+1);
	for(int i=0;i<nspecies;i++)
	{
		printf("nptcls_h[%i] = %i\n",i,nptcls_h[i]);


		// Sort the key / value pairs
		if(nptcls_h[i] > 0)
		{
			stupid_sort((int*)d_keys+i*ipitch/sizeof(int),(int*)d_values+i*ipitch/sizeof(int),nptcls_h[i]);
			//CUDA_SAFE_KERNEL((radixsort.sort(d_keys,d_values,nptcls_h[i],32)));
		}
	}
#endif

	XPlist temp_list(nptcls_max,nspecies,XPlistlocation_device);


	// sort the rest of the particle data
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	CUDA_SAFE_KERNEL((sort_remaining<<<cudaGridSize,cudaBlockSize>>>(*this,temp_list,XP_index_array,ipitch)));
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "sort_remaining %s\n", cudaGetErrorString(status));}

	XPlistCopy(*this,temp_list,nptcls_max,nspecies,cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();

	cudaFree(XP_index_array);
	cudaFree(cellindex_temp);
	temp_list.XPlistFree();
	cudaThreadSynchronize();
	return;

}

/*
__global__ void count_particles(XPlist particles,cudaMatrixi2 cellInfo,int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam= blockIdx.y;
	int cellindex;
	int nptcls = particles.nptcls[idBeam];

	if(nptcls == 0)
		return;

	__shared__ int XPcellindex[BLOCK_SIZE+101];

	if(gidx < particles.nptcls[idBeam]-1)
	{
		XPcellindex[idx] = particles.cellindex[0][gidx+idBeam*particles.nbi_ipitch/sizeof(int)];
		if((idx==BLOCK_SIZE-1)||(gidx == nptcls-2))
		{
			XPcellindex[idx+1] = particles.cellindex[0][gidx+idBeam*particles.nbi_ipitch/sizeof(int)+1];
		}
	}
	__syncthreads();
	if(gidx < particles.nptcls[idBeam]-1)
	{
		if(XPcellindex[idx] != XPcellindex[idx+1])
		{
			cellindex = XPcellindex[idx]+1;

			// Set index of first particle in next cell
			cellInfo(cellindex,idBeam).x = gidx+1;
			//printf("(GPU) %i particles before cell %i \n",gidx+1,XPcellindex[gidx]+1);
		}
	}
	else if(gidx == particles.nptcls[idBeam]-1)
	{
		cellindex = XPcellindex[idx]+1;
		cellInfo(0,idBeam).x = 0;
		cellInfo(cellindex,idBeam).x = gidx+1;
		cellInfo(ncells,idBeam).x = particles.nptcls[idBeam];
	}
}


__global__ void fix_cellinfo(cudaMatrixi2 cellinfo,int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	if(gidx < ncells)
	{
		int tempi0;
		int tempi1;
		int j=1;

		if(cellinfo(gidx,idBeam).x != 0)
		{
			while((cellinfo(gidx+j,idBeam).x == 0)&&((gidx+j) < ncells))
			{
				cellinfo(gidx+j,idBeam).x = cellinfo(gidx,idBeam).x+1;
				j++;
				__threadfence();
			}
		}

	}

}

__global__ void count_blocks(cudaMatrixi2 cellinfo,int* nblocksperspecies,int* blocksize,int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	__shared__ int tempCellInfo[BLOCK_SIZE+1];

	volatile int* s_ptr = tempCellInfo;

	__syncthreads();

	realkind temp;
	int tempi;

	tempCellInfo[idx] = 0;
	__syncthreads();
	if(gidx < ncells)
	{


		temp = ((realkind)(cellinfo(gidx+1,idBeam).x-cellinfo(gidx,idBeam).x))/((realkind)BLOCK_SIZE);

		tempCellInfo[idx] = ceil(temp);

		cellinfo(gidx,idBeam).y = tempCellInfo[idx];
	}
		__syncthreads();

		if (idx < 128) { tempCellInfo[idx] += tempCellInfo[idx + 128]; } __syncthreads();
		if (idx <  64) { tempCellInfo[idx] += tempCellInfo[idx +  64]; } __syncthreads();
		if (idx < 32)
		{
			s_ptr[idx] += s_ptr[idx + 32];
			s_ptr[idx] += s_ptr[idx + 16];
			s_ptr[idx] += s_ptr[idx +  8];
			s_ptr[idx] += s_ptr[idx +  4];
			s_ptr[idx] += s_ptr[idx +  2];
			s_ptr[idx] += s_ptr[idx +  1];
		}
		__syncthreads();
		if (idx == 0)
		{
			tempCellInfo[0] += atomicAdd(&nblocksperspecies[idBeam],tempCellInfo[0]);
			atomicMax(blocksize,tempCellInfo[0]);
		}


}

__global__
void populate_blockinfo(cudaMatrixi3 blockinfo,cudaMatrixi2 cellinfo,
											int nspecies, int ncells,int* nblocksperspecies)
{
	int k = 0;
	int firstParticle;
	unsigned int idBeam = threadIdx.x+blockIdx.x*blockDim.x;
	int nblocks = nblocksperspecies[idBeam];

	k=0;
	for(int i = 0;i<ncells;i++)
	{
		firstParticle = cellinfo(i,idBeam).x;

		for(int j = 0;j<cellinfo(i,idBeam).y;j++)
		{
			if(k<=nblocks)
			{
				blockinfo(k,idBeam).x = i;
				blockinfo(k,idBeam).y = firstParticle;
				if(j < cellinfo(i,idBeam).y-1)
				{
						blockinfo(k,idBeam).z = BLOCK_SIZE;
						firstParticle += BLOCK_SIZE;
				}
				else
				{
					blockinfo(k,idBeam).z = cellinfo(i+1,idBeam).x-firstParticle;
				}
			}
			k++;
		}
	}


}
*/

__device__
void XPlist::calc_binid(Environment* plasma_in,int icall,int idx)
{
	int nr = plasma_in -> griddims.x;
	int nz = plasma_in -> griddims.y;
	int ix;
	int iy;

	nx[icall][idx] = rint((px[icall][idx]-plasma_in->Rmin)/plasma_in->gridspacing.x);
	ny[icall][idx] = rint((py[icall][idx]-plasma_in->Zmin)/plasma_in->gridspacing.y);

	ix = max(0,min(nr-1,nx[icall][idx]));
	iy = max(0,min(nz-1,ny[icall][idx]));
	if(orbflag[idx] == 1)
	{
		cellindex[icall][idx] = zmap(ix,iy);
		//cellindex[icall][idx] = eval_NGC(plasma_in,idx,icall);
	}
	else
	{
		cellindex[icall][idx] = zmap(nr,nz);
	}


}

__device__
int XPlist::check_orbit(Environment* plasma_in,int icall)
{
	unsigned int idx = threadIdx.x;
	realkind Xi;
	int transp_zone;
	int ilim = 0;
	realkind limiter_distance;

//	if((orbflag[idx] == 0))
//	{
//		return 1;
//	}

	// Check GC;
	Xi = plasma_in -> Xi_map(px[0][idx],py[0][idx]);
	transp_zone = (int)rint(plasma_in -> transp_zone(px[0][idx],py[0][idx]));
	limiter_distance = plasma_in -> limiter_map(px[0][idx],py[0][idx]);

	ilim = 0;
	if(Xi > plasma_in -> xi_max){ilim+=1;}

	if(transp_zone < 0){ilim += 2;}


	if(-1.0f*limiter_distance <= rlarmor[idx]){ilim += 3;}

	if(px[0][idx] >= (plasma_in -> Rmax))
	{	ilim += 1;}
	if(py[0][idx] >= (plasma_in -> Zmax))
	{	ilim += 1;}

	if((px[0][idx] <= (plasma_in->Rmin))||(py[0][idx] <= (plasma_in->Zmin)))
	{	ilim += 1;}

/*
	if(icall == 1)
	{
		printf("Check Particle(%i,%i) = %i\nposition = %f,%f\nXi=%f\ntranps_zone=%i\nlimiter_distance=%f\n",
				idx,blockIdx.x,ilim,px[0][idx],py[0][idx],Xi,transp_zone,limiter_distance);
	}
*/

	return ilim;


}


__device__
int XPlist::check_outside(Environment* plasma)
{
	unsigned int idx = threadIdx.x;
	int transp_index;
	int beam_index;
	int result = 1;


	transp_index = plasma->transp_zone(px[1][idx],py[1][idx]);
	beam_index = plasma->beam_zone(px[1][idx],py[1][idx]);

	if((transp_index>=(plasma->ledge_transp))||(beam_index > plasma->nbeam_zones_inside))
	{
		result = 0;
	}

	return result;


}

__device__
int XPlist::check_midplane_cx(Environment* plasma_in,XPlist* particles_old)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;
	realkind ympl_old;
	realkind ympl;
	realkind old_x;
	realkind old_y;
	int old_fidx;
	int result;
	if(pexit[idx] == XPlistexit_newparticle)
	{
		return 0;
	}
	else
	{
		old_fidx = old_idx[idx]+(idBeam*(particles_old -> nbi_fpitch))/sizeof(realkind);
		old_x = particles_old -> px[0][old_fidx];
		old_y = particles_old -> py[0][old_fidx];
		ympl_old = old_y - (plasma_in -> Ymidplane(old_x,old_y));
	}

	ympl = py[0][idx] - (plasma_in -> Ymidplane(px[0][idx],py[0][idx]));

	if(ympl*ympl_old >= 0)
		result = 0;
	if(ympl*ympl_old < 0)
		result = 1;

	return result;



}

__device__
realkind XPlist::get_random(void)
{
	unsigned int idx = threadIdx.x;
	return curand_uniform(&(random_state[idx]));
}

__device__
realkind3 XPlist::Bvector(realkind r,realkind dPsidR,realkind dPsidZ,realkind g,int ilocation)
{
	realkind3 result;
	result.x = bphi_sign*dPsidZ/r;
	result.y = bphi_sign*(-1.0f)*dPsidR/r;
	result.z = g/(pow(r,2));

	return result;
}

__device__
realkind3 XPlist::eval_Bvector(XPTextureSpline Psispline,XPTextureSpline gspline,int ilocation)
{
	int idx = threadIdx.x;

	realkind dPsidR = Psispline.BCspline_eval<XPgridderiv_dfdx>(px[ilocation][idx],py[ilocation][idx]);
	realkind dPsidZ = Psispline.BCspline_eval<XPgridderiv_dfdy>(px[ilocation][idx],py[ilocation][idx]);
	realkind g = gspline.BCspline_eval<XPgridderiv_f>(px[ilocation][idx],py[ilocation][idx]);

	realkind3 result;

	result = Bvector(px[ilocation][idx],dPsidR,dPsidZ,g,ilocation);

	return result;


}

__device__
realkind XPlist::eval_Bmod(XPTextureSpline Psispline,XPTextureSpline gspline)
{
	int idx = threadIdx.x;

	realkind dPsidR = Psispline.BCspline_eval<XPgridderiv_dfdx>(px[0][idx],py[0][idx]);
	realkind dPsidZ = Psispline.BCspline_eval<XPgridderiv_dfdy>(px[0][idx],py[0][idx]);
	realkind g = gspline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]);

	return sqrt(dPsidR*dPsidR+dPsidZ*dPsidZ+g*g)/px[0][idx];
}

__device__
realkind XPlist::eval_mu(XPTextureSpline Psispline,XPTextureSpline gspline)
{
	unsigned int idx = threadIdx.x;

	realkind result = 0;
	realkind B;

	if(idx < nptcls_max)
	{
		B = eval_Bmod(Psispline,gspline);
		result = 0.5*mass[idx]*ZMP*pow(vperp[0][idx],2)/B;
	}

	return result;

}
__device__
realkind XPlist::eval_dt(Environment* plasma_in)
{
	unsigned int idx = threadIdx.x;
	realkind result;
	realkind Xi = plasma_in->Xi_map(px[0][idx],py[0][idx]);
	realkind time_left = max((plasma_in->delt - time_done[idx]),(plasma_in->delt)/10.0 );
	realkind steps_left = MAX_STEPS-plasma_in->istep+1;
	realkind avg_dt = max(deltat[idx],time_done[idx]/(plasma_in->istep+1));

	realkind accelfactor;




	accelfactor = abs((time_left/(avg_dt))/steps_left);
	accelfactor = min(accelfactor,2.0);

	//result = dl_limit/dldt;

	result = deltat[idx];

	steps_left = MAX_STEPS;
	result = min(result,5.0*plasma_in->delt /steps_left);

	if (Xi > 0.9)
	{
		if (Xi >= 1)
		{
			result = 0.25*result;
		}
		else
		{
			result = result*(1-0.75*(10.0*(Xi-0.9)));
		}
	}

	//result = max(result*accelfactor,result);


	//result = max(result,2.0f*orbit_dt_min);

	return result;
}

__device__
int XPlist::eval_NGC(Environment* plasma_in,int idx,int location)
{
	return plasma_in->transp_zone(px[location][idx],py[location][idx]);
}

__device__
realkind3 XPlist::eval_vperp_vector(realkind Br, realkind Bz, realkind Bphi,int ilocation)
{
	// ilocation = 0 for guiding center, 1 for finite larmor radius

	unsigned int idx = threadIdx.x;

	realkind rotation_matrix[3][3];
	realkind cospho = cos(phase_angle[idx]);
	realkind sinpho = sin(phase_angle[idx]);

	realkind tresult[3] = {0,0,0};
	realkind vperp_0[3]; // vperp for theta=0;
	realkind modB = sqrt(Br*Br+Bz*Bz+Bphi*Bphi);

	realkind3 result;

	// Normalize the components of the magnetic field

	Br /= modB;
	Bz /= modB;
	Bphi /= modB;

	// Setup vperp_0

	// In the R direction
	vperp_0[0] = -Br*Bz/sqrt(Br*Br+Bphi*Bphi);
	// In the Z direction
	vperp_0[1] = (Br*Br+Bphi*Bphi)/sqrt(Br*Br+Bphi*Bphi);
	// In the toroidal direction
	vperp_0[2] = -Bphi*Bz/sqrt(Br*Br+Bphi*Bphi);

	// Setup the rotation matrix
	rotation_matrix[0][0] = cospho+Br*Br*(1-cospho);
	rotation_matrix[1][0] = Br*Bz*(1-cospho)-Bphi*sinpho;
	rotation_matrix[2][0] = Br*Bphi*(1-cospho)+Bphi*sinpho;
	rotation_matrix[0][1] = Br*Bz*(1-cospho)+Bphi*sinpho;
	rotation_matrix[1][1] = cospho+Bz*Bz*(1-cospho);
	rotation_matrix[2][1] = Bz*Bphi*(1-cospho)-Br*sinpho;
	rotation_matrix[0][2] = Bphi*Br*(1-cospho)-Bz*sinpho;
	rotation_matrix[1][2] = Bphi*Bz*(1-cospho)+Br*sinpho;
	rotation_matrix[2][2] = cospho+Bphi*Bphi*(1-cospho);

	// Rotate the vperp_0 vector
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			tresult[i] += rotation_matrix[j][i]*vperp_0[j];
		}
	}

	result.x = tresult[0]*vperp[ilocation][idx];
	result.y = tresult[1]*vperp[ilocation][idx];
	result.z = tresult[2]*vperp[ilocation][idx];

	return result;

}

__device__
realkind3 XPlist::eval_velocity_vector(realkind Br, realkind Bz, realkind Bphi,int ilocation)
{
	unsigned int idx = threadIdx.x;

	realkind3 vperp_vector = eval_vperp_vector(Br,Bz,Bphi,ilocation);
	realkind3 result;
	realkind modB = sqrt(Br*Br+Bz*Bz+Bphi*Bphi);

	result.x = vpara[ilocation][idx]*Br/modB+vperp_vector.x;
	result.y = vpara[ilocation][idx]*Bz/modB+vperp_vector.y;
	result.z = vpara[ilocation][idx]*Bphi/modB+vperp_vector.z;

	return result;

}

__device__
realkind3 XPlist::eval_larmor_vector(realkind Br, realkind Bz, realkind Bphi)
{
	// ilocation = 0 for guiding center, 1 for finite larmor radius

	unsigned int idx = threadIdx.x;

	realkind rotation_matrix[3][3];
	realkind cospho = cos(phase_angle[idx]);
	realkind sinpho = sin(phase_angle[idx]);

	realkind tresult[3] = {0,0,0};
	realkind r_0[3]; // r vector for theta=0;
	realkind modB = sqrt(Br*Br+Bz*Bz+Bphi*Bphi);

	realkind3 result;

	// Normalize the components of the magnetic field

	Br /= modB;
	Bz /= modB;
	Bphi /= modB;

	// Setup vperp_0

	// In the R direction
	r_0[0] = -Bphi/sqrt(Br*Br+Bphi*Bphi);
	// In the Z direction
	r_0[1] = 0;
	// In the toroidal direction
	r_0[2] = Br/sqrt(Br*Br+Bphi*Bphi);

	// Setup the rotation matrix
	rotation_matrix[0][0] = cospho+Br*Br*(1-cospho);
	rotation_matrix[1][0] = Br*Bz*(1-cospho)-Bphi*sinpho;
	rotation_matrix[2][0] = Br*Bphi*(1-cospho)+Bphi*sinpho;
	rotation_matrix[0][1] = Br*Bz*(1-cospho)+Bphi*sinpho;
	rotation_matrix[1][1] = cospho+Bz*Bz*(1-cospho);
	rotation_matrix[2][1] = Bz*Bphi*(1-cospho)-Br*sinpho;
	rotation_matrix[0][2] = Bphi*Br*(1-cospho)-Bz*sinpho;
	rotation_matrix[1][2] = Bphi*Bz*(1-cospho)+Br*sinpho;
	rotation_matrix[2][2] = cospho+Bphi*Bphi*(1-cospho);

	// Rotate the vperp_0 vector
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			tresult[i] += rotation_matrix[j][i]*r_0[j];
		}
	}

	result.x = tresult[0]*rlarmor[idx];
	result.y = tresult[1]*rlarmor[idx];
	result.z = tresult[2]*rlarmor[idx];

	return result;



}

__device__
void XPlist::update_gc(Environment* plasma_in)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.x;

	realkind bmod;
	realkind Psi = (plasma_in->Psispline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]));
	realkind g = (plasma_in->gspline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]));

	nx[0][idx] = (px[0][idx]-(plasma_in -> Rmin))/(plasma_in -> gridspacing.x);
	ny[0][idx] = (py[0][idx]-(plasma_in -> Zmin))/(plasma_in -> gridspacing.y);

	potential[0][idx] = ZEL*charge[idx]*(plasma_in->Phispline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]));


	bmod = eval_Bmod(plasma_in->Psispline,plasma_in->gspline);

	mu[idx] = 0.5*mass[idx]*ZMP*pow(vperp[0][idx],2)/(bmod);

	energy[idx] = 0.5*ZMP*mass[idx]*(pow(vpara[0][idx],2))+mu[idx]*(bmod)+potential[0][idx];

	rlarmor[idx] = vperp[0][idx]*mass[idx]/(9.578e7*charge[idx]*bmod*1.0e-4);

	momentum[idx] = mass[idx]*ZMP*vpara[0][idx]*g/(bmod)-bphi_sign*ZEL*charge[idx]*Psi/ZC;



}

__device__
void XPlist::update_flr(Environment* plasma)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;
	realkind velocity;
	realkind3 B_vector;
	realkind3 v_vector;
	realkind v_mag;
	realkind PsiGC;
	realkind PsiFLR;
	realkind bmodFLR;

	int iflr = 1; // 0 means use gc instead of flr

	realkind zdrx,zdrth,ztop;

	if(iflr == 1)
	{
	B_vector.x = plasma->Bfieldr(px[0][idx],py[0][idx])*ZT2G;
	B_vector.y = plasma->Bfieldz(px[0][idx],py[0][idx])*ZT2G;
	B_vector.z = plasma->Bfieldphi(px[0][idx],py[0][idx])*ZT2G;

	zdrx = rlarmor[idx]*cos(phase_angle[idx]);
	zdrth = rlarmor[idx]*sin(phase_angle[idx])*plasma->phi_ccw*B_vector.z;

	if(py[0][idx] >= 0.0)
		ztop = 1.0;
	else
		ztop = -1.0;

	px[1][idx] = px[0][idx];
	px[1][idx] += zdrx*B_vector.y;
	px[1][idx] += zdrth*ztop*B_vector.x;

	py[1][idx] = py[0][idx];
	py[1][idx] += zdrth*B_vector.y;
	py[1][idx] -= zdrx*ztop*B_vector.x;




	//realkind3 larmor_vector = eval_larmor_vector(plasma->Bfieldr(px[0][idx],py[0][idx])*ZT2G,
	//												   plasma->Bfieldz(px[0][idx],py[0][idx])*ZT2G,
	//												   plasma->Bfieldphi(px[0][idx],py[0][idx])*ZT2G);

	// Set the finite larmor radius position
	//px[1][idx] = sqrt(pow(px[0][idx]+larmor_vector.x,2)+larmor_vector.z*larmor_vector.z);
	//py[1][idx] = py[0][idx]+larmor_vector.y;

	potential[1][idx] = ZEL*charge[idx]*plasma->Phispline.BCspline_eval<XPgridderiv_f>(px[1][idx],py[1][idx]);



	//B_vector.x = plasma->Bfieldr(px[1][idx],py[1][idx])*ZT2G;
	//B_vector.y = plasma->Bfieldz(px[1][idx],py[1][idx])*ZT2G;
	//B_vector.z = plasma->Bfieldphi(px[1][idx],py[1][idx])*ZT2G;
	//bmodFLR = pow(B_vector.x,2)+pow(B_vector.y,2)+pow(B_vector.z,2);
	//PsiGC = plasma->Psispline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]);
	//PsiFLR = plasma->Psispline.BCspline_eval<XPgridderiv_f>(px[1][idx],py[1][idx]);

	//v_vector = eval_velocity_vector(B_vector.x,B_vector.y,B_vector.z,0);



	// Conserve angular momentum
	//v_vector.z = px[0][idx]/px[1][idx]*v_vector.z+ZEL*charge[idx]/(ZMP*mass[idx]*px[1][idx])*(PsiFLR-PsiGC);

	v_mag = pow(vpara[0][idx],2)+pow(vperp[0][idx],2);
	// Conserve Energy
	if(potential[1][idx] != potential[0][idx])
		velocity = sqrt(max((v_mag+(potential[0][idx]-potential[1][idx])/V2TOEV),1.0e12f));
	else
		velocity = v_mag;
/*
	v_vector.x *= velocity/v_mag;
	v_vector.y *= velocity/v_mag;
	v_vector.z *= velocity/v_mag;

	B_vector.x /= bmodFLR;
	B_vector.y /= bmodFLR;
	B_vector.z /= bmodFLR;
*/
	//vpara[1][idx] = v_mag*(v_vector.x*B_vector.x+v_vector.y*B_vector.y+v_vector.z*B_vector.z);
	vpara[1][idx] = px[0][idx]*vpara[0][idx]/px[1][idx];

	vpara[1][idx] = max(-0.99999f*velocity,min(0.99999f*velocity,vpara[1][idx]));


	vperp[1][idx] = sqrt(max((pow(velocity,2)-pow(vpara[1][idx],2)),0.0f));
	pitch_angle[1][idx] = vpara[1][idx]/velocity;
	}
	else
	{
		vpara[1][idx] = vpara[0][idx];
		vperp[1][idx] = vperp[0][idx];
		pitch_angle[1][idx] = pitch_angle[0][idx];
		potential[1][idx] = potential[0][idx];
	}

	return;

}

__device__
void XPlist::gphase(void)
{
	unsigned int idx = threadIdx.x;

	phase_angle[idx] = 2.0*pi*get_random();
}
__device__
void XPlist::depsub(realkind3 velocityRZPhi,XPTextureSpline Phispline,XPTextureSpline Psispline,XPTextureSpline gspline,unsigned int idx = threadIdx.x)
{
	realkind3 B_vector = eval_Bvector(Psispline,gspline,1);
	realkind3 larmor_displacements;
	realkind larmor_factor;
	realkind bmod = sqrt(B_vector.x*B_vector.x+B_vector.y*B_vector.y+B_vector.z*B_vector.z);
	realkind velocity = sqrt(velocityRZPhi.x*velocityRZPhi.x+velocityRZPhi.y*velocityRZPhi.y+velocityRZPhi.z*velocityRZPhi.z);
	realkind PsiGC;
	realkind PsiFLR;
	realkind PhiGC;
	realkind PhiFLR;
	realkind dmomentum;

	B_vector.x = B_vector.x/bmod;
	B_vector.y = B_vector.y/bmod;
	B_vector.z = B_vector.z/bmod;

	larmor_displacements.x = -velocityRZPhi.z*B_vector.y-bphi_sign*velocityRZPhi.y*B_vector.z;
	larmor_displacements.y = bphi_sign*velocityRZPhi.x*B_vector.z+velocityRZPhi.z*B_vector.x;
	larmor_displacements.z = -jdotb*(velocityRZPhi.y*B_vector.x-velocityRZPhi.x*B_vector.y);

	larmor_factor = mass[idx]/(charge[idx]*(9.578E3)*bmod);

	larmor_displacements.x *= larmor_factor;
	larmor_displacements.y *= larmor_factor;
	larmor_displacements.z *= larmor_factor;

	rlarmor[idx] = sqrt(pow(larmor_displacements.x,2)+pow(larmor_displacements.y,2)+pow(larmor_displacements.z,2));

	px[0][idx] = sqrt(pow(px[1][idx]+larmor_displacements.x,2)+larmor_displacements.z*larmor_displacements.z);
	py[0][idx] = py[1][idx]+larmor_displacements.y;

	PsiGC = Psispline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]);
	PsiFLR = Psispline.BCspline_eval<XPgridderiv_f>(px[1][idx],py[1][idx]);

	dmomentum = 9.578E11*ZMP/mass[idx]*charge[idx]*(PsiGC-PsiFLR);
	vpara[0][idx] = (px[1][idx]*velocityRZPhi.z*ZMP*mass[idx]+dmomentum)/(px[0][idx]*B_vector.z*ZMP*mass[idx]);

	potential[0][idx] = ZEL*charge[idx]*Phispline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]);
	potential[1][idx] = ZEL*charge[idx]*Phispline.BCspline_eval<XPgridderiv_f>(px[1][idx],py[1][idx]);

	velocity = pow(velocity,2)*V2TOEV+potential[1][idx]-potential[0][idx];
	velocity = sqrt(velocity/V2TOEV);

	vpara[0][idx] = velocityRZPhi.z*B_vector.z-bphi_sign*(velocityRZPhi.x*B_vector.x+velocityRZPhi.y*B_vector.y);

	pitch_angle[0][idx] = vpara[0][idx]/velocity;

	vperp[0][idx] = sqrt(pow(velocity,2)-pow(vpara[0][idx],2));

	if(idx != 0) original_idx[idx] = original_idx_counter_d+blockIdx.x*blockDim.x+idx;

	deltat[idx] = 5.0*orbit_dt_min;





}

__device__
realkind3 XPlist::XPlist_derivs(XPTextureSpline Psispline,XPTextureSpline gspline,XPTextureSpline Phispline,
		realkind mu_in,int ilocation)
{ // dt, vpara,vperp,px,py should be shared arrays of size BLOCK_SIZE
	unsigned int idx = threadIdx.x;
	realkind3 result;

	// Evaluate Splines
	realkind Psi;
	realkind dPsidR;
	realkind dPsidZ;
	realkind dPsidRR;
	realkind dPsidZZ;
	realkind dPsidRZ;

	realkind g;
	realkind dgdR;
	realkind dgdZ;

	realkind Phi;
	realkind dPhidR;
	realkind dPhidZ;

	realkind3 Bvect;
	realkind B;
	realkind dBdR;
	realkind dBdZ;
	realkind rmajor;

	// Just some random variable to store coefficients
	realkind coeff;

	// (c*m*vl/q) curl b
	realkind3 Gvector;

	// mod Bs : Bs dot b
	realkind Bs;

	realkind3 bsvector;

	realkind g12;

	realkind2 df;


	if(idx < nptcls_max)
	{
		rmajor = px[ilocation][idx];
		mu_in = mu[idx];

		// Evaluate Splines
		Psi = Psispline.BCspline_eval<XPgridderiv_f>(px[ilocation][idx],py[ilocation][idx]);
		dPsidR = Psispline.BCspline_eval<XPgridderiv_dfdx>(px[ilocation][idx],py[ilocation][idx]);
		dPsidZ = Psispline.BCspline_eval<XPgridderiv_dfdy>(px[ilocation][idx],py[ilocation][idx]);
		dPsidRR = Psispline.BCspline_eval<XPgridderiv_dfdxx>(px[ilocation][idx],py[ilocation][idx]);
		dPsidZZ = Psispline.BCspline_eval<XPgridderiv_dfdyy>(px[ilocation][idx],py[ilocation][idx]);
		dPsidRZ = Psispline.BCspline_eval<XPgridderiv_dfdxy>(px[ilocation][idx],py[ilocation][idx]);

		g = gspline.BCspline_eval<XPgridderiv_f>(px[ilocation][idx],py[ilocation][idx]);
		dgdR = gspline.BCspline_eval<XPgridderiv_dfdx>(px[ilocation][idx],py[ilocation][idx]);
		dgdZ = gspline.BCspline_eval<XPgridderiv_dfdy>(px[ilocation][idx],py[ilocation][idx]);

		Phi = Phispline.BCspline_eval<XPgridderiv_f>(px[ilocation][idx],py[ilocation][idx]);
		dPhidR = Phispline.BCspline_eval<XPgridderiv_dfdx>(px[ilocation][idx],py[ilocation][idx]);
		dPhidZ = Phispline.BCspline_eval<XPgridderiv_dfdy>(px[ilocation][idx],py[ilocation][idx]);

		Bvect = Bvector(px[ilocation][idx],dPsidR,dPsidZ,g,ilocation);
		B = sqrt(dPsidR*dPsidR+dPsidZ*dPsidZ+g*g)/rmajor;
		dBdR = (dPsidR*dPsidRR+dPsidZ*dPsidRZ+g*dgdR)/(rmajor*rmajor*B)-B/rmajor;
		dBdZ = (dPsidZ*dPsidZZ+dPsidR*dPsidRZ+g*dgdZ)/(rmajor*rmajor*B);



		// Just some random variable to store coefficients
		// (c*m*vl/q) curl b
		coeff = ZC*((realkind)mass[idx])*ZMP*vpara[ilocation][idx]/(((realkind)charge[idx])*ZEL);


		Gvector.x = coeff*(g*dBdZ/B-dgdZ)/(rmajor*B);
		Gvector.y = coeff*(dgdR-g*dBdR/B)/(rmajor*B);
		Gvector.z = coeff*bphi_sign*((dPsidRR+dPsidZZ-dPsidR/rmajor)/rmajor-
											 (dPsidR*dBdR+dPsidZ*dBdZ)/(rmajor*B))/(rmajor*B);




		// mod Bs : Bs dot b
		Bs = g*(dPsidRR+dPsidZZ-dPsidR/rmajor)-(dPsidR*dgdR+dPsidZ*dgdZ);
		Bs = coeff*bphi_sign*Bs/(rmajor*rmajor*B*B);
		Bs = B+Bs;



		bsvector.x = (Bvect.x+Gvector.x)/Bs;
		bsvector.y = (Bvect.y+Gvector.y)/Bs;
		bsvector.z = (Bvect.z+Gvector.z)/Bs;


		coeff = ZC/(B*Bs*ZEL*((realkind)charge[idx]));

		g12 = g/rmajor;
		//g13 = dPsidR*bphi_sign/(px[idx]*px[idx]); Not needed because df.z = 0
		//g23 = dPsidZ*bphi_sign/(px[idx]*px[idx]); Not needed because df.z = 0



		df.x = ZEL*((realkind)charge[idx])*dPhidR+mu_in*dBdR;
		df.y = ZEL*((realkind)charge[idx])*dPhidZ+mu_in*dBdZ;
		// df.z = 0.0; This is 0 for now, it may change at a later date, but right now it is 0.0 in orbrz_derivs, so it will be 0 here

		//printf("Phi(%f,%f) = %g,%g,%g\n",rmajor,py[ilocation][idx],bsvector.x,bsvector.y,bsvector.z);


		result.x = vpara[ilocation][idx]*bsvector.x+coeff*g12*df.y;
		result.y = vpara[ilocation][idx]*bsvector.y-coeff*g12*df.x;
		result.z = -(bsvector.x*df.x+bsvector.y*df.y)/(ZMP*((realkind)mass[idx]));



		potential[ilocation][idx] = charge[idx]*ZEL*Phi;

		energy[idx] = 0.5*((realkind)mass[idx])*ZMP*vpara[ilocation][idx]*vpara[ilocation][idx]+mu_in*B+((realkind)charge[idx])*ZEL*Phi;
		momentum[idx] = ((realkind)mass[idx])*ZMP*vpara[ilocation][idx]*g/B-bphi_sign*ZEL*((realkind)charge[idx])*Psi/ZC;


	}
	else
	{
		result.x = 0;
		result.y = 0;
		result.z = 0;

		potential[ilocation][idx] = 0;
		energy[idx] = 0;
		momentum[idx] = 0;
	}

	return result;

}


// n0*sigmav for particles outside the plasma.
__device__
realkind XPlist::cxnsum_outside(Environment* plasma_in)
{
	/* TODO
	 * Include Outputs
	 */
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;

	realkind energy_ev = energy[idx]*5.2207E-13;
	realkind result =0.0;
	realkind tempfactor;
	realkind normalized_density;

	for(int i=0;i<plasma_in->ngases;i++)
	{
		tempfactor = plasma_in->cx_cross_sections.cx_outside_plasma(energy_ev,i,idBeam);
		normalized_density = plasma_in->background_density(plasma_in->ledge_transp,plasma_in->ledge_transp,i,0)/plasma_in->RhoSum;
		result += cxdt_goosed[idx]*tempfactor*normalized_density;
	}

	return result;

}

// n0*sigmav for particles inside the plasma
__device__
realkind XPlist::cxnsum_inside(Environment* plasma)
{
	/* TODO
	 *  Include outputs
	 */
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;

	realkind transp_index_flr = plasma->transp_zone(px[1][idx],py[1][idx]);
	realkind transp_index2_flr = plasma->beam_zone(px[1][idx],py[1][idx]);

	realkind energy_ev;
	realkind3 velocity;
	realkind velocity_shift;

	realkind tempfactor;
	realkind neutral_density;

	int iztarg;

	realkind result = 0.0f;

	realkind modB = sqrt(pow(plasma->Bfieldr(px[1][idx],py[1][idx]),2)+
								pow(plasma->Bfieldz(px[1][idx],py[1][idx]),2)+
								pow(plasma->Bfieldphi(px[1][idx],py[1][idx]),2));


	velocity = eval_velocity_vector(plasma->Bfieldr(px[1][idx],py[1][idx]),
												   plasma->Bfieldz(px[1][idx],py[1][idx]),
												   plasma->Bfieldphi(px[1][idx],py[1][idx]),1);

	// Wall (Edge) sourece neutrals
	for(int ig=0;ig<plasma->ngases;ig++)
	{
		velocity_shift = plasma->omega_wall_neutrals(transp_index_flr,ig)*px[1][idx];

		energy_ev = (pow(velocity.z-velocity_shift,2)+velocity.y*velocity.y+velocity.x*velocity.x)*5.2207E-13f;

		tempfactor = plasma->cx_cross_sections.cx_thcx_wall(energy_ev,transp_index_flr,ig,idBeam);

		result += tempfactor*cxdt_goosed[idx];

	}

	// Nth generation beam neutrals
	for(int isb=0;isb<plasma->nspecies;isb++)
	{
		if(plasma->beamcx_neutral_density(transp_index2_flr,isb)>0.0f)
		{

			 iztarg = plasma->species_atomic_number(isb);

			velocity_shift = plasma->beamcx_neutral_velocity(transp_index2_flr,isb);
			energy_ev = (velocity.x*velocity.x+
								 velocity.y*velocity.y+
								 pow(velocity.z-velocity_shift,2))*V2TOEV;
			energy_ev += plasma->beamcx_neutral_energy(transp_index2_flr,isb);

			energy_ev /= mass[idx];

			tempfactor = plasma->cx_cross_sections.cx_thcx_beam_beam(energy_ev,iztarg,charge[idx]);

			tempfactor *= cxdt_goosed[idx]*plasma->beamcx_neutral_density(transp_index2_flr,isb);
			tempfactor /= plasma->grid_zone_volume(transp_index2_flr);

			result+=tempfactor;
		}

	}

	// Thermal volume source neutrals
	for(int ig=0;ig<plasma->ngases;ig++)
	{
		velocity_shift = plasma->omega_thermal_neutrals(ig,transp_index2_flr);

		energy_ev = (velocity.x*velocity.x+
							 velocity.y*velocity.y+
							 pow(velocity.z-velocity_shift,2))*5.2207E-13f;

		tempfactor = plasma->cx_cross_sections.cx_thcx_halo(energy_ev,transp_index2_flr,ig,idBeam);

		result += cxdt_goosed[idx]*tempfactor;
	}

	// Beam-Beam Cx and Impact Ionization
	// Loop over relative energies
	for(int ien=0;ien<3;ien++)
	{	// Loop over beamlines
		for(int ib=0;ib<plasma->nbeams;ib++)
		{	// Only beams that are actually injecting
			if(plasma->injection_rate(ib)>0.0f)
			{	// Only H/He beams
				if(plasma->species_atomic_number(ib)<=2.05f)
				{
					iztarg = rint(plasma->species_atomic_number(ib)+0.1f);
					// Loop over ingoing/outgoing
					for(int ino=0;ino<2;ino++)
					{
						energy_ev = 0;
						neutral_density = plasma->beam_1stgen_neutral_density2d(ib,ien,ino,transp_index2_flr);
						if(neutral_density > 0.0f)
						{
							neutral_density /= plasma->grid_zone_volume(transp_index2_flr);

							// Set the directional components of the velocity
							// in the R direction
							velocity_shift = plasma->beam_ion_initial_velocity(ien,ib)*
													plasma->beam_ion_velocity_direction(ib,0,ino,transp_index2_flr);
							energy_ev += pow(velocity_shift-velocity.x,2);
							// in the Z direction
							velocity_shift = plasma->beam_ion_initial_velocity(ien,ib)*
													plasma->beam_ion_velocity_direction(ib,1,ino,transp_index2_flr);
							energy_ev += pow(velocity_shift-velocity.y,2);
							// in the toroidal direction
							velocity_shift = plasma->beam_ion_initial_velocity(ien,ib)*
													plasma->beam_ion_velocity_direction(ib,2,ino,transp_index2_flr);
							energy_ev += pow(velocity_shift-velocity.y,2);

							// Relative energy
							energy_ev = energy_ev*V2TOEV/mass[idx];
							energy_ev = max(1.0f,energy_ev);

							tempfactor = plasma->cx_cross_sections.cx_thcx_beam_beam(energy_ev,iztarg,mass[idx]);

							tempfactor *= cxdt_goosed[idx]*neutral_density; // should probably include excited states correction

							result += tempfactor;

						}
					}
				}
			}


		}
	}

	return result;

}

__device__
realkind XPlist::find_neutral_track_length(realkind3 velocity_vector,realkind rmin,realkind rmax,realkind zmin,realkind zmax)
{
	unsigned int idx = threadIdx.x;

	realkind temp_results[4];

	realkind result;

	realkind2 temp_time;

	realkind coeff_a;
	realkind coeff_b;
	realkind coeff_c;

	// The inside of the tokamak is modelled as two nested cylinders bounded by planes at zmin and zmax

	// First do the time to the outer cylinder
	coeff_a = velocity_vector.x*velocity_vector.x+velocity_vector.z*velocity_vector.z;
	coeff_b = 2*velocity_vector.x*px[1][idx];
	coeff_c = pow(px[1][idx],2)-rmax*rmax;

	temp_time = device_quadratic_equation(coeff_a,coeff_b,coeff_c);

	if((temp_time.x > 0.0f)&&(temp_time.y > 0.0f))
	{
		temp_results[0] = fmin(temp_time.x,temp_time.y);
	}
	else
	{ // Roots are either imaginary or one root is for the opposite half line

		temp_results[0] = fmax(temp_time.x,temp_time.y);
	}

	// Now do the time to the inner cylinder
	coeff_a = velocity_vector.x*velocity_vector.x+velocity_vector.z*velocity_vector.z;
	coeff_b = 2*velocity_vector.x*px[1][idx];
	coeff_c = pow(px[1][idx],2)-rmin*rmin;

	temp_time = device_quadratic_equation(coeff_a,coeff_b,coeff_c);

	if((temp_time.x > 0.0f)&&(temp_time.y > 0.0f))
	{
		temp_results[1] = fmin(temp_time.x,temp_time.y);
	}
	else
	{ // Roots are either imaginary or one root is for the opposite half line

		temp_results[1] = fmax(temp_time.x,temp_time.y);
	}


	// Now do the time to the roof and the floor
	if(velocity_vector.y != 0.0f)
	{
		// Avoid horizontal trajectories

		temp_results[2] = (zmax-py[1][idx])/velocity_vector.y;

		temp_results[3] = (zmin-py[1][idx])/velocity_vector.y;

		temp_results[2] = fmax(temp_results[2],temp_results[3]); // Get the positive one.
	}
	else
	{
		// Horrizontal trajectory

		temp_results[2] = fmax(temp_results[0],temp_results[1]);
	}

	if((temp_results[0] > 0.0f)&&(temp_results[1] > 0.0f))
	{
		result = fmin(temp_results[0],temp_results[1]);
		if(temp_results[2] > 0.0f)
		{
			result = fmin(temp_results[2],result);
		}
	}
	else
	{
		result = fmax(temp_results[2],0);
	}

	return result;
}

__device__
realkind3 XPlist::eval_plasma_frame_velocity(Environment* plasma_in)
{
	unsigned int idx = threadIdx.x;

	realkind3 result;// velocity,vparallel, pitch_angle

	realkind Bfieldr = plasma_in -> Bfieldr(px[0][idx],py[0][idx])*ZT2G;
	realkind Bfieldz = plasma_in -> Bfieldz(px[0][idx],py[0][idx])*ZT2G;
	realkind Bfieldphi = plasma_in -> Bfieldphi(px[1][idx],py[1][idx])*ZT2G;

	realkind velocity_temp = sqrt(pow(vperp[1][idx],2)+pow(vpara[1][idx],2));

	realkind Bmod = sqrt(pow(Bfieldr,2)+pow(Bfieldz,2)+pow(Bfieldphi,2));

	realkind plasma_rotation = plasma_in -> rotation(px[1][idx],py[1][idx]);

	realkind plasma_speed = plasma_rotation*px[1][idx]*Bfieldphi;

	realkind zeps = plasma_speed/velocity_temp;

	result.y = vpara[1][idx] - plasma_speed;

	result.x = velocity_temp*sqrt(max(0.0f,1.0+zeps*(zeps-2.0*pitch_angle[1][idx])));

	result.y = max(-0.9999f*result.x,min(0.9999f*result.x,result.y));

	if(result.x > 1.0e-9f)
		result.z = result.y/result.x;
	else
		result.z = 0;

	return result;


}

__device__
int XPlist::collide(Environment* plasma_in,int isteps)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.x;

	realkind3 velocities = eval_plasma_frame_velocity(plasma_in);

	realkind injection_energy = plasma_in -> injection_energy(idBeam);
	realkind energy_factor = plasma_in -> energy_factor;
	realkind energy_temp = velocities.x*velocities.x*V2TOEV;

	int transp_zone_flr = rint(plasma_in -> transp_zone(px[1][idx],py[1][idx]));
	int transp_zone_gc = rint(plasma_in -> transp_zone(px[0][idx],py[0][idx]));

	int beam_zone_flr = rint(plasma_in -> beam_zone(px[1][idx],py[1][idx]));
	int beam_zone_gc = rint(plasma_in -> beam_zone(px[0][idx],py[0][idx]));
	int nbflag = 0;

	realkind coeffC;
	realkind coeffD;
	realkind coeffE;

	realkind dVpara_toroidal = 0; // zdvpli
	realkind dVpara_same = 0; // zdvpsi
	realkind dVpara_normal = 0; // zdvpdi

	realkind electron_temperature = plasma_in -> electron_temperature(transp_zone_flr,1);
	realkind ion_temperature = plasma_in -> ion_temperature(transp_zone_flr,1);
	realkind thermal_velocity = plasma_in -> thermal_velocity(transp_zone_flr,idBeam);

	realkind vppov;
	realkind dves;
	realkind dvis;

	realkind Eforce;


	realkind epain;

	realkind electron_drag;
	realkind ion_drag;

	realkind drag_sum;
	realkind std_velocity;

	realkind velocity_old;
	realkind energy_old;

	realkind velocity_change;
	realkind energy_change;
	realkind diffusion_factor;

	realkind pitch_angle_old;

	realkind ceff;
	realkind swsq;

	realkind delTh;
	realkind sinedTh;
	realkind cosdTh;
	realkind sinePhr;
	realkind cosPhr;

	realkind vpll;
	realkind vpp1;
	realkind vpp2;

	realkind velocity_temp;

	int energy_index;



	// Compute Energy index for Fokker Planck coefficients
	if(energy_temp >= energy_factor*injection_energy)
		energy_index = 0;
	else if(energy_temp >= energy_factor*energy_factor*injection_energy)
		energy_index = 1;
	else if(energy_temp >= energy_factor*energy_factor*energy_factor*injection_energy)
		energy_index = 2;
	else
		energy_index = 3;

	// TODO Should eventually add in Heavy ion support here

	coeffC = plasma_in -> FPcoeff_arrayC(transp_zone_flr,idBeam,energy_index);
	coeffD = plasma_in -> FPcoeff_arrayD(transp_zone_flr,idBeam,energy_index);
	coeffE = plasma_in -> FPcoeff_arrayE(transp_zone_flr,idBeam,energy_index);

	Eforce = 9.58E11*(charge[idx])*(plasma_in -> loop_voltage(transp_zone_flr))*
				  (plasma_in -> current_shielding(transp_zone_flr))*
				  (plasma_in -> Bfieldphi(px[1][idx],py[1][idx])*ZT2G)/
				  (2*pi*px[1][idx]*(mass[idx]));

	velocities.x = velocities.x*velocities.x;
	velocities.x += pow(Eforce*fpdt_goosed[idx]/isteps,2);
	velocities.x += 2*Eforce*fpdt_goosed[idx]/isteps*velocities.y;
	velocities.x = sqrt(max(velocities.x,0.0f));

	velocities.y += Eforce*fpdt_goosed[idx]/isteps;

	dVpara_toroidal += Eforce*fpdt_goosed[idx]/isteps;

	velocities.z = velocities.y/velocities.x;

	vppov = sqrt(max(1.0-velocities.z*velocities.z,0.0f));

	epain = 1.0f/(velocities.x*velocities.x*(mass[idx]));

	// Electron Drag
	electron_drag = coeffC*fpdt_goosed[idx]/isteps*velocities.x*
							 (1.0-1.915E12f*epain*electron_temperature);

	// Ion Drag
	ion_drag = coeffD*fpdt_goosed[idx]/isteps*epain*(mass[idx])*
					  (1.0+9.577E11f*epain*ion_temperature);

	// Guard against too much drag at thermalization time

	drag_sum = ion_drag+electron_drag;
	if(drag_sum > velocities.x)
	{
		electron_drag *= velocities.x/drag_sum;
		ion_drag *= velocities.x/drag_sum;
		drag_sum = ion_drag+electron_drag;
	}

	velocities.x -= drag_sum;

	// Energy Diffusion
	if(abs(velocities.x) > thermal_velocity)
	{
		std_velocity = coeffC*electron_temperature*1.915E12f/(mass[idx]);
		std_velocity += ion_temperature*1.915E12f/(velocities.x*velocities.x*velocities.x*(mass[idx]))*coeffD;
		std_velocity *= fpdt_goosed[idx]/isteps;
		std_velocity = sqrt(max(std_velocity,0.0f));
		energy_old = velocities.x*velocities.x;
		velocity_old = velocities.x;

		velocities.x = max(0.5f*thermal_velocity,(randGauss(&random_state[idx],velocities.x,std_velocity)));

		velocity_change = velocity_old - velocities.x;
		energy_change = energy_old - velocities.x*velocities.x;
		diffusion_factor = coeffC*electron_temperature*energy_old*velocity_old/(coeffD*ion_temperature);
		diffusion_factor /= (diffusion_factor+1);

	}
	else
	{
		velocity_change = 0;
		energy_change = 0;
		diffusion_factor = 0;
	}

	dves = diffusion_factor*velocity_change+electron_drag;

	dvis = (1.0f-diffusion_factor)*energy_change+ion_drag;

	dVpara_toroidal -= velocities.z*(dves+dvis);
	dVpara_same -= vppov*(dves+dvis);

	if(abs(velocities.x) > thermal_velocity)
	{
		ceff = 2.0*coeffE/pow(velocities.x,3)*fpdt_goosed[idx]/isteps;
		swsq = (1.0f-velocities.z*velocities.z)*ceff;

		delTh = sqrt(max((-2.0f*ceff*log(curand_uniform(&random_state[idx]))),0.0f));

		sinedTh = sin(delTh);
		cosdTh = cos(delTh);

		delTh = curand_uniform(&random_state[idx])*2.0*pi;

		sinePhr = sin(delTh);
		cosPhr = cos(delTh);

		pitch_angle_old = velocities.z;

		velocities.z = sinedTh*vppov+cosdTh*pitch_angle_old;

		dVpara_same += velocities.x*((cosdTh-1.0)*vppov-sinedTh*cosPhr*pitch_angle_old);

		dVpara_normal += velocities.x*sinedTh*sinePhr;

		velocities.z = max(-1.0f,min(1.0f,velocities.z));

		dVpara_toroidal -= velocities.x*(pitch_angle_old-velocities.z);

	}
	else
	{
		nbflag = 2;
	}


	vpll = vpara[0][idx]+dVpara_toroidal;
	vpp1 = vperp[0][idx] + dVpara_same;
	vpp2 = dVpara_normal;

	velocity_temp = sqrt(vpll*vpll+vpp1*vpp1+vpp2*vpp2);
	velocity_temp = max(0.5*thermal_velocity,velocity_temp);

	vpara[0][idx] = max(-0.999999f*velocity_temp,min(0.999999f*velocity_temp,vpll));

	pitch_angle[0][idx] = vpara[0][idx]/velocity_temp;

	vperp[0][idx] = sqrt(max(1.0f-pow(pitch_angle[0][idx],2),0.0f))*velocity_temp;

	momentum[idx] = ZMP*mass[idx]*vpara[0][idx]*px[0][idx]*(plasma_in -> Bfieldphi(px[0][idx],py[0][idx])*ZT2G);

	if((nbflag==2))
	{
		momentum[idx] = 0.0f;
		energy[idx] = 0.0f;
		pexit[idx] = XPlistexit_thermalized;
		return 2;
	}

	return 0;

}

__device__
realkind4 XPlist::find_jacobian(XPTextureGrid* Xi_map,XPTextureGrid* Theta_map)
{
	unsigned int idx = threadIdx.x;
	realkind dxidR;
	realkind dthdR;
	realkind dxidZ;
	realkind dthdZ;
	realkind x = px[0][idx];
	realkind y = py[0][idx];
	realkind4 result;

	dxidR = Xi_map->deriv<XPgridderiv_dfdx>(x,y);
	dthdR = Theta_map->deriv<XPgridderiv_dfdx>(x,y);
	dxidZ = Xi_map->deriv<XPgridderiv_dfdy>(x,y);
	dthdZ = Theta_map->deriv<XPgridderiv_dfdy>(x,y);

	result.x = 1/dxidR;
	result.y = 1/dthdR;
	result.z = 1/dxidZ;
	result.w = 1/dthdZ;

	return result;





}

__device__
void XPlist::anomalous_diffusion(Environment* plasma_in)
{

	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;

	realkind4 jacobian = find_jacobian(&(plasma_in -> Xi_map),&(plasma_in -> Theta_map));
	realkind tau;
	realkind transp_zone = plasma_in -> transp_zone(px[0][idx],py[0][idx]);
	int nenergy_bins = plasma_in->n_diffusion_energy_bins;

	realkind velb;
	realkind2 velbrz;
	realkind difb;
	realkind2 gradD;
	realkind dDbdXi;
	realkind kenergy = (vpara[0][idx]*vpara[0][idx]+vperp[0][idx]*vperp[0][idx])*V2TOEV;

	realkind gxi;
	realkind veld;

	realkind energy_factor = 1.0f;

	realkind deltatl;
	realkind dxi_length;
	realkind tstep_length;

	realkind deltaR;
	realkind deltaZ;
	realkind deltaPhi;

	realkind temp_velocity;
	realkind tempB;
	realkind newpotential;

	//realkind mub = vperp[0][idx]*vperp[0][idx]/sqrt(tempB);

	int ictmax = 1;
	int ict = 0;

	if(nenergy_bins > 0)
	{
		if(kenergy < (plasma_in -> adif_energies(0)))
		{
			energy_factor = plasma_in -> adif_multiplier(0);
		}
		else if(kenergy >= (plasma_in -> adif_energies(nenergy_bins-1)))
		{
			energy_factor = plasma_in -> adif_multiplier(nenergy_bins-1);
		}
		else
		{
			for(int i=1;i<nenergy_bins;i++)
			{
				if(((plasma_in -> adif_energies(i-1))<=kenergy)&&((plasma_in -> adif_energies(i))>=kenergy))
				{
					energy_factor = (plasma_in->adif_multiplier(i-1))+
								((plasma_in->adif_multiplier(i))-(plasma_in->adif_multiplier(i-1)))*
								(kenergy-(plasma_in->adif_energies(i-1)))/
								((plasma_in->adif_energies(i))-(plasma_in->adif_energies(i-1)));
					break;
				}
			}
		}

	}

	while(ict < ictmax)
	{
		ict++;

		if(ict > ictmax)
			break;

		tau = 1.0f/(jacobian.x*jacobian.w-jacobian.y*jacobian.z);
		gxi = sqrt(pow(tau*jacobian.x,2)+pow(tau*jacobian.y,2));


		if((plasma_in -> is_fusion_product(idBeam)))
		{
			velb = plasma_in -> fusion_anomalous_radialv(px[0][idx],py[0][idx]);
			difb = plasma_in -> fusion_anomalous_diffusion(px[0][idx],py[0][idx]);

			gradD.x = plasma_in -> fusion_anomalous_diffusion.deriv<XPgridderiv_dfdx>(px[0][idx],py[0][idx]);
			gradD.y = plasma_in -> fusion_anomalous_diffusion.deriv<XPgridderiv_dfdx>(px[0][idx],py[0][idx]);

		}
		else
		{
			velb = plasma_in -> beam_anomalous_radialv(px[0][idx],py[0][idx]);
			difb = plasma_in -> beam_anomalous_diffusion(px[0][idx],py[0][idx]);

			gradD.x = plasma_in -> beam_anomalous_diffusion.deriv<XPgridderiv_dfdx>(px[0][idx],py[0][idx]);
			gradD.y = plasma_in -> beam_anomalous_diffusion.deriv<XPgridderiv_dfdx>(px[0][idx],py[0][idx]);
		}

		velb *= energy_factor;
		difb *= energy_factor;

		if(abs(velb) > 0)
		{
			velbrz.x = jacobian.w*velb/(sqrt(pow(jacobian.x,2)+pow(jacobian.z,2)));
			velbrz.y = -jacobian.y*velb/(sqrt(pow(jacobian.x,2)+pow(jacobian.z,2)));
		}
		else
		{
			velbrz.x = 0;
			velbrz.y = 0;
		}
		gradD.x *= energy_factor;
		gradD.y *= energy_factor;

		dDbdXi = (gradD.x*jacobian.x+gradD.y*jacobian.z)/(sqrt(pow(jacobian.x,2)+pow(jacobian.z,2)));

		veld = dDbdXi*gxi;

		if((difb > 0)||(abs(veld+velb)>0))
		{
			if(ict == 1)
			{
				dxi_length = sqrt(max(2*fpdt_goosed[idx]*difb,0.0));
				tstep_length = (plasma_in -> dxi_spacing2(transp_zone))/gxi;
				ictmax = pow(4*dxi_length/tstep_length+1,2);

				ictmax = max(ictmax,(int)rint(4*fpdt_goosed[idx]*abs(velb+veld)/tstep_length+1));
				ictmax = min(ictmax,50);
				deltatl = fpdt_goosed[idx]/ictmax;

			}

			dxi_length = sqrt(max(2.0f*deltatl*difb,0.0));
		}
		else
		{
			dxi_length = 0;
			if(ict == 1)
			{
				deltatl = fpdt_goosed[idx];
				ictmax = 1;
			}
		}

		if(dxi_length > 0)
		{
			deltaR = (randGauss(&random_state[idx],0,dxi_length));
			deltaZ = (randGauss(&random_state[idx],0,dxi_length));
			deltaPhi = (randGauss(&random_state[idx],0,dxi_length));
		}
		else
		{
			deltaR = 0;
			deltaZ = 0;
			deltaPhi = 0;
		}

		deltaR += deltatl*(gradD.x+velbrz.x);
		deltaZ += deltatl*(gradD.y+velbrz.y);

		deltaR += sqrt(pow(px[0][idx],2)+deltaPhi*deltaPhi)-px[0][idx];

		px[0][idx] += deltaR;
		py[0][idx] += deltaZ;

		transp_zone = plasma_in -> transp_zone(px[0][idx],py[0][idx]);
		jacobian = find_jacobian(&(plasma_in -> Xi_map),&(plasma_in -> Theta_map));



	}

	tempB = eval_Bmod(plasma_in->Psispline,plasma_in->gspline);

	newpotential = (plasma_in -> Phispline.BCspline_eval<XPgridderiv_f>(px[0][idx],py[0][idx]))*ZEL*charge[idx];

	temp_velocity = max(1.0f,kenergy-(newpotential-potential[0][idx]));

	temp_velocity = sqrt(temp_velocity/V2TOEV);

	vperp[0][idx] = sqrt(2.0f*tempB*mu[idx]/(ZMP*mass[idx]));

	vperp[0][idx] = min(0.999f*temp_velocity,vperp[0][idx]);

	if(vpara[0][idx] < 0)
	{
		vpara[0][idx] = -sqrt(max((temp_velocity*temp_velocity-pow(vperp[0][idx],2)),0.0f));
	}
	else
	{
		vpara[0][idx] = sqrt(max((temp_velocity*temp_velocity-pow(vperp[0][idx],2)),0.0f));
	}

	pitch_angle[0][idx] = vpara[0][idx]/temp_velocity;

}

__device__
void XPlist::update_timing_factors(Environment* plasma_in)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;

	realkind faccx;
	realkind facfp;
	realkind stpor = steps_midplanecx[idx];
	realkind dltcx;
	realkind maxsk = max(1.0f,stpor/3.0f);
	realkind fppcon = plasma_in -> fppcon;
	realkind cxpcon = plasma_in -> cxpcon;
	realkind cxskip_temp = cxskip[idx];
	realkind fpskip_temp;

	facfp = stpor/(max(1.0f,fpskip[idx])*(realkind)fp_count[idx]);

	faccx = 0.0;
	if(cx_count[idx] > 0)
		faccx = stpor/(cxskip_temp*(realkind)cx_count[idx]);

	dltcx = faccx*(time_since_cx[idx]-time_since_cx_pass[idx]);

	fpskip_temp = min(maxsk,max(1.0f,stpor/fppcon));

	if(cx_count[idx] >= 3)
	{
		if(dltcx == 0.0f)
		{
			cxskip_temp = maxsk;
		}
		else
		{
			cxskip_temp = stpor/(cxpcon*dltcx);
		}
	}
	else
	{
		cxskip_temp = max(1.0f,fpskip_temp);
	}

	cxskip_temp = min(maxsk,max(1.0f,cxskip_temp));

	cxskip[idx] = cxskip_temp;
	fpskip[idx] = fpskip_temp;

	bounce_init(0,plasma_in);

}

__device__
void XPlist::bounce_init(int jcall,Environment* plasma_in)
{
	unsigned int idx = threadIdx.x;
	unsigned int idBeam = blockIdx.y;

	steps_midplanecx[idx] = 0;
	cx_count[idx] = 0;
	fp_count[idx] = 0;

	istep_next_cx[idx] = 1+cxskip[idx]*curand_uniform(&random_state[idx]);
	istep_next_fp[idx] = 1+max(1.0,fpskip[idx])*curand_uniform(&random_state[idx]);

	if(jcall == 1)
	{
		time_till_next_cx[idx] = -log(curand_uniform(&random_state[idx]));
		time_since_cx[idx] = 0;
	}
	else
	{
		time_till_next_cx[idx] -= time_since_cx[idx];
		time_since_cx[idx] = 0;
	}

	time_since_cx_pass[idx] = 0;
}




__global__
void XPlist_setup_kernel(Environment* plasma_in,XPlist* particles_global,
										  cudaMatrixd xksidy,cudaMatrixd vay,cudaMatrixd wghtay,
										  cudaMatrixd xiay,cudaMatrixd thay,cudaMatrixd xzionay,
										  cudaMatrixd cxpray,cudaMatrixd fppray,cudaMatrixd xzbeama,
										  cudaMatrixd abeama,cudaMatrixui splitting_condition,
										  int** int_data_in,
										  size_t nbi_ipitch,size_t nbi_dpitch,int minb,int mibs)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;

	unsigned int block_start = (idBeam*(nbi_dpitch));
	unsigned int block_start_i = (idBeam*(nbi_ipitch));

	__shared__ XPlist particles;
	__shared__ int nptcls[BLOCK_SIZE];
	//if(idx == 0)
	//{
	//	particles = *particles_global;
	//}
	__syncthreads();

	nptcls[idx] = 0;

	particles.shift_local(particles_global);


	__syncthreads();



	unsigned int element_id = gidx+gridDim.x*idBeam;

	int* nbscay = int_data_in[0]+blockIdx.x*blockDim.x+block_start_i/sizeof(int);
	int* nbndex = int_data_in[1]+blockIdx.x*blockDim.x+block_start_i;

	realkind Xi;
	realkind theta;
	realkind r;
	realkind z;
	int nr = plasma_in -> griddims.x;
	int nz = plasma_in -> griddims.y;
	int nptcls_block;
	realkind Bfield;

	__syncthreads();

	int2 spline_index;
	realkind2 spline_params;
	if(gidx == 0)
		particles.species[idBeam] = idBeam;



	if(gidx < minb)
	{
		particles.time_done[idx] = 0;

		if(fabs(vay(gidx,idBeam)) > 0)
		{
			Xi = (realkind)xiay(gidx,idBeam);
			theta = (realkind)thay(gidx,idBeam);

			plasma_in -> polarintrp(Xi,theta,&spline_params,&spline_index);

			if(spline_index.y >= (plasma_in -> nint-1))
			{
				spline_index.y -= (plasma_in -> nint-1);
				r = (plasma_in -> rsplinex).BCspline_eval<XPgridderiv_f>(spline_index,spline_params);
				z = (plasma_in -> zsplinex).BCspline_eval<XPgridderiv_f>(spline_index,spline_params);
			}
			else
			{
				r = (plasma_in -> rspline).BCspline_eval<XPgridderiv_f>(spline_index,spline_params);
				z = (plasma_in -> zspline).BCspline_eval<XPgridderiv_f>(spline_index,spline_params);
			}

		//	printf("For particle %i,%i: \n position = %f, %f \n ",gidx,idBeam,r,z);

			particles.px[0][idx] = r;
			particles.py[0][idx] = z;

			particles.pitch_angle[0][idx] = (realkind)xksidy(gidx,idBeam);
			particles.vperp[0][idx] = sqrt(1.0f-min(1.0f,(pow(particles.pitch_angle[0][idx],2))))*((realkind)vay(gidx,idBeam));
			particles.vpara[0][idx] = ((realkind)xksidy(gidx,idBeam))*((realkind)vay(gidx,idBeam));
			particles.mass[idx] = max(1,(int)ceil(abeama(idBeam)));
			particles.charge[idx] = max(1,(int)ceil(xzionay(idBeam)));
			particles.atomic_number[idx] = max(1,(int)ceil(xzionay(idBeam)));
			particles.weight[idx] = wghtay(gidx,idBeam);

			particles.cxskip[idx] = max(cxpray(gidx,idBeam),1.0);
			particles.fpskip[idx] = fppray(gidx,idBeam);

			particles.beam_source[idx] = nbscay[idx];
			particles.nbndex[idx] = nbndex[idx];

			Bfield = particles.eval_Bmod(plasma_in -> Psispline,plasma_in -> gspline);

			particles.nx[0][idx] = (particles.px[0][idx]-(plasma_in -> Rmin))/(plasma_in -> gridspacing.x);
			particles.ny[0][idx] = (particles.py[0][idx]-(plasma_in -> Zmin))/(plasma_in -> gridspacing.y);

			particles.nx[0][idx] = max(2,min(nr-3,particles.nx[0][idx]));
			particles.ny[0][idx] = max(2,min(nz-3,particles.ny[0][idx]));

			particles.calc_binid(plasma_in,0,idx);

			particles.update_gc(plasma_in);

			particles.original_idx[idx] = gidx;
			particles.old_idx[idx] = gidx;

			particles.time_done[idx] = 0;
			particles.steps_midplanecx[idx] = 0;


			particles.orbflag[idx] = 1;
			particles.pexit[idx] = XPlistexit_newparticle;

			particles.deltat[idx] = 5.0*orbit_dt_min;

			particles.bounce_init(1,plasma_in);


			//curand_init(112345,element_id,idx,&(particles.random_state[idx]));

			nptcls[idx] = 1;



			//particles.print_members();




		}
		else
		{
			particles.orbflag[idx] = 0;
			particles.pexit[idx] = XPlistexit_neverorbiting;

		}

		if(particles.orbflag[idx] == 1)
		{
			splitting_condition(gidx,idBeam) = 0;
		}
		else
		{
			splitting_condition(gidx,idBeam) = 1;
		}

	}

	__syncthreads();

	nptcls_block = reduce<BLOCK_SIZE>(nptcls);

	__syncthreads();

	if(idx==0)
	{
		nptcls_block = atomicAdd((particles_global -> nptcls+idBeam),nptcls_block);
		//particles_global -> nptcls[idBeam] = minb;
		atomicMax(&(particles_global->nptcls_max),particles_global -> nptcls[idBeam]);


			//printf("nptcls_max = %i \n",particles_global->nptcls_max);
	}


}

__global__
void bounce_init_1_kernel(Environment* plasma_in,XPlist* particles_global)
{

	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;
	unsigned int idBeam = blockIdx.y;

	__shared__ XPlist particles;

	unsigned int orbflag = 0;


	particles.shift_local(particles_global);


	__syncthreads();

	if(idx < particles.nptcls_max)
	{
		orbflag = particles.orbflag[idx];
	}

	if(orbflag == 1)
	{
		particles.bounce_init(1,plasma_in);
	}

}

__host__

void XPlist::setup(Environment* plasma_d,double** dbl_data_in_h,int** int_data_in_h,int minb,int mibs)
{
	int ndoubles = 10;
	int nints = 2;

	size_t dpitch;
	size_t ipitch;

	size_t hdpitch = minb*sizeof(double);
	size_t hipitch = minb*sizeof(int);

	XPlist* particles_d;
	XPlist extra_particles;



	nptcls_max = 0;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,sizeof(XPlist)));
	CUDA_SAFE_CALL((cudaMemcpy(particles_d,this,sizeof(XPlist),cudaMemcpyHostToDevice)));
	CUDA_SAFE_CALL(cudaMemset(nptcls,0,mibs*sizeof(int)));
	nptcls_max = minb;
	nspecies = mibs;

	cudaMatrixd xksidy(minb,mibs);
	cudaMatrixd vay(minb,mibs);
	cudaMatrixd wghtay(minb,mibs);
	cudaMatrixd xiay(minb,mibs);
	cudaMatrixd thay(minb,mibs);
	cudaMatrixd xzionay(minb,mibs);
	cudaMatrixd cxpray(minb,mibs);
	cudaMatrixd fppray(minb,mibs);
	cudaMatrixd xzbeama(mibs);
	cudaMatrixd abeama(mibs);

	xksidy.cudaMatrixcpy(dbl_data_in_h[0],cudaMemcpyHostToDevice);
	vay.cudaMatrixcpy(dbl_data_in_h[1],cudaMemcpyHostToDevice);
	wghtay.cudaMatrixcpy(dbl_data_in_h[2],cudaMemcpyHostToDevice);
	xiay.cudaMatrixcpy(dbl_data_in_h[3],cudaMemcpyHostToDevice);
	thay.cudaMatrixcpy(dbl_data_in_h[4],cudaMemcpyHostToDevice);
	xzionay.cudaMatrixcpy(dbl_data_in_h[5],cudaMemcpyHostToDevice);
	cxpray.cudaMatrixcpy(dbl_data_in_h[6],cudaMemcpyHostToDevice);
	fppray.cudaMatrixcpy(dbl_data_in_h[7],cudaMemcpyHostToDevice);
	xzbeama.cudaMatrixcpy(dbl_data_in_h[8],cudaMemcpyHostToDevice);
	abeama.cudaMatrixcpy(dbl_data_in_h[9],cudaMemcpyHostToDevice);

	cudaMatrixui splitting_condition(nptcls_max,mibs);



	dim3 cudaBlockSize(BLOCK_SIZE,1,1);
	dim3 cudaGridSize((minb+BLOCK_SIZE-1)/BLOCK_SIZE,mibs,1);

	dim3 cudaBlockSize2(512,1,1);
	dim3 cudaGridSize2((minb+512-1)/512,mibs,1);

	int nelements = minb*mibs;

	int* int_data_in_d[nints];
	int** int_data_in_ptr_d;
	//__align__(16) double* dbl_data_in_d[ndoubles];
	//__align__(8) double** dbl_data_in_ptr_d;
	//cudaMalloc((void**)&dbl_data_in_ptr_d,sizeof(double*)*10);
	cudaMalloc((void**)&int_data_in_ptr_d,sizeof(int*)*2);
/*
	for(int i=0;i<ndoubles-2;i++)
	{

		cudaMallocPitch((void**)&(dbl_data_in_d[i]),&dpitch,minb*sizeof(double),mibs);

		CUDA_SAFE_KERNEL(cudaMemcpy2D(dbl_data_in_d[i],dpitch,
															  dbl_data_in_h[i],hdpitch,
															  minb*sizeof(double),mibs,
															  cudaMemcpyHostToDevice));
	}

	cudaMalloc((void**)&dbl_data_in_d[8],mibs*sizeof(double));
	cudaMalloc((void**)&dbl_data_in_d[9],mibs*sizeof(double));

	CUDA_SAFE_KERNEL(cudaMemcpy(dbl_data_in_d[8],dbl_data_in_h[8],mibs*sizeof(double),cudaMemcpyHostToDevice));
	CUDA_SAFE_KERNEL(cudaMemcpy(dbl_data_in_d[9],dbl_data_in_h[9],mibs*sizeof(double),cudaMemcpyHostToDevice));
*/

	CUDA_SAFE_CALL(cudaMallocPitch((void**)&int_data_in_d[0],&ipitch,minb*sizeof(int),mibs));

	CUDA_SAFE_KERNEL(cudaMemcpy2D(int_data_in_d[0],ipitch,
														  int_data_in_h[0],hipitch,
														  minb*sizeof(int),mibs,
														  cudaMemcpyHostToDevice));

	cudaMallocPitch((void**)&int_data_in_d[1],&ipitch,minb*sizeof(int),mibs);

	CUDA_SAFE_KERNEL(cudaMemcpy2D(int_data_in_d[1],ipitch,
														  int_data_in_h[1],hipitch,
														  minb*sizeof(int),mibs,
														  cudaMemcpyHostToDevice));

	CUDA_SAFE_KERNEL((curand_init_kernel<<<cudaGridSize,cudaBlockSize>>>(
										random_state,nbi_curpitch,random_state_counter,nptcls_max)));
	random_state_counter += minb*mibs;

//	CUDA_SAFE_KERNEL(cudaMemcpy(dbl_data_in_ptr_d,dbl_data_in_d,10*sizeof(double*),cudaMemcpyHostToDevice));
	CUDA_SAFE_KERNEL(cudaMemcpy(int_data_in_ptr_d,int_data_in_d,2*sizeof(int*),cudaMemcpyHostToDevice));

	CUDA_SAFE_KERNEL((XPlist_setup_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,particles_d,
								xksidy,vay,wghtay,xiay,thay,xzionay,cxpray,fppray,xzbeama,abeama,
								splitting_condition,int_data_in_ptr_d,ipitch,dpitch,minb,mibs)));


	cudaDeviceSynchronize();
	CUDA_SAFE_KERNEL(cudaMemcpy(this,particles_d,sizeof(XPlist),cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	printf("!!nptcls_max = %i \n",nptcls_max);

	//extra_particles = split(splitting_condition,1);
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL((cudaMemcpy(particles_d,this,sizeof(XPlist),cudaMemcpyHostToDevice)));

	cudaGridSize.x = (nptcls_max+BLOCK_SIZE-1)/BLOCK_SIZE;

//	CUDA_SAFE_KERNEL((bounce_init_1_kernel<<<cudaGridSize,cudaBlockSize>>>(plasma_d,particles_d)));


	cudaDeviceSynchronize();






	random_state_counter += nptcls_max;

	CUDA_SAFE_CALL((cudaMemcpyToSymbol(random_state_counter_symbol,
								&random_state_counter,sizeof(int))));




	//extra_particles.XPlistFree();


	cudaFree(particles_d);
	splitting_condition.cudaMatrixFree();
	xksidy.cudaMatrixFree();
	vay.cudaMatrixFree();
	wghtay.cudaMatrixFree();
	xiay.cudaMatrixFree();
	thay.cudaMatrixFree();
	xzionay.cudaMatrixFree();
	cxpray.cudaMatrixFree();
	fppray.cudaMatrixFree();
	xzbeama.cudaMatrixFree();
	abeama.cudaMatrixFree();
	cudaFree(int_data_in_d[0]);
	cudaFree(int_data_in_d[1]);
	cudaFree(int_data_in_ptr_d);



}

__global__
void XPlist_check_sort_kernel(XPlist particles_global)
{
	unsigned int idBeam = blockIdx.x;
	int k;
	int dx;
	int dy;



	__shared__ XPlist particles;

	particles.shift_local(&particles_global);
	__syncthreads();

	if(threadIdx.x == 0)
	{
		for(int i=1;i<particles_global.nptcls[idBeam];i++)
		{
			particles.print_members(i,i,idBeam);
		//	printf("for particle %i\ncellindex = %i\n",i,particles.cellindex[0][i]);
			/*
			if(particles.cellindex[0][i-1] != particles.cellindex[0][i])
			{
				k = i+1;
				while((particles.cellindex[0][i] == particles.cellindex[0][k])&&(k<particles_global.nptcls[idBeam]))
				{
					dx = particles.nx[0][i] - particles.nx[0][k];
					dy = particles.ny[0][i] - particles.ny[0][k];
					printf("for particles %i : (%i) and %i : (%i)\n",i,particles.cellindex[0][i],k,particles.cellindex[0][k]);
					printf("dx = %i, dy = %i\n",dx,dy);
					k++;
				}

			}
			*/
		}
	}
}

__host__
void XPlist::check_sort(void)
{
	CUDA_SAFE_KERNEL((XPlist_check_sort_kernel<<<nspecies,256>>>(*this)));
}

__device__
void XPlist::print_members(int idx,int gidx,int idBeam)
{

	printf("For particle %i,%i:\nposition = %10.7f, %10.7f\n ",gidx,idBeam,px[0][idx],py[0][idx]);
	printf("velocity = %g, %g, %g\n",vperp[0][idx],vpara[0][idx],pitch_angle[0][idx]);
	printf("rlarmor = %g\n",rlarmor[idx]);
	printf("mass/charge = %i, %i, %i\n",mass[idx],charge[idx],atomic_number[idx]);
	printf("index %i, %i, orbflag = %i \n",nx[0][idx],ny[0][idx],orbflag[idx]);
}


#ifdef DO_BEAMCX
__global__
void XPlist_check_CX(XPlist particles_global,cudaMatrixui splittinglist, int istep)
{
	/*
	 * This kernel checks to see which particles need to check for charge exchange events
	 * Then this kernel totals up the number of particles that need to undergo events.
	 * This kernel also sets up block_info_d for later calls
	 */

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;
	unsigned int block_start = blockIdx.x*blockDim.x;

	__shared__ int sdata[BLOCK_SIZE2];

	__shared__ int shift_param;

	__shared__ XPlist particles;

	sdata[idx] = 0;

	int iinc;

	if(gidx < particles_global.nptcls_max)
		splittinglist(gidx,idBeam) = 0;

if(block_start < particles_global.nptcls[idBeam])
{
	particles.shift_local(&particles_global);

	__syncthreads();
	sdata[idx] = 1;

	if(idx < particles.nptcls_max)
	{
		if((particles.steps_midplanecx[idx] >= particles.istep_next_cx[idx])&&(particles.orbflag[idx] == 1))
		{
			iinc = (int)rint(particles.cxskip[idx]);
			if((particles.cxskip[idx] - iinc)>particles.get_random()) iinc++;

			particles.istep_next_cx[idx] += iinc;

			particles.cx_count[idx]++;

			sdata[idx] = 1;
		}
		else
		{
			sdata[idx] = 0;
		}


	}
	__syncthreads();

	if(idx < particles.nptcls_max)
	{
		splittinglist(gidx,idBeam) = sdata[idx];
		//splittinglist(gidx,idBeam) = 0;


	}
}
}


__global__
void beamcx_kernel(Environment* plasma_in,XPlist particles_global,cudaMatrixr nutrav_weight_in,cudaMatrixi ievent,
								cudaMatrixui nsplit,cudaMatrixui splittinglist)
{

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;
	unsigned int block_start = blockIdx.x*blockDim.x;

	unsigned int itry_nsplit = 0;

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
	int idocx = 0;
	unsigned int orbflag = 0;

	int jsplit = 0;

	realkind weight_sum = 0.0;

	if(gidx < particles_global.nptcls[idBeam])
	{
		idocx = splittinglist(gidx,idBeam);
	}

	__shared__ XPlist particles;

	//__shared__ realkind* nutrav_weight;

	__shared__ int sdata[BLOCK_SIZE];

if(block_start < particles_global.nptcls[idBeam])
{

	particles.shift_local(&particles_global);
	//if(idx == 0) nutrav_weight = &nutrav_weight_in(gidx,idBeam);

	__syncthreads();

	if((idx<particles.nptcls_max)&&(idocx==1))
	{
		if(particles.orbflag[idx] == 1)
		{
			nsplit(gidx,idBeam) = 0;
			ievent(gidx,idBeam) = 0;
			// First Get the probability that we have a charge exchange

			switch(particles.check_outside(plasma_in))
			{
			case 0: // Particles are outside the plasma
				cxnsum = particles.cxnsum_outside(plasma_in);
				break;
			case 1: // Particles are inside the plasma
				cxnsum = particles.cxnsum_inside(plasma_in);
				break;
			default:
				break;
			}


			splitting_factor = particles.weight[idx]/plasma_in->average_beam_weight(plasma_in->lcenter_transp-1,idBeam)/plasma_in->average_weight_factor;



			splitting_factor = min(splitting_factor,0.25*plasma_in->max_particles);

			isplit = floor(splitting_factor);
			if(particles.get_random() < (splitting_factor-isplit)) isplit+=1;
			isplit = max(1,isplit);

		//	printf("splitting_factor = %14.10g for particle with weight %14.10g, isplit = %i\n",
			//		splitting_factor,particles.weight[idx],isplit);

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

				ireto = 1;
				imul = 1;
				dtime_step = (particles.time_till_next_cx[idx]-time_since_cx_old)/dtime_since_cx*particles.cxdt_goosed[idx];
				time_step = dtime_step;

				neutrals_remaining = isplit-1;

				while(neutrals_remaining > 0)
				{
					time_next = -log(particles.get_random());
					time_left = (particles.cxdt_goosed[idx]-time_step)/particles.cxdt_goosed[idx]*(neutrals_remaining)*dtime_since_cx;

					if(time_next > time_left)
					{
						particles.time_till_next_cx[idx] = time_next-time_left;
						weight_sum +=  (particles.cxdt_goosed[idx]-time_step)*neutrals_remaining*particles.weight[idx]/((realkind)isplit);
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
						weight_sum += dtime_step*neutrals_remaining*particles.weight[idx]/((realkind)isplit);
						neutrals_remaining -= 1;
						time_step += dtime_step;
					}

					if(itry_nsplit >= 2*Max_Splits)
					{
						break;
					}
					itry_nsplit++;
				}

				switch(particles.check_outside(plasma_in))
				{
				case 0: // Particles are outside the plasma
					cxnsum = particles.cxnsum_outside(plasma_in);
					break;
				case 1: // Particles are inside the plasma
					cxnsum = particles.cxnsum_inside(plasma_in);
					break;
				default:
					break;
				}


				// Total weight to be followed as fast cx neutrals in nutrav
				nutrav_weight_in(gidx,idBeam) = imul*particles.weight[idx]/((realkind)isplit);

				// If orbiting particle survives, reduce its weight
				if(ireto == 1)
				{
					particles.weight[idx] -= imul*particles.weight[idx]/((realkind)isplit);
				}

				jsplit = min(Max_Splits,max(((int)rint(imul*(plasma_in->cxsplit))),1));

				if(jsplit < Max_Splits)
				{
					if(particles.get_random() < ((plasma_in->cxsplit-(realkind)jsplit))) jsplit += 1;
				}

				if(cxnsum == 0.0f) jsplit -=1;



				if((jsplit < 1)&&(ireto < 1))
				{
					particles.orbflag[idx] = 0;
					particles.pexit[idx] = XPlistexit_neutralized;
				}

				nsplit(gidx,idBeam) = jsplit;
				if(jsplit > 0)
				{
					ievent(gidx,idBeam) = 4;

				}


			}
		}

		if(jsplit > 0)
		{
			splittinglist(gidx,idBeam) = 1;

			//printf("parent %i spawned %i neutrals \n",gidx,jsplit);
			if(particles.orbflag[idx] == 0)
				{
					splittinglist(gidx,idBeam) = 0;
				}
		}
		else
		{
			splittinglist(gidx,idBeam) = 0;
		}
	}
}

}



#endif

#ifndef DONT_DO_NUTRAV

__global__
void setup_nutrav(Environment* plasma_in,XPlist cx_particles, XPlist neutrals_global,cudaMatrixr nutrav_weight_in,
								 cudaMatrixui nsplit_global,cudaMatrixui old_ids,int nptcls_cx_h)
{

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int bidx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;


	realkind split_weight = 0.0;
	realkind normalized_weight;

	__shared__ XPlist neutrals;
	__shared__ XPlist parents;
	__shared__ realkind weight_sum[64];

	weight_sum[idx] = 0;
	weight_sum[idx+32] = 0;

	__shared__ unsigned int nsplit;
	__shared__ unsigned int old_id;
	__shared__ realkind dphase_angle;

	if(idx == 0)
	{
		if(nptcls_cx_h > 0)
		{old_id = old_ids(bidx,idBeam);}
		else
		{old_id = bidx;}

		nsplit = nsplit_global(old_id,idBeam);
		dphase_angle = 2.0*pi/nsplit;
		printf("parent %i, %i spawned %i neutrals \n",bidx,old_id,nsplit);
		neutrals_global.nptcls[idBeam] = neutrals_global.nptcls_max;
	}
	__syncthreads();


	parents.shared_parent(&cx_particles,bidx);
	neutrals.shift_local(&neutrals_global);
	__syncthreads();

	if(idx < nsplit)
	{
		neutrals.copyfromparent(&parents);

		if(neutrals.phase_angle[idx]+dphase_angle*idx < 2.0*pi)
		{
			neutrals.phase_angle[idx] += dphase_angle*idx;

		}
		else
		{
			neutrals.phase_angle[idx] +=dphase_angle*idx-2.0*pi;
		}

		neutrals.update_flr(plasma_in);

		switch(neutrals.check_outside(plasma_in))
		{
		case 0:
			split_weight = neutrals.cxnsum_outside(plasma_in);
			break;
		case 1:
			split_weight = neutrals.cxnsum_inside(plasma_in);
			break;
		default:
			break;
		}

		neutrals.orbflag[idx] = 1;
		neutrals.pexit[idx] = XPlistexit_newparticle;

	}
	else
	{
		neutrals.orbflag[idx] = 0;
		neutrals.pexit[idx] = XPlistexit_neverorbiting;
	}

	__syncthreads();
	weight_sum[idx] = split_weight;
	__syncthreads();

	normalized_weight = nutrav_weight_in(old_id,idBeam)/(reduce<32>(weight_sum));

	__syncthreads();

	if(idx < nsplit)
	{
		neutrals.weight[idx] *= normalized_weight;

	}

}

__global__
void nutrav_kernel(Environment* plasma_in,XPlist neutrals_global,
								cudaMatrixT<realkind3> velocity_vector_out,cudaMatrixui splittinglist)
{
	unsigned int idx = threadIdx.x;
	unsigned int parent_idx = blockIdx.x;
	unsigned int idBeam = blockIdx.y;

	size_t ipitch = neutrals_global.nbi_ipitch/sizeof(int);

	if(neutrals_global.orbflag[parent_idx+ipitch*idBeam] == 0)
	{
		splittinglist(parent_idx,idBeam) = 1;
		return;
	}

	// This kernel launches a separate thread for each track segment, each block is 1 neutral

	__shared__ XPlist neutral;
	__shared__ realkind p_recapture[Max_Track_segments];
	__shared__ realkind probability_integral[Max_Track_segments];
	__shared__ int recapture_point[Max_Track_segments];
	__shared__ realkind b_vector;

	neutral.shared_parent(&neutrals_global,parent_idx);
	__syncthreads();

	__shared__ realkind3 v_vector;
	__shared__ realkind track_length;
	__shared__ realkind velocity;
	__shared__ realkind random_number;
	__shared__ realkind russian_roulette_num;

	volatile realkind* volatile_ptr = probability_integral;
	volatile int* volatile_iptr = recapture_point;

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

if(neutral.orbflag[0] == 1)
{
	// First thread sets up everthing
	if(idx == 0)
	{
		// Figure out what direction the Bfield is in
		v_vector = neutral.eval_Bvector(plasma_in->Psispline,plasma_in->gspline,0);

		// Figure out velocity vector

		v_vector = neutral.eval_velocity_vector(v_vector.x,v_vector.y,v_vector.z,1);

		velocity = sqrt(v_vector.x*v_vector.x+v_vector.y*v_vector.y+v_vector.z*v_vector.z);

		// Figure out the track length to the boundary of the grid

		track_length = neutral.find_neutral_track_length(v_vector,plasma_in->Rmin,plasma_in->Rmax,plasma_in->Zmin,plasma_in->Zmax);
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

	plasma_velocity = plasma_in->rotation(RZPhiposition[idx].x,RZPhiposition[idx].y)*RZPhiposition[idx].z;

	velocityR = position[idx].x*v_vector.x/RZPhiposition[idx].x+position[idx].y*v_vector.z/RZPhiposition[idx].x;
	velocityZ = v_vector.y;
	velocityPhi = -position[idx].z*v_vector.x/RZPhiposition[idx].x+position[idx].x*v_vector.z/RZPhiposition[idx].x;

	velocity_relative = velocityR*velocityR+velocityZ*velocityZ+pow(plasma_velocity-velocityPhi,2);

	transp_zone = plasma_in->transp_zone(RZPhiposition[idx].x,RZPhiposition[idx].y);
	beam_zone = plasma_in->beam_zone(RZPhiposition[idx].x,RZPhiposition[idx].y);

	// Interpolate table of plasma interactions
	plasma_ionization_frac = plasma_in->cx_cross_sections.thermal_total(velocity_relative*V2TOEV/neutral.mass[0],transp_zone,idBeam);

	// Interpolate table of Beam-Beam interactions

	beam2_ionization_frac = 0.0f;

	for(int isb=0;isb<plasma_in->nbeams;isb++)
	{
		velocity_relative = velocityR*velocityR+
									 velocityZ*velocityZ+
									 pow(velocityPhi-plasma_in->toroidal_beam_velocity(beam_zone,isb),2);
		beam2_ionization_frac += plasma_in->cx_cross_sections.beam_beam_cx(
												velocity_relative*V2TOEV/neutral.mass[0],beam_zone,isb,idBeam);
		beam2_ionization_frac += plasma_in->cx_cross_sections.beam_beam_ii(
												velocity_relative*V2TOEV/neutral.mass[0],beam_zone,isb,idBeam);
	}

	inverse_mfp = (beam2_ionization_frac+plasma_ionization_frac)/velocity;

	probability_integral[idx] = inverse_mfp*velocity*track_length/Max_Track_segments;
	__syncthreads();

	// Sum up the probability integral
	for(int i=1;i<Max_Track_segments;i<<=1)
	{
		if(idx > (i-1))temp_exp = volatile_ptr[idx-i]+volatile_ptr[idx];
		__syncthreads();
		if(idx > (i-1))volatile_ptr[idx] = temp_exp;
		__syncthreads();
	}

	// Calculate the exponential
	if(probability_integral[idx] > 80.0f)
	{
		p_recapture[idx] = 0.0f;
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

	// Sum up the recapture probability
	volatile_ptr = p_recapture;
	for(int i=1;i<Max_Track_segments;i<<=1)
	{
		if(idx > (i-1))temp_exp = volatile_ptr[idx-i]+volatile_ptr[idx];
		__syncthreads();
		if(idx > (i-1))volatile_ptr[idx] = temp_exp;
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

	volatile_iptr = recapture_point;
	__syncthreads();

	if (idx <  64) { volatile_iptr[idx] = min(volatile_iptr[idx+64],volatile_iptr[idx]); }
	__syncthreads();
	if (idx < 32)
	{
		volatile_iptr[idx] = min(volatile_iptr[idx+32],volatile_iptr[idx]);
		volatile_iptr[idx] = min(volatile_iptr[idx+16],volatile_iptr[idx]);
		volatile_iptr[idx] = min(volatile_iptr[idx+8],volatile_iptr[idx]);
		volatile_iptr[idx] = min(volatile_iptr[idx+4],volatile_iptr[idx]);
		volatile_iptr[idx] = min(volatile_iptr[idx+2],volatile_iptr[idx]);
		volatile_iptr[idx] = min(volatile_iptr[idx+1],volatile_iptr[idx]);
	}

	__syncthreads();

	if(recapture_point[0] < Max_Track_segments)
	{
		// Recapture the neutral
		if(idx == recapture_point[0])
		{
			rr_factor = min(1.0f,neutral.weight[0]/plasma_in->average_beam_weight(transp_zone,idBeam)/plasma_in->average_weight_factor);
			// Russian Roulette
			if(russian_roulette_num > rr_factor)
			{
				neutral.pexit[0] = XPlistexit_russian_roulette;
				neutral.orbflag[0] = 0;
			}
			else
			{
				neutral.px[1][0] = RZPhiposition[idx].x;
				neutral.py[1][0] = RZPhiposition[idx].y;
				v_vector.x = velocityR;
				v_vector.y = velocityZ;
				v_vector.z = velocityPhi;

				velocity_vector_out(parent_idx,idBeam) = v_vector;
				neutral.pexit[0] = XPlistexit_newparticle;
				neutral.orbflag[0] = 1;
			}
		}
	}
	else
	{
		neutral.pexit[0] = XPlistexit_neutralWallLoss;
		neutral.orbflag[0] = 0;
	}
}
	// We want to remove all of the particles that aren't going to be deposited from the list
__syncthreads();
	if(neutral.orbflag[0] == 1)
	{
		splittinglist(parent_idx,idBeam) = 0;
	}
	else
	{
		splittinglist(parent_idx,idBeam) = 1;
	}


	return;
}

__global__
void recapture_neutrals(Environment* plasma_in,XPlist neutrals_global,
										  cudaMatrixT<realkind3> velocityRZPhi,cudaMatrixui old_ids,int nneutrals_dead)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;
	unsigned int idBeam = blockIdx.y;

	unsigned int old_id;

	int nx,ny;

	__shared__ XPlist neutrals;

	neutrals.shift_local(&neutrals_global);
	__syncthreads();

	if(idx < neutrals.nptcls_max)
	{
		if(nneutrals_dead > 0)
			{old_id = old_ids(gidx,idBeam);}
		else
			{old_id = gidx;}

		neutrals.depsub(velocityRZPhi(old_id,idBeam),
				plasma_in->Phispline,plasma_in->Psispline,plasma_in->gspline);
		neutrals.update_gc(plasma_in);
		neutrals.gphase();
		neutrals.update_flr(plasma_in);
		neutrals.calc_binid(plasma_in,0,idx);

	}

}


#endif


















































