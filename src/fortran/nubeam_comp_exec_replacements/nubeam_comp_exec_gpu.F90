program nubeam_comp_exec
  ! DMC Apr 2008:
  ! file based NUBEAM component driver -- serial version.
  !    generalization for MPI use: Kumar & DMC May 2008 ... 
  !    DMC -- convert to use mpi_portlib -- Nov. 2008

  ! Control of actions and path to working directory is set by 
  ! ENVIRONMENT VARIABLES; then, a namelist is read to set specific
  ! filenames.  Environment variables are described first:

  !    required: NUBEAM_ACTION
  !    optional: NUBEAM_WORKPATH  NUBEAM_POSTPROC  NUBEAM_REPEAT_COUNT
  !    optional (thermal neutral gas model): FRANTIC_INIT  FRANTIC_ACTION

  !    NUBEAM_ACTION = "INIT" -- initialize (or initialize and step)
  !                              (random number seed is reset from sys clock).
  !    NUBEAM_ACTION = "INIT_HOLD" -- initialize (or init and step)
  !                              (namelist random number seed is used).

  !    NUBEAM_ACTION = "STEP" -- step (read saved state from prior step first)
  !       (if no such state: error)
  !       (note: in MPI runs, state is distributed across processors)

  !    NUBEAM_ACTION = "BACKUP" -- save backup copies of all NUBEAM state 
  !       files: <filename> -> <filename>_save
  !       (NOTE: at least one STEP needs to have completed before this
  !       can be done safely).

  !    NUBEAM_ACTION = "RETRIEVE" -- retrieve backup copies of all NUBEAM 
  !       state files: <filename>_save -> <filename>
  !       (NOTE: at least one BACKUP needs to have completed before this
  !       can be done safely).

  !    NUBEAM_ACTION = "SKIP" -- set this for FRANTIC-only calculation.

  ! (optional):
  !    NUBEAM_WORKPATH = <path-to-work-directory>
  !       (if undefined: assume current working directory)
  !       NUBEAM files are read/written in the NUBEAM_WORKPATH directory
  !       => because there are a lot of files, it is recommended to set this!

  ! (optional):
  !    NUBEAM_POSTPROC -- post-processing options after completion of 
  !       NUBEAM step (default is "NONE"):
  !          "SUMMARY_TEST" -- execute nbdrive_summary and nbdrive_summary_2d
  !                            routines-- demo access to XPLASMA stored NUBEAM
  !                            data which is more detailed than what is written
  !                            in the plasma state;
  !          "FBM_WRITE" ----- write distribution function data in NetCDF
  !                            file (same contents as is produced by TRANSP
  !                            get_fbm program); filename = <runid>_fbm_r8.cdf
  !                            where <runid> is taken from the Plasma State.
  !          "FBM_WRITE:<filename>" -- as above, but with the output filename
  !                            specified, e.g. "FBM_WRITE:my_fbm.cdf" to have
  !                            the data written to file my_fbm.cdf.
  !          "NONE" -- no post-processing action, but normal output files
  !                            are written.
  !          "NO_OUTPUT" -- suppress output after a time step.

  ! (optional):
  !    NUBEAM_REPEAT_COUNT -- number of times to repeat step 
  !       default:
  !         single state input: 1 * [step size specified in plasma state]
  !         multi state input: [#states] x [avg-time-spacing of states]
  !
  !       An integer value between 1 and 999 means repeat the step 
  !       the indicated number of times; all other values out of range.
  !       An expression of the form 
  !            <integer btw 1 and 999>x<positive floating point value>
  !       means: override the time step dt = ss_in%t1 - ss_in%t0 with
  !       the value provided.
  !          examples (valid):
  !             setenv NUBEAM_REPEAT_COUNT 10 -- 10 steps, dt from plasma state
  !             setenv NUBEAM_REPEAT_COUNT 5x0.050 -- 5 steps of dt = 0.05 sec
  !             setenv NUBEAM_REPEAT_COUNT 100x0.01 -- 100 steps dt = 0.01 sec
  !             setenv NUBEAM_REPEAT_COUNT 100x1.0e-2 -- the same
  !    NOTE: if NUBEAM_REPEAT_COUNT is used with .gt.1 steps, or, if a list
  !    of input files are given, it is necessary to specify an output state 
  !    file.  This prevents an input state from being modified, so that 
  !    multiple steps can be taken against a fixed target plasma (or a
  !    prescribed and unmodified time series.  The output state filename
  !    (which can be given as a full state or a state update) is specified 
  !    in the namelist input file $NUBEAM_WORKPATH/nubeam_step_files.dat, 
  !    described below.
  !
  ! (optional): 
  !    FRANTIC_INIT  (set to integer value: <numzones>)
  !                   -- initialize GAS component in Plasma State:
  !                if ps%nrho_gas = 0 , reset it to 1 + <numzones> decoded 
  !                as integer; e.g. the value "50" --> ps%nrho_gas = 51.  
  !                BUT:
  !                if the state element nrho_gas is already set to a non-
  !                zero value, the prior value is retained and <numzones>
  !                is ignored with just a warning message.  On the other 
  !                hand, if a FRANTIC computation is requested and the
  !                ps%nrho_gas element has not been set, the rule is (a)
  !                set it to match max(ps%nrho_nbi,ps%nrho_fus) if these
  !                are non-zero; (b) if the absence of any other information,
  !                set it to 51, corresponding to 50 radial zones.

  ! (optional): 
  !    FRANTIC_ACTION -- 
  !          "NONE" -- (default) FRANTIC not to be used;
  !          "EXECUTE" -- after NUBEAM calculation, compute neutral gas
  !                    transport due to halo source, recombination source,
  !                    and due to all wall sources (taking updated profile
  !                    of ionization by interaction with fast ion species).
  !          "HALO_ONLY" -- after NUBEAM calculation, update halo source
  !                    driven neutral transport only.
  !          "WALL_ONLY" -- after NUBEAM calculation, update wall source
  !                    driven neutral transport only.
  !          "RECO_ONLY" -- after NUBEAM calculation, update recombination
  !                    driven neutral transport only.
  !          "HALO_RECO" or "RECO_HALO" -- recombination & halo
  !          "WALL_RECO" or "RECO_WALL" -- wall & recombination
  !          "WALL_HALO" or "HALO_WALL" -- wall & halo
  !
  !          (the environment variable value is converted to uppercase;
  !          the parser looks for substrings HALO RECO and WALL).
  ! 
  !    Note that FRANTIC is only called after NUBEAM and does not affect
  !    Plasma State data input to NUBEAM.  By setting NUBEAM_ACTION="SKIP" 
  !    this execution of nubeam_comp_exec can be made to run FRANTIC only,
  !    using data from the input state.
  !
  ! Sample scripts are provided for driving this program:
  !    tftr_test.csh
  !    d3d_test.csh
  !    
  !--------------------------------------
  ! nubeam_comp_exec program procedure:
  !    (a) translate above environment variables;
  !    (b) read namelist NUBEAM_FILES; file path is
  !          $NUBEAM_WORKPATH/nubeam_skip_files.dat   (ISKIP=.TRUE.)  ...or...
  !          $NUBEAM_WORKPATH/nubeam_init_files.dat   (INIT=.TRUE.)   ...or...
  !          $NUBEAM_WORKPATH/nubeam_step_files.dat   (INIT=.FALSE.)  
  !    (c) execute, using files and logic dictated by namelist.
  !
  ! Files are further described below.  Categories of files:
  ! ========================================================
  !   private state -- NUBEAM-specific data retained from prior calls.
  !                   (created or updated on exit)
  !
  !   plasma state -- physics input data: beam geometry, plasma geometry,
  !                   plasma parameters...
  !
  !   plasma state -- physics output data: heating and current drive profiles
  !                   fast ion densities and pressures and particle sources
  !                   (set by execution of this program).  FRANTIC can also
  !                   be invoked to compute thermal source neutral densities
  !                   and associated plasma source/sink profiles.
  !
  !   control files -- NUBEAM control namelist or NetCDF file; a full set 
  !                   of controls can be set at INIT time; only a subset 
  !                   are modifiable over time, i.e. during a STEP execution --
  !                   input data.
  !
  !   ICRF data files (to be combined with plasma state data) --
  !                   information related to ICRF wave fields as can be
  !                   used by NUBEAM RF quasi linear operator (this is
  !                   work in progress, noted DMC July 2008).
  !                   ...This is *optional* input data for NUBEAM.
  !
  !--------------------------------------
  ! Synopsis of calls to NUBEAM library:
  !
  !   if using MPI, set communicator PRIOR to any other NUBEAM call!
  !   (this step can be stepped if the default communicator MPI_COMM_WORLD
  !   is used).

  !      CALL nbi_set_comm(MPI_COMM_WORLD)  ! (example of call)

  !   initialization (begin)
  !      CALL nubeam_ctrl_init(<plasma_state>, <init_data_object>, ierr)

  !   note: the input <plasma_state> is to contain the beam geometry 
  !         and target plasma species lists including neutral species; 
  !         an MHD equilibrium and all profiles -- even if a step is not
  !         to be computed on this call...

  !         <init_data_object> is an instance of the "simple" data type
  !         "nbi_ctrl_init" defined in module nbi_pstypes -- can be 
  !         equivalenced to a C struct.

  !         in this program (nubeam_comp_exec) these objects are loaded
  !         from files that are specified in the namelist; Plasma State
  !         and NUBEAM calls are used to perform these loads.

  !   caution:  nubeam_ctrl_init(...) cannot by itself complete initialization
  !         so that a nubeam private state can be generated.  To be able to
  !         save a step, either nubeam_ctrl_step(...) or 
  !                             nubeam_ctrl_nostep(...) must be called!

  !   step ... initialization completed first if necessary...
  !      CALL nubeam_ctrl_step(<plasma_state_in>, <plasma_state_out>, &
  !                            <step_data_object>, ierr)

  !   note: <plasma_state_in> and <plasma_state_out> can be the same state
  !         instance, in which case the input state is modified on output.
  !         Or, they can be distinct state instances.  The input plasma_state
  !         must include the beam description, equilibrium, and target plasma
  !         profiles including neutral temperatures and densities.

  !         <step_data_object> is an instance of the "simple" data type
  !         "nbi_ctrl_update" defined in module nbi_pstypes -- can be 
  !         equivalenced to a C struct.

  !         in this program (nubeam_comp_exec) these objects are loaded
  !         from files that are specified in the namelist; Plasma State
  !         and NUBEAM calls are used to perform these loads.

  !   nostep... complete initialization, when nubeam_ctrl_step(...) not called:

  !      CALL nubeam_ctrl_nostep(<plasma_state_in>, ierr)
  

  !-----------------------------------------
  !  modules:
  
  use plasma_state_mod  ! plasma state (PS) module definitions & methods
                        ! PS contains both NUBEAM input and output data
                        ! elements

  use nbi_pstypes       ! NUBEAM control data structures & methods

  use nbi_types         ! definition of NUBEAM output data types
                        ! (used only for nbotype_cptimes in this program)
  use logmod  ! A set of utilities for writing to a log file


  implicit NONE

  !-----------------------------------------
  !  local, set in step (a)

  logical :: iskip,init,nostep,iskip_output
  logical :: iwrite_fbm
  logical :: ilseed_reset

  logical :: list_flag   ! set .TRUE. if list of states (time evolution) input
  logical :: echo_error

  character*200 :: save_workpath,workpath,action_value,env_preactdir,env_adasdir
  character*200 :: postproc_value,fbm_filename
  character*40 :: repeat_value
  character*180 :: wk_prefix,wk_suffix
  character*240 :: nubeam_files_namelist
  character*10 :: valstep,valincod,val_list_flag
  character*9 :: len_iblk,len_r8blk
  integer :: ilenv,ilenw,idot,ii,indx,ic,ilenf

  !  DMC: in MPI usage, myid.gt.0 processes to get their own files on unit 6
  character*200 msg_file
  integer :: ilun_msgs = 6

  !  among nubeam namelist inputs:
  integer, parameter :: idefault = -123456789  ! code for "defaulted" integers
  !-------------------------------------------------------------------------
  !  local, set in step (b):  namelist variables follow...

  integer :: ilun_naml = 77
  character*240 :: zfile
  integer :: istat,ierr,iertmp,iwarn,iinit,istep,jstep,icolon,ierr_check
  integer :: ibackup = 0
  integer :: iretrieve = 0
  integer :: irestore = 0

  integer :: istep_ct = 1
  real*8 :: dt_step = 0.0    ! zero means, use the Plasma State data
  real*8 :: t0_save,t1_save  ! save ss_in time values as needed
  real*8 :: ans1(1),vec1(2),fnum,eppav,eplav,zfac,ztprev,ztnext,zdtacc

  integer :: inicrf, jf, jj
  integer :: myfile, mpierr, ichar, ierrnew

  !-------------------------------------------------------------------------
  !  NUBEAM private state data

  character*240 :: private_fld_state   ! NUBEAM's field state file (or blank)
  !  if specified:
  !    $NUBEAM_WORKPATH/trim(private_fld_state) must exist for STEP call 
  !    (master process) -- the field state, data from prior time steps.
  !  if not specified:
  !    $NUBEAM_WORKPATH/<runid>_nbi_fld_state.cdf  must exist and contain
  !    the field state.  <runid> taken from the input plasma state.

  character*240 :: private_ptcl_state
  ! NUBEAM's ptcl state file
  ! as a mandatory
  !    $NUBEAM_WORKPATH/<runid>_nbi_ptcl_state_cpu<myid>_<numprocs>.cdf will contain the
  !    nubeam ptcl state - serial and MPI codes.
  !    <runid> taken from the input plasma state.
  !    there is a possibility to use user specified ptcl. file name (for future use).

  !-------------------------------------------------------------------------
  !  PLASMA STATE data

  integer :: ccselect(ps_ccount)  ! component selector

  character*240 :: input_plasma_state  ! INPUT plasma state file (or blank)
  !  $NUBEAM_WORKPATH/trim(input_plasma_state) ... must be specified and
  !  must exist for ANY call.

  character*240 :: output_plasma_state ! output plasma state file
  !  optional:  if non-blank, $NUBEAM_WORKPATH/trim(output_plasma_state) is
  !  read if it exists; otherwise a skeleton for the object is copied from
  !  the input state; the file is written on program exit.

  character*240 :: plasma_state_update ! output plasma state update file
  !  optional:  if non-blank, a skeleton output state object is created
  !  and modified by NUBEAM calls.  The modifications (only) are written
  !  to thisnamed file.

  !  output_plasma_state and plasma_state_update cannot both be specified.
  !  If either is specified, the state object read from input_plasma_state
  !  is never modified (important for successful use of REPEAT_COUNT).
  !  If neither is specified, $NUBEAM_WORKPATH/trim(input_plasma_state) is
  !  written at the end, with NBI component data modified.

  integer :: inzon_nbi,inzon_gas

  !-------------------------------------------------------------------------
  !  NUBEAM control inputs (nbi_pstypes I/O files)

  !  one of the following on an INIT call

  character*240 :: init_namelist       ! control variables in namelist format
  !  $NUBEAM_WORKPATH/trim(init_namelist)

  character*240 :: init_cdf            ! control variables in NetCDF format
  !  $NUBEAM_WORKPATH/trim(init_cdf)

  !  if both are present, the cdf file is preferred.

  !  one of the following must be present on a STEP call.  On an INIT call,
  !  if one or more of these are present, it means that a step is to be
  !  executed after initialization:

  character*240 :: step_namelist       ! updatable control variables namelist
  !  $NUBEAM_WORKPATH/trim(step_namelist)

  character*240 :: step_cdf            ! updatable control variables NetCDF
  !  $NUBEAM_WORKPATH/trim(step_cdf)

  !  if both are present, the cdf file is preferred.

  integer :: irns   ! integer function to set RNG seed from system clock

  !-------------------------------------------------------------------------
  !  RFXQLO data

  !  ***optional***
  !  if filenames are left blank, then, no RF data.

  !  RF data, in case the Monte Carlo RF Operator is active

  character*240 :: rf_idata   ! file containing data that was input to RF
  !  wave field code; $NUBEAM_WORKPATH/trim(rf_idata)

  character*240 :: rf_odata   ! file containing data that was output of RF
  !  wave field code; $NUBEAM_WORKPATH/trim(rf_idata)

  !-------------------------------------------------------------------------
  !  namelist:

  namelist/NUBEAM_FILES/ &
       input_plasma_state, output_plasma_state, plasma_state_update, &
       private_fld_state, &
       init_namelist, init_cdf, step_namelist, step_cdf, &
       rf_idata,rf_odata

  !-------------------------------------------------------------------------
  !  local, set in step (c) during execution:

  type (plasma_state), pointer :: ss_in  ! input state
  type (plasma_state), pointer :: ss_out ! output state

  type (plasma_state), pointer :: ss_out_base  ! input state, basis for output

  type (plasma_state), dimension(:), pointer :: ss_list
  integer :: nss_list
  real*8, dimension(:), allocatable :: stimes
  logical, dimension(:), allocatable :: sawflags

  real*8, dimension(:), allocatable :: xtimes
  integer, dimension(:), allocatable :: index1_state,index2_state
  integer, dimension(:), allocatable :: index_sawstate
  logical, dimension(:), allocatable :: index1_exact

  logical :: ps_update_flag       ! .TRUE. to write plasma state update file
  logical :: ps_output_flag       ! .TRUE. to write full plasma state file
  logical :: ps_output_any        ! .TRUE. if any plasma state data will be
                                  ! written...

  type(nbi_ctrl_init) :: znbinit  ! control inputs for initialization
  type(nbi_ctrl_update) :: znbup  ! modifiable control inputs for step
 
  type (nbotype_cptimes) :: zcptimes
  integer :: wall1(8),wall2(8)
  real*8 :: walltim
  real*8 :: cpu_minsum,cpu_maxsum,cpu_rootsum
  !
  !-------------------------------------------------------------------------
  !  FRANTIC

  ! input grid size from FRANTIC_INIT (0 if absent) (two copies):
  integer :: inum_frantic

  logical :: ifran_wall ! FRANTIC post-processing => output plasma state:
  !            .TRUE. to compute wall sources.
  logical :: ifran_halo ! FRANTIC post-processing => output plasma state:
  !            .TRUE. to compute halo sources.
  logical :: ifran_reco ! FRANTIC post-processing => output plasma state:
  !            .TRUE. to compute recombination sources.
  
  !-------------------------------------------------------------------------
  !  runid & runid filename
  !  if the state is read, runid_filename is written & contains runid...
  !  if the state is not read, runid_filename is read to get the runid...

  character*32 runid
  character*200 runid_filename

  !--------------------------------------------------------------
  integer mmib
  integer :: nbi_comm
#ifdef __MPI

  include 'mpif.h'

  character*(MPI_MAX_PROCESSOR_NAME) proc_name
  integer iresult,ierror

#endif
  integer :: myid,numprocs
  !--------------------------------------------------------------
  character*180 log_file, logfile_level ! log file prefix for error messages 
  integer :: iost
  !--------------------------------------------------------------

  nullify(ss_in)
  nullify(ss_out)
  nullify(ss_out_base)

  !--------------------------------------------------------------
  !  MPI initialization

#ifdef __MPI
  nbi_comm = MPI_COMM_WORLD
  call nbi_set_comm(nbi_comm)

  call MPI_INIT(mpierr)
  call MPI_COMM_RANK(nbi_comm,myid,mpierr)
  call MPI_COMM_SIZE(nbi_comm,numprocs,mpierr)

  call MPI_GET_PROCESSOR_NAME(proc_name,iresult,ierror)
  write(0,'(a,i5,a,a)') '%nubeam_comp_exec: MPI Mode: myid= ', &
       myid,' proc_name= ',proc_name(1:iresult)
#else
  nbi_comm = -1
  myid=0
  numprocs=1
  write(0,'(a)') '%nubeam_comp_exec: Non-MPI Mode of operation'
#endif

  !-------------------------------------------------------------------------
  !  set namelist variables' default values

  list_flag = .FALSE.
  echo_error = .FALSE.

  input_plasma_state = ' '
  output_plasma_state = ' '
  plasma_state_update = ' '

  private_fld_state = ' '

  init_namelist = ' '
  init_cdf = ' '

  step_namelist = ' '
  step_cdf = ' '

  rf_idata = ' '
  rf_odata = ' '

  !-------------------------------------------------------------------------
  ! additional initializations...

  runid = ' '

  ilseed_reset = .TRUE.
  !-------------------------------------------------------------------------
  ! FRANTIC

  inum_frantic = 0

  ifran_wall = .FALSE.
  ifran_halo = .FALSE.
  ifran_reco = .FALSE.

  if(myid.eq.0) then
     call mpi_sget_env('FRANTIC_INIT',action_value,ierr)
     if(action_value.ne.' ') then
        ilenv=len(trim(action_value))
        if(ilenv.gt.4) then
           write(0,*) ' ?nubeam_comp_exec: FRANTIC_INIT value "'// &
                trim(action_value)//'" too long.'
           call bad_exit
        endif
        valincod=' '
        valincod(10-ilenv+1:10)=action_value(1:ilenv)
        read(valincod,'(I10)',iostat=istat) inum_frantic
        if(istat.ne.0) then
           write(0,*) ' ?nubeam_comp_exec: FRANTIC_INIT value "'// &
                trim(action_value)//'" integer decode failed.'
           call bad_exit
        endif
        if((inum_frantic.lt.10).or.(inum_frantic.gt.2000)) then
           write(0,*) ' ?nubeam_comp_exec: FRANTIC_INIT value "'// &
                trim(action_value)//'" not in range 10 to 2000.'
           call bad_exit
        endif
        write(0,*) ' %nubeam_comp_exec: FRANTIC_INIT=',inum_frantic
     endif
     call mpi_sget_env('FRANTIC_ACTION',action_value,ierr)
     if(action_value.ne.' ') then 
        call uupper(action_value)
        if(action_value.eq.'EXECUTE') then 
           ifran_wall=.TRUE.
           ifran_halo=.TRUE.
           ifran_reco=.TRUE.
        else if(action_value.eq.'NONE') then
           continue  ! no action
        else
           ifran_wall = (index(action_value,'WALL').gt.0)
           ifran_halo = (index(action_value,'HALO').gt.0)
           ifran_reco = (index(action_value,'RECO').gt.0)
           ! at least one must be activated...
           if(.not.(ifran_wall.or.ifran_halo.or.ifran_reco)) then
              write(0,*) ' ?nubeam_comp_exec: FRANTIC_ACTION value "'// &
                   trim(action_value)//'" not recognized.'
              call bad_exit
           endif
        endif
        write(0,*) ' '
        write(0,*) ' %nubeam_comp_exec: FRANTIC_ACTION value: '// &
             trim(action_value)
        if(action_value.ne.'NONE') then
           if(ifran_wall) write(0,*) '  => compute wall neutral transport.'
           if(ifran_halo) write(0,*) '  => compute halo neutral transport.'
           if(ifran_reco) write(0,*) '  => compute recombination neutral transport.'
        endif
     endif
  endif
  !-------------------------------------------------------------------------
  ! Get the environmental variables
  ! NUBEAM_ACTION, NUBEAM_WORKPATH and PREACTDIR, ADASDIR,LOG_LEVEL
  ! read namelist -- root process only...

  if (myid.eq.0) then
     call mpi_sget_env('NUBEAM_ACTION',action_value,ierr)
     call check_nubeam_action  ! set init flag
     call mpi_sset_env('NUBEAM_ACTION',action_value,ierr)

     call mpi_sget_env('NUBEAM_WORKPATH',workpath,ierr)
     call check_workpath
     call mpi_sset_env('NUBEAM_WORKPATH',workpath,ierr)

     call mpi_sget_env('LOG_LEVEL',logfile_level,ierr)
     call check_loglevel
     call mpi_sset_env('LOGFILE_LEVEL',logfile_level,ierr)

     call mpi_sget_env('MAX_MPI_INT_BLOCK',len_iblk,ierr)
     
     call mpi_sget_env('MAX_MPI_R8_BLOCK',len_r8blk,ierr)

     call mpi_sget_env('PREACTDIR',env_preactdir,ierr)
     if(env_preactdir.eq.' ') then
        write(0,*) ' ?nubeam_comp_exec: failed to tr_anslate PREACTDIR environment variable!'
        call bad_exit
     endif
     call mpi_sget_env('ADASDIR',env_adasdir,ierr)
     if(env_adasdir.eq.' ') then
        write(0,*) ' ?nubeam_comp_exec: failed to translate ADASDIR environment variable!'
        call bad_exit
     endif

     call mpi_sget_env('NUBEAM_POSTPROC',postproc_value,ierr)
     call check_nubeam_postproc
     call mpi_sset_env('NUBEAM_POSTPROC',trim(postproc_value),ierr)
     if(postproc_value(1:9).eq.'NO_OUTPUT') iskip_output=.TRUE.

     call mpi_sget_env('NUBEAM_REPEAT_COUNT',repeat_value,ierr)
     if(repeat_value.eq.' ') repeat_value='1'
     if(repeat_value.ne.'1') then
        write(0,*) ' %nubeam_comp_exec(0) NUBEAM_REPEAT_COUNT: '// &
             trim(repeat_value)
        call parse_repeat_value
     endif
     call mpi_sset_env('NUBEAM_REPEAT_COUNT',repeat_value,ierr)

     !----------------------------
     !  read the namelist

     if(iskip) then
        zfile = trim(workpath)//'nubeam_skip_files.dat'
     else if(init) then
        zfile = trim(workpath)//'nubeam_init_files.dat'
     else
        zfile = trim(workpath)//'nubeam_step_files.dat'
     endif

     write(ilun_msgs,*) ' %nubeam_comp_exec: open file: '//trim(zfile)

     open(unit=ilun_naml,file=trim(zfile),status='old',iostat=istat)
     if(istat.ne.0) then
        write(ilun_msgs,*) ' ?nubeam_comp_exec: failed to open: '//trim(zfile)
        call bad_exit
     endif
     !  read (let system report any error & kill job)

     write(ilun_msgs,*) ' %nubeam_comp_exec: read namelist in file:'
  
     READ(unit=ilun_naml,NML=NUBEAM_FILES)

     close(unit=ilun_naml)

     !--------------------------------------
     !  echo; prepend workpath to filenames
     write(ilun_msgs,*) ' '
     write(ilun_msgs,*) '  NUBEAM_FILES namelist contents:'

     write(ilun_msgs,*) ' '
     call echo('input_plasma_state',input_plasma_state)
     call echo('output_plasma_state',output_plasma_state)
     call echo('plasma_state_update',plasma_state_update)

     write(ilun_msgs,*) ' '
     call echo('private_fld_state',private_fld_state)
     
     write(ilun_msgs,*) ' '
     call echo('init_namelist',init_namelist)
     call echo('init_cdf',init_cdf)

     write(ilun_msgs,*) ' '
     call echo('step_namelist',step_namelist)
     call echo('step_cdf',step_cdf)

     write(ilun_msgs,*) ' '
     call echo('RF_idata',rf_idata)
     call echo('RF_odata',rf_odata)
     write(ilun_msgs,*) ' '

     if(echo_error) then
        write(ilun_msgs,*) ' ?nubeam_comp_exec: "echo" pre-processing error.'
        call bad_exit
     endif

     call check_namelist_logic

     ! this is broadcast to indicate whether an INIT call is combined
     ! with combined with a first time step, or not (istep set in the
     ! preceding call to CONTAINED routine check_namelist_logic):

     ! read RUNID from file (myid=0) for BAKUP and RETRIEVE action
     if(ibackup.eq.1.or.iretrieve.eq.1) then
        open(unit=ilun_naml,file=trim(runid_filename),status='OLD',iostat=istat)
        if(istat.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: failed to open: '//trim(runid_filename)
           call bad_exit
        endif
        read(ilun_naml,'(1x,A)') runid
        close(unit=ilun_naml)
        call mpi_sset_env('NUBEAM_RUNID',runid,ierr)
     endif
     
     if(istep.eq.1) then
        call mpi_sset_env('STEPFLAG','TRUE',ierr)
        if(list_flag) then
           call mpi_sset_env('LIST_FLAG','TRUE',ierr)
           call get_state_list(ierr)
           if(ierr.ne.0) then
              write(ilun_msgs,*) ' ?nubeam_comp_exec: state list acquisition error.'
              call bad_exit
           endif
       else
          call mpi_sset_env('LIST_FLAG','FALSE',ierr)
       endif
     else
        call mpi_sset_env('STEPFLAG','FALSE',ierr)
        call mpi_sset_env('LIST_FLAG','FALSE',ierr)
     endif

  endif
  
  !-------------------------------------------------------------------------
  !-------------------------------------------------------------------------
  ! portlib -- mpi_share_env
  ! this uses MPI_BCAST to broadcast the current working directory
  ! and environment variables; all child processes cd to 
  ! current working directory also (at PPPL these are usually directories on
  ! local disks, so, children on different nodes see different directories).

  call mpi_share_env(myid,ierr)
  if(ierr.ne.0) then
     write(0,*) ' ?nubeam_comp_exec: myid=',myid,' -- mpi_share_env error.'
     call bad_exit
  endif

  if (myid.gt.0) then
     
     call mpi_sget_env('NUBEAM_ACTION',action_value,ierr)
     call mpi_sget_env('NUBEAM_WORKPATH',workpath,ierr)
     call mpi_sget_env('LOGFILE_LEVEL',logfile_level,ierr)
     call mpi_sget_env('PREACTDIR', env_preactdir,ierr)
     call mpi_sget_env('ADASDIR', env_adasdir,ierr)
     if(workpath.ne.'./') then
        call gmkdir(' ',trim(workpath),ierr)
        if(ierr.ne.0) then
           write(0,*) ' ?nubeam_comp_exec: myid=',myid, &
                ' -- gmkdir(workpath) error.'
           call bad_exit
        endif
     endif

     call mpi_sget_env('NUBEAM_POSTPROC',postproc_value,ierr)
     call mpi_sget_env('NUBEAM_REPEAT_COUNT',repeat_value,ierr)
     call parse_repeat_value
     call mpi_sget_env('STEPFLAG',valstep,ierr)
     if(valstep.eq.'TRUE') then
        istep=1      ! STEP action, or, 2nd part of INIT action
     else if(valstep.eq.'FALSE') then
        istep=0
     else

        write(0,*) ' ?nubeam_comp_exec: STEPFLAG value not "TRUE" or "FALSE".'
        write(0,*) '  error in process myid = ',myid
        call bad_exit

     endif
     call mpi_sget_env('LIST_FLAG',val_list_flag,ierr)
     if(val_list_flag.eq.'TRUE') then
        list_flag=.TRUE.
     else if(val_list_flag.eq.'FALSE') then
        list_flag=.FALSE.
     else
        
        write(0,*) ' ?nubeam_comp_exec: LIST_FLAG value not "TRUE" or "FALSE".'
        write(0,*) '  error in process myid = ',myid
        call bad_exit
        
     endif
     
  endif
  call ulower(logfile_level)


  ! open separate message output file for each processor...
  ! select separate name for ptcl state for each processor...

  call uupper(action_value)


  if(myid.gt.0) then

     call mknam('nubeam_comp','msgs',msg_file)

     write(0,*) ' myid=',myid,'  msg I/O on unit ',ilun_msgs,' written to: ', &
          trim(msg_file)

     if(action_value.eq.'STEP') then
        !  append if a step call
        open(unit=ilun_msgs,file=trim(msg_file),status='unknown', &
             position='append',iostat=istat)
     else
        !  otherwise start fresh
        open(unit=ilun_msgs,file=trim(msg_file),status='unknown', &
             iostat=istat)
     endif
     if(istat.ne.0) then
        write(0,*) ' ?nubeam_comp_exec: myid=',myid,' open failure: ', &
             trim(msg_file)
        call bad_exit
     endif
     if(action_value.eq.'BACKUP') ibackup=1
     if(action_value.eq.'RETRIEVE') iretrieve=1
     if(ibackup.eq.1.or.iretrieve.eq.1) then
        call mpi_sget_env('NUBEAM_RUNID',runid,ierr) ! set 'NUBEAM_RUNID' for myid>0
        
     endif
          
  endif

  ! echo the environment, all procs...

  call mpi_printenv(myid, ilun_msgs)

  !------------------------------------------------------------
  !  INIT or STEP ...?
  !  all processes...

  if(myid.gt.0) then
     call check_nubeam_action  ! looks at action_value; sets INIT
  endif

  !  restore state -- all processes

  if((.not.init).and.(.not.iskip).and.(ibackup.eq.0).and.(iretrieve.eq.0)) then
     ! read NUBEAM-private state
     ! root processor broadcasts field state; all processes read their own
     !   separate particle states

     call chk_runid

     if(myid.eq.0) then 
        call mark_nbi_states("R")
        call nbi_fld_state( trim(private_fld_state),'R',ierr)
        if(ierr.ne.0) then
           write(ilun_msgs,*) &
                ' ?nubeam_comp_exec: NUBEAM private state restoration failure.'
           call bad_exit
        endif
        irestore=1
     endif
  endif

  !  timestep modification -- if list of states are input...
  if(istep.eq.1) then
     if(list_flag) then

        if(myid.eq.0) then
           call get_exec_times(ierr)
           if(ierr.ne.0) then
              write(ilun_msgs,*) &
                   ' ?nubeam_comp_exec: get_exec_times(...) failure.'
              call bad_exit
           endif
        endif

#ifdef __MPI
        call MPI_BCAST(istep_ct,1,MPI_INTEGER,0,nbi_comm,mpierr)
#endif
     endif
  endif

  !----------------------------
  !  set up input and output states
  !  NOTE: if time series of states are input, "get_exec_times" has a hand
  !  in this...

  if((myid.eq.0).and.(ibackup.eq.0).and.(iretrieve.eq.0)) then
     !=========================================================================
     !--------------------------------------
     !  read plasma state file(s)
     !  only the root process does this
     !  if there are multiple input states (ss_in) is already associated...

     if(.not.associated(ss_in)) then
        !  read input state, hold in (psp) state instance...
        call ps_get_plasma_state(ierr,filename=trim(input_plasma_state), &
             state=psp)
        if(ierr.ne.0) then
           write(ilun_msgs,*) &
                ' ?nubeam_comp_exec: input_plasma_state read failed.'
           call bad_exit
        else
           ss_in => psp
           ss_out_base => psp
        endif
     else
        !  ss_out_base also associated (see get_exec_times)
        continue
     endif

     runid = ss_in%runid
     if(runid.eq.' ') then
        write(ilun_msgs,*) &
             ' ?nubeam_comp_exec: input plasma state RUNID data element blank!'
        call bad_exit
     endif

     ! structure to select NBI, FUS, and GAS components in Plasma State
     ccselect = ps_cczero
     ccselect(ps_cnum_NBI)=1
     ccselect(ps_cnum_FUS)=1
     if(ifran_halo.or.ifran_wall.or.ifran_reco) then
        ccselect(ps_cnum_GAS)=1
     endif

     if(ps_output_flag) then
        !  hold output state in plasma_state_mode instance (aux)
        write(ilun_msgs,*) ' %nubeam_comp_exec: try file read (success not required):'
        call ps_get_plasma_state(ierr,filename=trim(output_plasma_state), &
             state=aux)
        if(ierr.ne.0) then
           write(ilun_msgs,*) &
                ' %nubeam_comp_exec: output_plasma_state read failed.'
           write(ilun_msgs,*) &
                '  SO: create output_plasma_state by copying input state.'

           ! only select data expected for output
           call ps_copy_plasma_state(ss_out_base,aux,ierr, &
                cclist=ccselect, iwant_1deq=1)
           if(ierr.ne.0) then
              call errmsg_exit(' ?nubeam_comp_exec: output ps_copy_plasma_state failure.')
           endif
           if(.not.iskip) then
              call ps_clear_profs(ierr,state=aux,cclist=ccselect, &
                   save_hash=.TRUE.)
           endif
        endif
        ss_out => aux

     else if(ps_update_flag) then
        ! only select data expected for output -- hold in (aux)
        call ps_copy_plasma_state(ss_out_base,aux,ierr, &
             cclist=ccselect, iwant_1deq=1)
        if(ierr.ne.0) then
           call errmsg_exit(' ?nubeam_comp_exec: update ps_copy_plasma_state failure.')
        endif
        if(.not.iskip) then
           call ps_clear_profs(ierr,state=aux,cclist=ccselect, &
                save_hash=.TRUE.)
        endif
        ss_out => aux
        call ps_save_hash_codes(iwarn,state=ss_out)

     else
        if(iskip_output) then
           nullify(ss_out)  ! (no output)
        else
           ! the (single) input state is modified and written back out
           ss_out => psp
        endif
     endif

     write(ilun_msgs,*) ' %nubeam_comp_exec: state file(s) read OK.'

#ifdef _JONGKYU_PARK
     call ps_xplasma_write 
#endif

     !--------------------------------------
     !  read RF data -- this is optional...
     !
     !if input data not found, assume there is no output; but, if input
     !data is found, output data must also be found.
     !(only the root process does this)
     !(There are no mechanisms for broadcasting this data yet)

     CALL rfxqlo_free(iwarn)

     inicrf = ss_in%nicrf_src
     if((.not.iskip).and.(inicrf.gt.0).and.(rf_idata.ne.' ')) then

        call rfxqlo_recover(rf_idata,rf_odata,ss_in,ierr)
        if(ierr.ne.0) then
           write(ilun_msgs,*) &
                ' ?nubeam_comp_exec: rfxqlo data recovery error.'
           call bad_exit
        endif
     endif

  endif

  call chk_runid  ! make sure we have RUNID

  iwrite_fbm=.FALSE.
  if((istep.eq.1).and.(myid.eq.0)) then
     if(postproc_value(1:9).eq.'FBM_WRITE') then
        iwrite_fbm=.TRUE.

        write(ilun_msgs,*) ' ' 

        icolon=index(postproc_value,':')
        if(icolon.le.0) then
           fbm_filename = trim(workpath)//trim(ss_in%runid)//'_fbm_data.cdf'
           write(ilun_msgs,*) ' FBM_WRITE: (default filename)'
        else
           fbm_filename = trim(workpath)//adjustl(trim(postproc_value(icolon+1:)))
        endif
        write(ilun_msgs,*) ' FBM_WRITE: '//trim(fbm_filename)
        if(ss_out%nspec_alla.gt.0) then
           do ii=1,ss_out%nspec_beam
              indx = ss_out%snbi_to_alla(ii)
              ss_out%dist_fun(indx) = trim(fbm_filename)
           enddo
           do ii=1,ss_out%nspec_fusion
              indx = ss_out%sfus_to_alla(ii)
              ss_out%dist_fun(indx) = trim(fbm_filename)
           enddo
        endif

     endif
  endif

  !--------------------------------------
  !  sanity checks

  if((myid.eq.0).and.(ps_output_any)) then
     if (min(ss_in%nrho_nbi,ss_out%nrho_nbi).gt.0) then
        if (ss_in%nrho_nbi .ne. ss_out%nrho_nbi ) then
           write(0,*) ' input and output state inconsistency in nrho_nbi:'
           write(0,*) '   ss_in%nrho_nbi = ',ss_in%nrho_nbi
           write(0,*) '   ss_out%nrho_nbi= ',ss_out%nrho_nbi
           write(0,*) ' these must match!'
           call bad_exit
        endif
     endif
     if (min(ss_in%nrho_fus,ss_out%nrho_fus).gt.0) then
        if (ss_in%nrho_fus .ne. ss_out%nrho_fus ) then
           write(0,*) ' input and output state inconsistency in nrho_fus:'
           write(0,*) '   ss_in%nrho_fus = ',ss_in%nrho_fus
           write(0,*) '   ss_out%nrho_fus= ',ss_out%nrho_fus
           write(0,*) ' these must match!'
           call bad_exit
        endif
     endif
  endif

  !--------------------------------------
  !  initialization (if indicated)

  if(init) then

     ! Only the root processor needs carry out the initialization step

     if (myid.eq.0) then

        if(init_cdf.ne.' ') then
           call nbi_ctrl_init_cdfread(init_cdf, znbinit, ierr)
           if(ierr.ne.0) then
              write(ilun_msgs,*) ' ?nubeam_comp_exec: init_cdf read failed.'
              call bad_exit
           else
              write(ilun_msgs,*) ' %nubeam_comp_exec: init_cdf read OK.'
           endif

        else
           call nbi_ctrl_init_read(init_namelist, znbinit, ierr)
           if(ierr.ne.0) then
              write(ilun_msgs,*) ' ?nubeam_comp_exec: init_namelist read failed.'
              call bad_exit
           else
              write(ilun_msgs,*) ' %nubeam_comp_exec: init_namelist read OK.'
           endif
        endif

        !-------------------------------------------
        ! if the NBI and or FUSion ion grids are already set:
        ! force NUBEAM to match...

        if(max(ss_in%nrho_nbi,ss_in%nrho_fus).gt.0) then
           write(ilun_msgs,*) '  setting NZONES to match Plasma State: '// &
                ' ss_in%nrho_nbi - 1 = ',ss_in%nrho_nbi - 1
           write(ilun_msgs,*) ' '

           znbinit%nzones = ss_in%nrho_nbi - 1
        endif

        !-------------------------------------------
        ! do not permit "only_io" mode here -- this
        ! could be a relic from an input namelist generated
        ! during a TRANSP runs

        if(znbinit%only_io.eq.1) then
           write(ilun_msgs,*) '  %nubeam_comp_exec: "only_io" mode disabled.'
           znbinit%only_io = 0
        endif

        !-------------------------------------------
        ! reset RNG seed

        if(ilseed_reset) then
           znbinit%nseed = irns(0)
           write(ilun_msgs,*) ' '
           write(ilun_msgs,*) '  ---> RNG seed reset to: ',znbinit%nseed
           write(ilun_msgs,*) ' '
        else
           write(ilun_msgs,*) ' '
           write(ilun_msgs,*) '  ---> used namelist RNG seed: ',znbinit%nseed
           write(ilun_msgs,*) ' '
        endif

        !-------------------------------------------
        ! do not allow dt_acc > 1.0d-4

        zdtacc = 1.0d-4
        if(znbinit%dt_acc .gt. zdtacc) then
           write(ilun_msgs,*) ' '
           write(ilun_msgs,*) '  ---> nubeam_comp_exec enforces dt_acc .le. 1.0e-4'
           write(ilun_msgs,*) ' '
           znbinit%dt_acc = zdtacc
        endif

        !-------------------------------------------
        ! and apply...

        write(ilun_msgs,*) ' '
        write(ilun_msgs,*) '  executing NUBEAM initialization:'
        write(ilun_msgs,*) ' '

        call nubeam_ctrl_init(ss_in, znbinit, ierr)

        if(ierr.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: nubeam_ctrl_init failure.'
           call bad_exit
        else
           write(ilun_msgs,*) ' %nubeam_comp_exec: nubeam_ctrl_init OK.'
        endif

     endif

     !-------------------------------------------

  endif

  !--------------------------------------
  !  modify workpath to match requested value

  if(.not.iskip) then
     if(irestore.eq.1) then
        call nubeam_workpath_retrieve(save_workpath)  ! save value
     else
        save_workpath = workpath
     endif

     call nubeam_workpath_reset(workpath)
  endif

  !--------------------------------------
  !  step (if indicated)

  nostep = .TRUE.

  ! open file for on cpu  trim(workpath)//'nubeam_comp_<muid>.log
  ! to load error messages  instead of write(0,*) 
  
  log_file=' '
  write(log_file,'(a)') trim(workpath)//'nubeam_comp.log'

  if(istep.gt.0) then

     nostep = .FALSE.

     ! Only the root reads the step data

     if (myid.eq.0) then
        if(step_cdf.ne.' ') then
           call nbi_ctrl_update_cdfread(step_cdf, znbup, ierr)
           if(ierr.ne.0) then
              write(ilun_msgs,*) ' ?nubeam_comp_exec: step_cdf read failed.'
              call bad_exit
           else
              write(ilun_msgs,*) ' %nubeam_comp_exec: step_cdf read OK.'
           endif

        else
           call nbi_ctrl_update_read(step_namelist, znbup, ierr)
           if(ierr.ne.0) then
              write(ilun_msgs,*) &
                   ' ?nubeam_comp_exec: step_namelist read failed.'
              call bad_exit
           else
              write(ilun_msgs,*) ' %nubeam_comp_exec: step_namelist read OK.'
           endif
        endif
     endif


     if((myid.eq.0).AND.(.not.list_flag)) then

        if((dt_step.eq.0.0d0).and.(ss_in%t0.eq.ss_in%t1)) then
           write(0,*) ' '
           write(0,*) ' %nubeam_comp_exec: no time step in STATE and'
           write(0,*) '  none specified in NUBEAM_REPEAT_COUNT =>'
           write(0,*) '  => use default time step: 0.01s'
           write(0,*) ' '
           dt_step = 0.01d0
        endif

        t1_save = ss_in%t1   ! save t1
        if(dt_step.gt.0.0d0) then
           ss_in%t1 = ss_in%t0 + dt_step
        endif
        if((ss_in%t1 - ss_in%t0).lt.1.0d-6) then
           write(0,*) ' ?nubeam_comp_exec: no time step in Plasma State:'
           write(0,*) '    ss_in%t0 = ',ss_in%t0
           write(0,*) '    ss_in%t1 = ',ss_in%t1
           write(0,*) '  timestep > 1 microsecond expected; 0.01s is normal.'
           write(0,*) '  NOTE: override Plasma State time step by setting'
           write(0,*) '  environment variable NUBEAM_REPEAT_COUNT.'
           write(0,*) '  For example: setenv NUBEAM_REPEAT_COUNT 1x0.025 ...'
           write(0,*) '    for a single 0.025s step;'
           write(0,*) '  Or, setenv NUBEAM_REPEAT_COUNT 5x0.010 ...'
           write(0,*) '    for five 0.01s steps.'
           call bad_exit
        endif
     endif

     call date_and_time(VALUES=wall1)
     cpu_minsum = 0.0d0
     cpu_maxsum = 0.0d0
     cpu_rootsum = 0.0d0

     vec1(1)=0.0d0
     vec1(2)=1.0d0

     call nubeam_gpu_init(ierr)

     do jstep = 1, istep_ct
        if(trim(logfile_level).ne.'nomsg') then
           call openLog(file=trim(log_file),status='RENAME',iostat=iost,comm_world=nbi_comm)
        endif
        call setLogLevel(trim(logfile_level)) ! log file for error messages
        write(ilun_msgs,*) ' '
        write(ilun_msgs,*) '  executing NUBEAM step:',jstep
        if(myid.eq.0) write(0,*) '  executing NUBEAM step:',jstep
        

        if(myid.eq.0) then
           if(list_flag.AND.(jstep.gt.1)) then
              if(index1_exact(jstep)) then
                 ss_in => ss_list(index1_state(jstep))
              else
                 !  time interpolated state (held in psp state instance).
                 ztprev = stimes(index1_state(jstep))
                 ztnext = stimes(index2_state(jstep))
                 zfac = (ztnext-xtimes(jstep))/(ztnext-ztprev)
                 call chk_alloc(ss_list(index1_state(jstep)),psp)
                 call ps_merge_plasma_state(zfac, &
                      ss_list(index1_state(jstep)), &
                      ss_list(index2_state(jstep)), ierr, &
                      new_state = psp, icheck=ps_ignore)
                 ss_in => psp
                 if(ierr.ne.0) then 
                    call errmsg_exit( &
                         ' ?nubeam_comp_exec: nth state merge error.')
                 endif
              endif
           endif

           if(list_flag) then
              t0_save = ss_in%t0
              t1_save = ss_in%t1
              ss_in%t0 = xtimes(jstep)
              ss_in%t1 = xtimes(jstep+1)
              dt_step = xtimes(jstep+1)-xtimes(jstep)
           endif

           ! check whether to activate neutral track data capture and lost orbit data
           if(iwrite_fbm.and.(jstep.eq.istep_ct)) then
              znbup%nltrk_dep0 = 1
              znbup%nlbout = 1

              !  DMC: adjusted here: if both ndep_set controls are defaulted, the error checks
              !       are unnecessary, so, avoid potentially confusing warning messages...
              !       6/2009 -- "idefault" value means "use default".

              if((znbup%ndep_set_beam.ne.idefault).or.(znbup%ndep_set_ien.ne.idefault)) then
                 call nubeam_mib_retrieve(mmib) !retrieve number of beams
                 call check_beam_set(znbup%ndep_set_beam,znbup%ndep_set_ien,mmib,ierr_check)
                 if (ierr_check.ne.0) then
                    ! user settings for beam are incorrect, assuming all beams and all energy fraction
                    write(ilun_msgs,1002) znbup%ndep_set_beam,znbup%ndep_set_ien
1002                format(/,' ?check_beam_set: user settings for beam are incorrect'/&
                         &'   track data will be collected for ndep_set_beam =',i3,/&
                         &'   and for energy fraction ndep_set_ien =',i1)
                 endif
                 if(znbup%ndep_set_beam > 0.and.znbup%ndep_set_ien > 0) then !check if there is any 
                    !beam current for a given
                    !energy fraction
                    call check_beam_energy(znbup%ndep_set_beam,znbup%ndep_set_ien,ierr_check)
                    if(ierr_check.gt.0) then !there is no beam energy for a given beam source
                       !collect track data for all beam sources
                       write(ilun_msgs,1003) znbup%ndep_set_beam,znbup%ndep_set_ien
1003                   format(/,' ?check_beam_energy: there is no beam energy for a given beam'/&
                            &'   track data will be collected for ndep_set_beam =',i3,/&
                            &'   and for energy fraction ndep_set_ien =',i1)
                    endif
                 endif
              endif
           endif
           ! control & execute step
           call nubeam_ctrl_step(ss_in, ss_out, znbup, ierr)

           call nbo_get_cptimes(zcptimes)
           cpu_minsum = cpu_minsum + zcptimes%cpbmin
           cpu_maxsum = cpu_maxsum + zcptimes%cpbmax
           cpu_rootsum= cpu_rootsum+ zcptimes%cpbroot

        else
           ! MPI job -- help execute step
          
           call nubeam_ctrl_step_child(ierr)
        
        endif

        if(ierr.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: nubeam_ctrl_step failure.'
           call bad_exit
        endif
        if(myid.eq.0) then
           if(list_flag) then
              ss_in%t0 = t0_save
              ss_in%t1 = t1_save
           endif

           call ps_state_memory_update(ierr, state=ss_out, eqcheck=.FALSE.)
           write(ilun_msgs,*) ' ---------------------------------------'
           write(ilun_msgs,*) ' %nubeam_comp_exec: nubeam_ctrl_step OK:'// &
                ' N(ptcls) & <<E>>(KeV):'
           do ii=1,ss_out%nspec_beam
              call ps_rho_rezone(vec1,ss_out%id_nbeami(ii),ans1,iertmp, &
                   nonorm=.TRUE.,state=ss_out)
              fnum=ans1(1)
              call ps_rho_rezone(vec1,ss_out%id_eperp_beami(ii),ans1,iertmp, &
                   state=ss_out)
              eppav=ans1(1)
              call ps_rho_rezone(vec1,ss_out%id_epll_beami(ii),ans1,iertmp, &
                   state=ss_out)
              eplav=ans1(1)
              write(ilun_msgs,'(1x,a,3(a,1pe11.4))') 'beam ions ', &
                   trim(ss_out%snbi_name(ii))//': N=',fnum, &
                   ' <<Eperp>>=',eppav,' <<Epll>>=',eplav
           enddo
           do ii=1,ss_out%nspec_fusion
              call ps_rho_rezone(vec1,ss_out%id_nfusi(ii),ans1,iertmp, &
                   nonorm=.TRUE.,state=ss_out)
              fnum=ans1(1)
              call ps_rho_rezone(vec1,ss_out%id_eperp_fusi(ii),ans1,iertmp, &
                   state=ss_out)
              eppav=ans1(1)
              call ps_rho_rezone(vec1,ss_out%id_epll_fusi(ii),ans1,iertmp, &
                   state=ss_out)
              eplav=ans1(1)
              write(ilun_msgs,'(1x,a,3(a,1pe11.4))') 'fusion ions ', &
                   trim(ss_out%sfus_name(ii))//': N=',fnum, &
                   ' <<Eperp>>=',eppav,' <<Epll>>=',eplav
           enddo
           write(ilun_msgs,*) ' ---------------------------------------'
        endif
     enddo

     call date_and_time(VALUES=wall2)
     call wclock_diff_r8(wall1,wall2,walltim)

		! Reset the GPU to finish this call.
     call nubeam_gpu_cleanup(ierr)

     if((myid.eq.0).AND.(.not.list_flag)) then
        ss_in%t1 = t1_save   ! restore t1
     endif

     ! all procs see this option; only has affect if time steps were done:
     if(postproc_value(1:9).eq.'NO_OUTPUT') iskip_output=.TRUE.

     !-------------------------------------------

  else if(init) then

     ! on initialization call with no step, further processing is needed
     ! to establish a NUBEAM state from which a later step can be initiated

     ! only myid.eq.0 need do this...

     if(myid.eq.0) then

        if(trim(logfile_level).ne.'nomsg') then
           call openLog(file=trim(log_file),status='RENAME',iostat=iost,comm_world=nbi_comm)
        endif
        call setLogLevel(trim(logfile_level)) ! log file for error messages
        
        write(ilun_msgs,*) ' '
        write(ilun_msgs,*) '  complete NUBEAM initialization: call nubeam_ctrl_nostep(...)'

        call nubeam_ctrl_nostep(ss_in, ierr)
        if(ierr.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: nubeam_ctrl_nostep failure.'
           call bad_exit
        else
           write(ilun_msgs,*) ' %nubeam_comp_exec: nubeam_ctrl_nostep OK.'
        endif

        call nubeam_chk_state_grids(ss_out, ierr)
        if(ierr.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: nubeam_chk_state_grids failure.'
           call bad_exit
        else
           write(ilun_msgs,*) ' %nubeam_comp_exec: nubeam_chk_state_grids OK.'
        endif
     endif
     
  endif

  if ((myid.eq.0).and.(.NOT.nostep)) then

     !--------------------------------------
     !  print summary information;
     !  write plasma state file & addl outputs



     write(ilun_msgs,*) ' -------------------------------------------- '
     write(ilun_msgs,*) ' summary output: '
     write(ilun_msgs,*) ' '
     write(ilun_msgs,*) ' nubeam_ctrl_step timer results:'
     write(ilun_msgs,'(a,1pe11.4,a)') ' wallclock time ',walltim,' seconds.'
     write(ilun_msgs,'(a,1pe11.4,a)') ' root thread cpu',cpu_rootsum, &
          ' seconds.'
     write(ilun_msgs,'(a,1pe11.4,a)') ' min thread cpu ',cpu_minsum, &
          ' seconds.'
     write(ilun_msgs,'(a,1pe11.4,a)') ' max thread cpu ',cpu_maxsum, &
          ' seconds.'
     write(ilun_msgs,*) ' '
     write(ilun_msgs,*) ' heating: '
     write(ilun_msgs,*) ' zone   Pbe          Pbi          Pbth   (watts/zone)'

     do ii = 1, ss_out%nrho_nbi - 1
        write(ilun_msgs,'(2x,i3,3(2x,1pe11.4))') &
             ii, ss_out%pbe(ii), ss_out%pbi(ii), ss_out%pbth(ii)
     enddo

  endif

  if((myid.eq.0).and.(ibackup.eq.0).and.(iretrieve.eq.0)) then

     !  make sure the state grids are initialized for:  NBI FUS GAS
     !  if (iskip) only initialize the GAS grid.

     inzon_nbi = 0  ! for now.  Zero means, read from existing state or NUBEAM

     call frantic_init_check
     inzon_gas = inum_frantic

     call nubeam_comp_ps_ini(ss_out,ilun_msgs,(.not.iskip), &
          (ifran_wall.or.ifran_halo.or.ifran_reco), &
          inzon_nbi,inzon_gas)

     !  Now perform FRANTIC computations as indicated...

     if(ifran_wall.or.ifran_halo.or.ifran_reco) then
        call ps_state_memory_update(ierr,state=ss_out,eqcheck=.FALSE.)
        if(ierr.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: ps_memory_update before FRANTIC.'
           call bad_exit
        endif

        call ps_nbi_frantic(ss_in,ss_out,ilun_msgs, &
             ifran_wall,ifran_halo,ifran_reco, &
             workpath,ierr)
        if(ierr.ne.0) then
           write(ilun_msgs,*) ' ?nubeam_comp_exec: FRANTIC error.'
           call bad_exit
        endif
     endif
        
     if(output_plasma_state.eq.' ') output_plasma_state = input_plasma_state
     if(ps_output_any) then
        if(ps_update_flag) then
           !  filename is plasma_state_update(:)
           call ps_restore_hash_codes(ierr, state=ss_out)
           call ps_write_update_file(trim(plasma_state_update), ierr, &
                state = ss_out)
        else
           call ps_store_plasma_state(ierr, state = ss_out, eqcheck = .FALSE., &
                filename=trim(output_plasma_state))
        endif
     endif

     if(ierr.ne.0) then
        write(ilun_msgs,*) ' ?nubeam_comp_exec: plasma state output failure.'
        call bad_exit
     endif
     
     ! addl outputs

     ierr=0
     if((.NOT.nostep).AND.(.NOT.iskip_output)) then
        call nbo_output_all(ierr)
        if(ierr.ne.0) then
           call errmsg_exit('?nubeam_comp_exec: NUBEAM file output failed.')
        endif
     endif

  endif

  !--------------------------------------
  !  write NUBEAM internal state files
  !  all processes participate in this...

  ierr=0
  private_ptcl_state='_nbi_ptcl_state_cpu' ! suffix for ptcl list file
                                           ! need to be consistent with
                                           ! suffix in nubeam_comp_exe/nubeam_comp_exec.f90
                                           ! and outcor/wrstrt.for
                                           ! and trcore/resume.for
                                           ! and trmpi/trmpi_listener.f90
                                           ! and nubeam/nubeam_step.f90
  if(ibackup.eq.1) then
     call mark_nbi_states("B")
     call nubeam_runid_set(runid)
     call nbi_states(trim(private_fld_state),trim(private_ptcl_state), &
          'B',0,ierr)
  else if(iretrieve.eq.1) then
     call nubeam_runid_set(runid)
     call mark_nbi_states("C")
     call nbi_states(trim(private_fld_state),trim(private_ptcl_state), &
          'C',0,ierr)
  else if((.not.iskip).AND.(.not.iskip_output)) then
     ! state save after INIT or STEP...
     call mark_nbi_states("S")
     call nbi_states(trim(private_fld_state),trim(private_ptcl_state), &
          'S',0,ierr)


  endif

  if(ierr.ne.0) then
     write(ilun_msgs,*) &
          ' ?nubeam_comp_exec: NUBEAM private state file operation failure.'
     write(0,*) ' myid=',myid,' ACTION=',trim(action_value),' State file I/O operation failed.'
     call bad_exit
  endif

  if((istep.gt.0).and.(myid.eq.0)) then
     ! implement postprocessing...

     if(trim(postproc_value).eq.'SUMMARY_TEST') then

        write(ilun_msgs,*) ' ' 
        write(ilun_msgs,*) ' SUMMARY_TEST output: '
        write(ilun_msgs,*) ' ' 

        ! restore time step information if necessary

        t1_save = ss_in%t1   ! save t1
        if(dt_step.gt.0.0d0) then
           ss_in%t1 = ss_in%t0 + dt_step
        endif

        call nbdrive_load(ss_in,ierr)
        if(ierr.eq.0) then
           call nbdrive_summary(ierr)
        endif

        ss_in%t1 = t1_save   ! restore t1

        if(ierr.ne.0) call errmsg_exit( &
             ' ?nubeam_comp_exec: SUMMARY_TEST post-processing error.')

     else if(postproc_value(1:9).eq.'FBM_WRITE') then

        write(ilun_msgs,*) ' ' 

        ! restore time step information if necessary

        t1_save = ss_in%t1   ! save t1
        if(dt_step.gt.0.0d0) then
           ss_in%t1 = ss_in%t0 + dt_step
        endif

        call nbdrive_load(ss_in,ierr)
        if(ierr.eq.0) then
           call nbdrive_fbm_write(fbm_filename,ierr)
        endif

        ss_in%t1 = t1_save   ! restore t1

        if(ierr.ne.0) call errmsg_exit( &
             ' ?nubeam_comp_exec: FBM_WRITE post-processing error.')

     endif
  endif

  call fort_flush(ilun_msgs)
  if(myid.gt.0) close(unit=ilun_msgs)

#ifdef __MPI
  call MPI_FINALIZE(mpierr)
#endif


CONTAINS

  subroutine chk_runid

    if(myid.eq.0) then
       ! if RUNID has not been set from the plasma state: read it from file
       if(runid.eq.' ') then
          open(unit=ilun_naml,file=trim(runid_filename),status='OLD')
          read(ilun_naml,'(1x,A)') runid
          close(unit=ilun_naml)
       else
          open(unit=ilun_naml,file=trim(runid_filename),status='UNKNOWN')
          write(ilun_naml,'(1x,A)') runid
          close(unit=ilun_naml)
       endif
       call default_filename(private_fld_state,'_nbi_fld_state.cdf')
    endif

  end subroutine chk_runid

  subroutine mknam(prefix,suffix,filenam)

    character*(*), intent(in) :: prefix
    character*(*), intent(in) :: suffix

    character*(*), intent(out) :: filenam

    ! form filename: <workpath/><prefix>_<myid>.<suffix>

    character*240 :: full_prefix

    !--------------------------------------

    full_prefix = trim(workpath)//trim(prefix)//'_'

    if(myid.le.9) then
       write(filenam,'(a,i1,".",a)') trim(full_prefix),myid,trim(suffix)
    else if(myid.le.99) then
       write(filenam,'(a,i2,".",a)') trim(full_prefix),myid,trim(suffix)
    else if(myid.le.999) then
       write(filenam,'(a,i3,".",a)') trim(full_prefix),myid,trim(suffix)
    else if(myid.le.9999) then
       write(filenam,'(a,i4,".",a)') trim(full_prefix),myid,trim(suffix)
    else if(myid.le.99999) then
       write(filenam,'(a,i5,".",a)') trim(full_prefix),myid,trim(suffix)
    endif

  end subroutine mknam

  subroutine echo(varnam,value)

    character*(*), intent(in) :: varnam
    character*(*), intent(inout) :: value   ! prepend workpath if necessary

    !  echo namelist variable (filename) value
    !  append ".data" if value is not blank but no filename .ext is present
    !  prepend workpath

    !  DMC Apr 2009 -- process list option for input_plasma_state

    !--------------------------
    ! local:
    character*140 zfile,zline
    character*280 :: zcmd
    character*40 zkey
    character*5 :: ztest5
    integer :: jsystem
    real*8 :: ztime
    !--------------------------
    
    ilenf = len(trim(value))
    write(ilun_msgs,*) '     '//trim(varnam)//' = "'//value(1:ilenf)//'"'
    if(value.eq.' ') return

    ztest5=value(1:5)
    call ulower(ztest5)

    !  DMC -- handle list option if necessary
    if(varnam.eq.'input_plasma_state') then
       if(ztest5.eq.'list') then
          if(init) then
             write(ilun_msgs,*) ' '
             write(ilun_msgs,*) ' ?? "list" setting invalid: action = INIT.'
             echo_error = .TRUE.
             return
          else if(iskip) then
             write(ilun_msgs,*) ' '
             write(ilun_msgs,*) ' ?? "list" setting invalid: action = SKIP.'
             echo_error = .TRUE.
             return
          endif
          list_flag = .TRUE.
       else if(ztest5.eq.'list:') then
          if(.not.init) then
             write(ilun_msgs,*) ' '
             write(ilun_msgs,*) ' ?? "list:<filename>" setting only valid for action = INIT.'
             echo_error = .TRUE.
             return
          endif
          list_flag = .TRUE.
          zfile = value(6:)
          if(zfile.eq.' ') then
             write(ilun_msgs,*) ' '
             write(ilun_msgs,*) ' ?? "list:<filename>" syntax during INIT:'
             write(ilun_msgs,*) '    <filename> must not be blank.'
             echo_error = .TRUE.
             return
          endif
          zcmd = 'cp '//trim(workpath)//trim(zfile)//' '// &
               trim(workpath)//'input_states.list'
          istat = jsystem(zcmd)
          if(istat.ne.0) then
             write(ilun_msgs,*) ' '
             write(ilun_msgs,*) ' ?? "list:<filename>" copy failure.'
             echo_error = .TRUE.
             return
          endif
       endif

       !  DMC -- if list option is active check/read the file
       if(list_flag) then
          write(ilun_msgs,*) ' '
          write(ilun_msgs,*) '   (open/check plasma state list file...)'
          write(ilun_msgs,*) ' '

          open(unit=ilun_naml,file=trim(workpath)//'input_states.list', &
               status='old',iostat=istat)
          if(istat.ne.0) then
             write(ilun_msgs,*) '   ?? open failure: '//trim(workpath)//'input_states.list'
             echo_error = .TRUE.
             return
          endif

          call read1line(zline,istat)  ! will skip header...
          if(istat.ne.0) then
             write(ilun_msgs,*) '   ?? first lin read error: '//trim(workpath)//'input_states.list'
             echo_error = .TRUE.
             close(unit=ilun_naml)
             return
          endif

          call read1line(zline,istat)
          if(istat.ne.0) then
             write(ilun_msgs,*) '   ?? second lin read error: '//trim(workpath)//'input_states.list'
             echo_error = .TRUE.
             close(unit=ilun_naml)
             return
          endif

          !  OK -- parse line; reassign value => 1st input plasma state

          call pslist_line_parse('echo',zline,ztime,value,zkey,iertmp)
          if(iertmp.ne.0) then
             echo_error = .TRUE.
             close(unit=ilun_naml)
             return
          endif

          close(unit=ilun_naml)
       endif
    endif

    idot=0
    do ic=ilenf,1,-1
       if(value(ic:ic).eq.'/') exit
       if(value(ic:ic).eq.'.') then
          idot=ic
          exit
       endif
    enddo

    if(idot.eq.0) then
       value(ilenf+1:ilenf+5)='.data'
       ilenf=ilenf+5
       write(ilun_msgs,*) '     ... appended: ".data" to file path.'
    else if(idot.eq.ilenf) then
       value(ilenf+1:ilenf+4)='data'
       ilenf=ilenf+4
       write(ilun_msgs,*) '     ... appended: "data" to file path.'
    endif

    zfile = trim(workpath)//trim(value)
    value = trim(zfile)

  end subroutine echo

  subroutine get_state_list(ier)

    !  get the state list; read states & time vector
    !  this is only done if actual NUBEAM time steps are expected,
    !  and a list rather than a single state is provided as input.

    !  handling of sawtooth event-- list is truncated if there is 
    !  a sawtooth at the last time.

    integer, intent(out) :: ier   ! completion code (0=OK)

    !---------------------------------
    character*140 :: zline
    character*80 :: zfile
    character*32 :: zkey
    character*5 :: zi5
    integer :: indx,iss,ilast_saw
    logical :: sawflag
    !---------------------------------

    open(unit=ilun_naml,file=trim(workpath)//'input_states.list', &
         status='old',iostat=ier)
    if(ier.ne.0) then
       ier=1
       write(ilun_msgs,*) ' ?get_state_list: file open failure: '// &
            trim(workpath)//'input_states.list'
       return
    endif

    do 
       call read1line(zline,ier)
       if(ier.ne.0) then
          ier=1
          write(ilun_msgs,*) ' ?get_state_list: header line read failure: '// &
               trim(workpath)//'input_states.list'
          exit
       endif

       indx = index(zline,'Nstates:')
       if(indx.le.0) then
          ier=1
          write(ilun_msgs,*) ' ?get_state_list: header "Nstates" field missing: '// &
               trim(workpath)//'input_states.list'
          write(ilun_msgs,*) '  1st line: '//trim(zline)
          exit
       endif

       indx=indx+8
       read(zline(indx:indx+4),'(I5)') nss_list
       if(nss_list.le.0) then
          ier=1
          write(ilun_msgs,*) ' ?get_state_list: header "Nstates" field invalid: '// &
               trim(workpath)//'input_states.list'
          write(ilun_msgs,*) '  1st line: '//trim(zline)
          exit
       endif

       write(ilun_msgs,*) ' '
       write(ilun_msgs,*) ' ...nubeam_comp_exec:get_state_list: read ', &
            nss_list,' states...'
       write(ilun_msgs,*) ' '

       allocate(stimes(nss_list),ss_list(nss_list),sawflags(nss_list))

       do iss=1,nss_list
          write(zi5,'(I5.5)') iss

          call ps_init_user_state(ss_list(iss),'nbss_list_'//zi5,ier)
          if(ier.ne.0) exit
       enddo
       if(ier.ne.0) exit

       sawflag = .FALSE.
       do iss=1,nss_list
          sawflags(iss)=.FALSE.
          call read1line(zline,ier)
          if(ier.ne.0) then
             write(ilun_msgs,*) ' ?get_state_list: read error @state no.: ',iss
             exit
          endif

          call pslist_line_parse('get_state_list',zline, &
               stimes(iss),zfile,zkey,ier)
          if(ier.ne.0) then
             write(ilun_msgs,*) ' ?get_state_list: line parse error @state no.: ',iss
             exit
          endif

          write(ilun_msgs,*) ' '
          if(zkey.ne.' ') write(ilun_msgs,*) ' keyword: ',trim(zkey)
          write(ilun_msgs,'(a,1pe13.6)') &
               ' reading plasma state: '//trim(zfile)//' at t=',stimes(iss)

          call ps_get_plasma_state(ier, filename=trim(workpath)//zfile, &
               state=ss_list(iss))
          if(ier.ne.0) then
             write(ilun_msgs,*) ' ?get_state_list: state read error @state no.: ',iss
             exit
          endif

          if(sawflag) then
             stimes(iss) = stimes(iss-1)  ! make sawtooth instantaneous
          endif

          if((zkey.eq.'SAWTOOTH_START').and.(iss.lt.nss_list)) then
             if(sawflag) then
                write(ilun_msgs,*) ' ?get_state_list: cannot have two consecutive SAWTOOTH_START records.'
                ier=1
                exit
             endif

             sawflag=.TRUE.
             sawflags(iss)=.TRUE.
             ilast_saw=iss
          else
             sawflag=.FALSE.
          endif

       enddo
       write(ilun_msgs,*) ' '
       if(ier.ne.0) exit

       exit
    enddo
    close(unit=ilun_naml)

    if(stimes(ilast_saw).eq.stimes(nss_list)) then
       nss_list=ilast_saw
       sawflags(nss_list)=.FALSE.  ! ignore sawtooth event @end
    endif

  end subroutine get_state_list

  subroutine read1line(zline,ier)

    character*(*), intent(out) :: zline  ! ascii data read in
    integer, intent(out) :: ier          ! completion code, 0=normal

    read(ilun_naml,'(1x,a)',iostat=ier) zline

  end subroutine read1line

  subroutine pslist_line_parse(subr,zline,ztime_val,zfilenam,zkey,ier)

    !  parse input line  <time>  <filename>  [<keyword>]
    !    keyword if present is converted to uppercase; blank keyword OK

    character*(*), intent(in) :: subr   ! subroutine name (for messages)
    character*(*), intent(in) :: zline  ! input line

    real*8, intent(out) :: ztime_val    ! time value found
    character*(*), intent(out) :: zfilenam  ! filename found
    character*(*), intent(out) :: zkey  ! keyword found, or blank

    integer, intent(out) :: ier         ! completion code, 0=OK

    !  an error message is printed, if the parse fails.

    !------------------------------
    integer :: ic,ilen,i1,i2,ilenf,ifield
    logical :: cur_field
    character*20 z20
    !------------------------------

    ilen = len(trim(zline)) + 1   ! a trailing blank is assumed...
    ifield = 0
    cur_field = .FALSE.

    !  initial output values

    ztime_val = 0.0d0
    zfilenam  = ' '
    zkey      = ' '
    ier       = 0

    !  scan line, find fields...
    do ic=1,ilen

       if(.not.cur_field) then
          if(zline(ic:ic).ne.' ') then
             cur_field = .TRUE.
             ifield = ifield + 1
             i1 = ic    ! start of field #ifield
          endif

       else
          if(zline(ic:ic).eq.' ') then
             cur_field = .FALSE.
             i2 = ic-1  ! end of field #ifield
             ilenf = i2-i1+1
          else
             cycle
          endif

          ! parse the completed field now...

          if(ifield.gt.3) then

             write(ilun_msgs,*) ' ?'//trim(subr)//': too many fields in line:'
             write(ilun_msgs,*) '  '//trim(zline)
             ier = 1
             exit

          else if(ifield.eq.1) then
             z20 = ' '
             z20(20-ilenf+1:20) = zline(i1:i2)
             read(z20,'(G20.0)',iostat=ier) ztime_val
             if(ier.ne.0) then
                ier=1
                write(ilun_msgs,*) ' ?'//trim(subr)//': time value parse error:'
                write(ilun_msgs,*) '  '//trim(zline)
                exit
             endif

          else if(ifield.eq.2) then
             zfilenam = zline(i1:i2)

          else if(ifield.eq.3) then
             zkey = zline(i1:i2)
             call uupper(zkey)

          endif
       endif

    enddo

  end subroutine pslist_line_parse

  subroutine get_exec_times(ier)

    !  if input data consists of a time series of states, set up
    !  the time steps through the time series:
    !    (a) default, recognized here by {istep_ct=1, dt_step=0.0},
    !        is to step through the input states, using their time
    !        spacing to determine time steps, until the end.
    !    (b) if NUBEAM_REPEAT_COUNT is used, the time range covered
    !        is constrained to the available time in the state data;
    !    (c) deal with restart -- get time information from NUBEAM state

    !----------------------------------
    integer, intent(out) :: ier
    !----------------------------------
    logical :: idefault_ct, idefault_dt, isaw_next

    integer :: ibsteps,insafe,inumt,inumtarg,ict,iit
    real*8 :: ztbm1,ztbm2,ztime0,zdt_step,zdtnext,zdt_total
    !----------------------------------

    ier = 0

    !  get safe upper limit on number of NUBEAM calls in time loop...

    idefault_dt = (dt_step.le.0.0d0)
    idefault_ct = (istep_ct.eq.1).AND.idefault_dt

    zdt_total = istep_ct*dt_step
    if(zdt_total.gt.(stimes(nss_list)-stimes(1))) then
       write(ilun_msgs,'(a,1pe13.6)') &
            ' %get_exec_times: NUBEAM_REPEAT time range = ',zdt_total
       write(ilun_msgs,'(a,1pe13.6)') &
            '  exceeds time range of states: ', stimes(nss_list)-stimes(1)
    endif

    if(idefault_dt) then
       insafe = 2*max(istep_ct,nss_list)
    else
       insafe = 2*(stimes(nss_list)-stimes(1))/dt_step
       insafe = max(insafe,2*istep_ct)
    endif

    inumtarg=insafe
    if(.not.idefault_ct) inumtarg=max(1,istep_ct)

    allocate(xtimes(insafe),index1_state(insafe),index2_state(insafe))
    allocate(index_sawstate(insafe),index1_exact(insafe))

    xtimes=0.0d0; index1_state = 0; index2_state = 0; index_sawstate = 0
    index1_exact = .FALSE.

    call nubeam_prev_timestep(ibsteps,ztbm1,ztbm2)
    if(ibsteps.eq.0) then
       ztime0=stimes(1)
    else
       ztime0=ztbm2
    endif

    if(ztime0.ge.stimes(nss_list)) then
       ier=1
       write(ilun_msgs,*) ' '
       write(ilun_msgs,*) ' ?nubeam_comp_exec: input data time range: '
       write(ilun_msgs,*) '  ',stimes(1),' to ',stimes(nss_list),' seconds.'
       write(ilun_msgs,*) '  NUBEAM state is at time: ',ztime0
       write(ilun_msgs,*) '  Time limit of the input data has been reached.'
       return
    endif

    !  ztime0 is the start time for current execution of nubeam_comp_exec

    inumt=1
    xtimes(1)=ztime0

    call xt_indices(xtimes(1),1,isaw_next,zdtnext)
    if(index1_exact(1)) then
       ss_in => ss_list(index1_state(1))
    else
       ! hold time interpolated initial input state in instance (psp)
       ztprev = stimes(index1_state(1))
       ztnext = stimes(index2_state(1))
       zfac = (ztnext-xtimes(1))/(ztnext-ztprev)
       call chk_alloc(ss_list(index1_state(1)),psp)
       call ps_merge_plasma_state(zfac, &
            ss_list(index1_state(1)), ss_list(index2_state(1)), ier, &
            new_state = psp, icheck=ps_ignore)
       ss_in => psp
       if(ier.ne.0) then 
          write(ilun_msgs,*) ' ?nubeam_comp_exec: 1st state merge error.'
          return
       endif
    endif

    ict=0
    do
       if(ict.ge.inumtarg) exit
       if(xtimes(inumt).ge.stimes(nss_list)) exit

       ztprev = xtimes(inumt)
       ztnext = stimes(index2_state(inumt))

       ict=ict+1
       inumt=inumt+1

       if(idefault_dt) then
          xtimes(inumt)=ztnext
       else
          zdt_step=min(dt_step,zdtnext)
          if(zdt_step.lt.dt_step) then
             write(ilun_msgs,*) ' %get_exec_times: time step shortened...'
          endif
          if(isaw_next) then
             if((ztnext-ztprev).lt.1.5d0*zdt_step) then
                xtimes(inumt)=ztnext
             else if((ztnext-ztprev).lt.2.0d0*zdt_step) then
                xtimes(inumt)=ztprev + 0.5d0*(ztnext-ztprev)
             else
                xtimes(inumt)=ztprev + zdt_step
             endif
          else
             xtimes(inumt)=ztprev + zdt_step
          endif
       endif
       call xt_indices(xtimes(inumt),inumt,isaw_next,zdtnext)
    enddo

    if(index1_exact(inumt)) then
       ss_out_base => ss_list(index1_state(inumt))
    else
       ! hold time interpolated state, at final output time, in (ps_next).
       ztprev = stimes(index1_state(inumt))
       ztnext = stimes(index2_state(inumt))
       zfac = (ztnext-xtimes(inumt))/(ztnext-ztprev)
       call chk_alloc(ss_list(index1_state(inumt)),ps_next)
       call ps_merge_plasma_state(zfac, &
            ss_list(index1_state(inumt)), ss_list(index2_state(inumt)), ier, &
            new_state = ps_next, icheck=ps_ignore)
       ss_out_base => ps_next
       if(ier.ne.0) then 
          write(ilun_msgs,*) ' ?nubeam_comp_exec: last state merge error.'
          return
       endif
    endif

    istep_ct = ict
    write(ilun_msgs,*) ' get_exec_times summary: '
    write(ilun_msgs,*) '  #steps: ',istep_ct
    do iit=1,ict
       write(ilun_msgs, &
            '(3x,i3,". ",1pe13.6," to ",1pe13.6," dt=",1pe13.6," s.")') &
            iit, xtimes(iit), xtimes(iit+1), xtimes(iit+1)-xtimes(iit)
    enddo

  end subroutine get_exec_times

  subroutine xt_indices(ztime,it,isaw_next,zdtnext)

    !-----------------------------
    !  find time indices
    !-----------------------------

    real*8, intent(in) :: ztime  ! time to use
    integer, intent(in) :: it    ! index in xtimes(...) etc.
    logical, intent(out) :: isaw_next  ! return .TRUE. if next time -> sawtooth
    real*8, intent(out) :: zdtnext     ! upcoming dt spacing of states

    !-----------------------------
    integer :: ii,ifound,i1,i2
    !-----------------------------

    if((ztime.lt.stimes(1)).OR.(ztime.gt.stimes(nss_list))) then
       call errmsg_exit('?? time out of range, xt_indices(...)')
    endif

    isaw_next = .FALSE.

    ifound = 0
    do ii=1,nss_list
       if(ztime.eq.stimes(ii)) then
          ifound = ii
          index1_exact(it)=.TRUE.
          if(sawflags(ii)) then
             !  when sawflags is defined we make sure there are subsequent
             !  times...
             index_sawstate(it)=ii
             index1_state(it)=ii+1
             index2_state(it)=ii+2
             isaw_next=sawflags(ii+2)
          else
             index1_state(it)=ii
             index_sawstate(it)=0
             if(ii.lt.nss_list) then
                index2_state(it)=ii+1
                isaw_next=sawflags(ii+1)
             else
                index2_state(it)=ii
             endif
          endif
          exit
       endif
    enddo

    if(ifound.eq.0) then
       do ii=1,nss_list-1
          if((stimes(ii).lt.ztime).AND.(ztime.lt.stimes(ii+1))) then
             index1_state(it)=ii
             index2_state(it)=ii+1
             index_sawstate(it)=0
             isaw_next=sawflags(ii+1)
             exit
          endif
       enddo
    endif

    i1=index1_state(it)
    i2=index2_state(it)
    zdtnext=stimes(i2)-stimes(i1)
    if((.not.isaw_next).AND.(i2.lt.nss_list)) then
       zdtnext=min(zdtnext,(stimes(i2+1)-ztime))
    endif

    if(i2.eq.nss_list) isaw_next = .TRUE.  ! treat last time pt like an event

  end subroutine xt_indices

  subroutine chk_alloc(state1,state2)

    ! check if state2%nrho is non-zero; if not, copy dims from state1 and
    ! allocate...

    type (plasma_state) :: state1,state2

    if(state2%nrho.gt.0) return

    call ps_copy_dims(state1,state2,1,ierr)
    if(ierr.ne.0) then
       call errmsg_exit(' ??nubeam_comp_exec(chk_alloc): ps_copy_dims error.')
    endif

    call ps_alloc_plasma_state(ierr, state=state2)
    if(ierr.ne.0) then
       call errmsg_exit( &
            ' ??nubeam_comp_exec(chk_alloc): ps_alloc_plasma_state error.')
    endif

  end subroutine chk_alloc

#ifdef _JONGKYU_PARK
  subroutine ps_xplasma_write
    !  extract xplasma pointer; write xplasma file
    !  (option provided at request of grad student Jong-kyu Park).

    use xplasma_definitions
    use xplasma_obj_instance

    type :: interp_object
       type (xplasma), pointer :: s_xpobj => NULL()
    end type interp_object

    type (interp_object) :: xobj
    character*200 :: xpath

    !------------------------------

    xobj = transfer(ss_in%iobj,xobj)
    s => xobj%s_xpobj

    xpath='pstate-xplasma-write.dat'
    call xplasma_write(s,trim(xpath),ierr)

    write(ilun_msgs,*) ' '
    if(ierr.eq.0) then
       write(ilun_msgs,*) ' %jsocdf2ps:  wrote xplasma file: ',trim(xpath)
    else
       write(ilun_msgs,*) ' %jsocdf2ps:  error writing xplasma file: ', &
            trim(xpath)
       call xplasma_error(s,ierr,ilun_msgs)
    endif

  end subroutine ps_xplasma_write
#endif

  subroutine check_nubeam_action

    ilenv = max(1,len(trim(action_value)))
    call uupper(action_value(1:ilenv))

    ISKIP=.FALSE.
    INIT=.FALSE.
    ISKIP_OUTPUT=.FALSE.
    ibackup=0
    iretrieve=0

    if(action_value(1:ilenv).eq.'INIT') then
       INIT=.TRUE.
       write(ilun_msgs,*) ' %nubeam_comp_exec:  NUBEAM_ACTION = INIT (rng init from system clock).'

    else if(action_value(1:ilenv).eq.'INIT_HOLD') then
       INIT=.TRUE.
       ilseed_reset=.FALSE.
       action_value = 'INIT'
       ilenv = 4
       write(ilun_msgs,*) ' %nubeam_comp_exec:  NUBEAM_ACTION = INIT (rng init from namelist).'

    else if(action_value(1:ilenv).eq.'STEP') then
       write(ilun_msgs,*) ' %nubeam_comp_exec:  NUBEAM_ACTION = STEP.'

    else if(action_value(1:ilenv).eq.'BACKUP') then
       write(ilun_msgs,*) ' %nubeam_comp_exec:  NUBEAM_ACTION = BACKUP.'
       ibackup=1

    else if(action_value(1:ilenv).eq.'RETRIEVE') then
       write(ilun_msgs,*) ' %nubeam_comp_exec:  NUBEAM_ACTION = RETRIEVE.'
       iretrieve=1

    else if(action_value(1:ilenv).eq.'SKIP') then
       write(ilun_msgs,*) ' %nubeam_comp_exec:  NUBEAM_ACTION = SKIP.'
       ISKIP=.TRUE.

    else
       write(ilun_msgs,*) ' '
       write(ilun_msgs,*) '  NUBEAM_ACTION = "'//action_value(1:ilenv)//'".'
       write(ilun_msgs,*) &
            ' ?nubeam_comp_exec:  unrecognized environment variable control value:'
       write(ilun_msgs,*) '  (expected to be "INIT" or "STEP" or "BACKUP" or "RETRIEVE" or "SKIP").'
       write(ilun_msgs,*) ' '
       write(0,*) ' ?nubeam_comp_exec: myid=',myid, &
            ' NUBEAM_ACTION value error: '//action_value(1:max(1,ilenv))
       call bad_exit
    endif

    if((ibackup.eq.1).or.(iretrieve.eq.1)) then
       if((inum_frantic.gt.0).or.ifran_wall.or.ifran_halo.or.ifran_reco) then
          write(0,*) ' ?check_nubeam_action: when NUBEAM_ACTION="backup" or "retrieve",'
          write(0,*) '  FRANTIC_INIT and/or FRANTIC_ACTION cannot be set.'
          call bad_exit
       endif
    endif

    if(iskip) then
       if(.not.((inum_frantic.gt.0).or.ifran_wall.or.ifran_halo.or.ifran_reco)) then
          write(0,*) ' ?check_nubeam_action: when NUBEAM_ACTION="skip",'
          write(0,*) '  FRANTIC_ACTION must select a neutral transport action.'
          call bad_exit
       endif
    endif

  end subroutine check_nubeam_action

  subroutine check_nubeam_postproc

    integer :: ilentot,iarg_ok

    ilenv = index(postproc_value,':')
    if(ilenv.le.0) then
       ilenv = max(1,len(trim(postproc_value)))
       ilentot = ilenv
    else
       ilentot = max(1,len(trim(postproc_value)))
       if(ilentot.eq.ilenv) then
          ! trim trailing colon
          postproc_value(ilenv:ilenv)=' '
          ilenv=max(1,(ilenv-1))
          ilentot=ilenv
       else
          ilenv=ilenv-1  ! so tests below do not see ":"
       endif
    endif
    call uupper(postproc_value(1:ilenv))

    if(postproc_value(1:ilenv).eq.' ') then
       postproc_value(1:4) = 'NONE'
       ilenv=4
       ilentot=ilenv
    endif

    iarg_ok = 0
    if(postproc_value(1:ilenv).eq.'NONE') then
       write(ilun_msgs,*) ' %nubeam_comp_exec: postprocessing option: NONE.'

    else if(postproc_value(1:ilenv).eq.'SUMMARY_TEST') then
       write(ilun_msgs,*) ' %nubeam_comp_exec: SUMMARY_TEST postprocessing requested.'

    else if(postproc_value(1:ilenv).eq.'FBM_WRITE') then
       write(ilun_msgs,*) ' %nubeam_comp_exec: FBM_WRITE postprocessing requested.'
       if(ilenv.lt.ilentot) then
          write(ilun_msgs,*) '  output filename: '// &
               postproc_value(ilenv+2:ilentot)
       endif
       iarg_ok=1

    else if(postproc_value(1:ilenv).eq.'NO_OUTPUT') then
       write(ilun_msgs,*) ' %nubeam_comp_exec: postprocessing option: NO_OUTPUT.'

    else
       write(0,*) ' ?nubeam_comp_exec: value of NUBEAM_POSTPROC: '// &
            trim(postproc_value)//' invalid.'
       call bad_exit
    endif

    if(ilenv.lt.ilentot) then
       if(iarg_ok.eq.0) then
          write(0,*) ' ?nubeam_comp_exec: filename argument invalid: '// &
               postproc_value(1:ilentot)
          call bad_exit
       endif
    endif

  end subroutine check_nubeam_postproc

  subroutine check_workpath

    !  WORKPATH adjustment

    if(workpath.eq.' ') then
       workpath='./'
       ilenw=2

    else
       ilenw=len(trim(workpath))
       if(workpath(ilenw:ilenw).ne.'/') then
          ilenw=ilenw + 1
          workpath(ilenw:ilenw)='/'
       endif
    endif
    write(0,*) ' %nubeam_comp_exec: NUBEAM_WORKPATH: '//trim(workpath)

    runid_filename = trim(workpath)//'nubeam_comp_exec.RUNID'

  end subroutine check_workpath

  subroutine check_loglevel
    !LOG_LEVEL check

    write(0,*) ' %nubeam_comp_exec: LOG_LEVEL env. var:' , trim(logfile_level)

    if(trim(logfile_level).eq.'0') then
       logfile_level='info'
    else if(trim(logfile_level).eq.'1') then
       logfile_level='warn'
    else if(trim(logfile_level).eq.'2') then
       logfile_level='err'
    else if(trim(logfile_level).eq.'3') then
       logfile_level='nomsg'
    else
       logfile_level='warn'
    endif
    write(0,*) ' !nubeam_comp_exec:   LOGFILE_LEVEL: ',trim(logfile_level)

  end subroutine check_loglevel
  
  subroutine check_namelist_logic

    !--------------------------------------
    !  namelist input -- logic check

    integer :: jj,jdot
    character*1 :: tchar

    ibackup = 0
    iretrieve = 0
    if(action_value.eq.'BACKUP') ibackup=1
    if(action_value.eq.'RETRIEVE') iretrieve=1

    ierr = 0

    if((ibackup.eq.0).AND.(iretrieve.eq.0)) then
       if(input_plasma_state.eq.' ') then
          write(ilun_msgs,*) &
               ' ?nubeam_comp_exec: "input_plasma_state" not specified.'
          ierr = ierr + 1
       endif
    else
       iskip_output = .TRUE.  ! backup or retrieve ONLY
    endif

    ! output selection (if neither: input state file is rewritten with
    ! modifications...

    ps_update_flag = (plasma_state_update.ne.' ')
    ps_output_flag = (output_plasma_state.ne.' ')
    if(.not.iskip_output) then
       if(list_flag.or.(istep_ct.gt.1)) then
          if(.not.(ps_output_flag.or.ps_update_flag)) then
             write(ilun_msgs,*) &
                  ' ?nubeam_comp_exec: must name output or update state file'
             write(ilun_msgs,*) &
                  '  when NUBEAM_REPEAT_COUNT is set, or when a list of input'
             write(ilun_msgs,*) &
                  '  states are given.'
             ierr = ierr + 1
          endif
       endif
    else
       ! iskip_output TRUE

       if(ps_update_flag) then
          write(ilun_msgs,*) ' %nubeam_comp_exec: ignored plasma_state_update.'
       endif
       ps_update_flag=.FALSE.

       if(ps_output_flag) then
          write(ilun_msgs,*) ' %nubeam_comp_exec: ignored output_plasma_state.'
       endif
       ps_output_flag=.FALSE.

    endif
    ps_output_any = ps_update_flag.OR.ps_output_flag

    iinit=0
    if(.not.iskip) then
       if((init_namelist.ne.' ').or.(init_cdf.ne.' ')) then
          iinit=1
       endif
    endif
 
    istep=0
    if(.not.iskip) then
       if((ibackup.eq.0).and.(iretrieve.eq.0)) then
          if((step_namelist.ne.' ').or.(step_cdf.ne.' ')) then
             istep=1
          endif
       endif
    endif

    if(init) then
       if(iinit.eq.0) then
          write(ilun_msgs,*) &
               ' ?nubeam_comp_exec: INIT action requires input file:'
          write(ilun_msgs,*) &
               '  "init_namelist" or "init_cdf" must be specified.'
          ierr = ierr + 1
       endif
       ! (it is optional whether step input is specified, when INIT is on).
    else
       if(iinit.eq.1) then
          write(ilun_msgs,*) &
               ' ?nubeam_comp_exec: action is not "INIT" so init filenames:'
          write(ilun_msgs,*) &
               '  "init_namelist" or "init_cdf" must NOT be specified.'
          ierr = ierr +1 
           
       endif
       if((istep.eq.0).and.(ibackup.eq.0).and.(iretrieve.eq.0).and. &
            (.not.iskip)) then
          write(ilun_msgs,*) &
               ' ?nubeam_comp_exec: STEP action requires input file:'
          write(ilun_msgs,*) &
               '  "step_namelist" or "step_cdf" must be specified.'
          ierr = ierr + 1
       endif

    endif

    ! RF data checks

    if((rf_idata.ne.' ').OR.(rf_odata.ne.' ')) then
       if((rf_idata.eq.' ').OR.(rf_odata.eq.' ')) then
          write(ilun_msgs,*) ' ?nubeam_comp_exec: missing RF filename:'
          write(ilun_msgs,*) '  rf_idata: ',trim(rf_idata)
          write(ilun_msgs,*) '  rf_odata: ',trim(rf_odata)
          write(ilun_msgs,*) '  ...if one is specified BOTH must be specified.'
          ierr = ierr + 1
       endif
       if(list_flag) then
          write(ilun_msgs,*) ' ?nubeam_comp_exec: use of RF data with time'
          write(ilun_msgs,*) '  series state input data: not supported yet!'
          ierr = ierr + 1
       endif
    endif

    if(ierr.gt.0) then
       call bad_exit
    else
       write(ilun_msgs,*) ' %nubeam_comp_exec: namelist logic check OK.'
    endif

  end subroutine check_namelist_logic

  subroutine default_filename(filename,suffix)
    !  if filename is blank, provide a default: <workpath>/<runid><suffix>

    character*(*), intent(inout) :: filename
    character*(*), intent(in) :: suffix

    if(filename.eq.' ') then
       filename = trim(workpath)//trim(runid)//trim(suffix)
       write(ilun_msgs,*) ' %using default filename: '//trim(filename)
    endif
  end subroutine default_filename


  subroutine mark_nbi_states(zchar)

    character*1 :: zchar

    write(ilun_msgs,*) ' ...CALL nbi_states(...,"'//zchar//'",...)'
    if(myid.eq.0) then
       write(0,*) ' (cpu0) ...CALL nbi_states(...,"'//zchar//'",...)'
    endif

  end subroutine mark_nbi_states

  subroutine parse_repeat_value

    ! parse expression in C*40 repeat_value
    !   either an integer of form n or nn or nnn (value btw 1 and 999)
    !   or an integer "x" a floating point value: nnnxmm.mmm
    !     e.g. "10x0.050" or "5x1.0e-2"

    integer :: ixbrk,ilenr,ileni,ierloc,istat

    character*4 :: zint
    character*20 :: zflt

    ilenr = len(trim(repeat_value))
    ixbrk = index(repeat_value,'x')
    
    if(ixbrk.le.0) then
       ileni=ilenr
    else
       ileni=ixbrk-1
    endif

1001 format(/ &
          ' generic format: <integer> or <integer>x<float>'/ &
          ' <integer> value must be between 1 and 999 inclusive;'/ &
          ' <float> value if present must be between 0.0001 and 100.000 sec.'/)
    
    if((ileni.lt.1).or.(ileni.gt.4)) then
       write(ilun_msgs,1001)
       call errmsg_exit('?nubeam_comp_exec: unexpected integer field length in NUBEAM_REPEAT_COUNT: '//repeat_value)
    endif

    zint=' '
    zint(4-ileni+1:4) = repeat_value(1:ileni)

    read(zint,'(I4)',iostat=istat) istep_ct
    if(istat.ne.0) then
       write(ilun_msgs,1001)
       call errmsg_exit('?nubeam_comp_exec: integer field parse error in NUBEAM_REPEAT_COUNT: '//repeat_value)
    endif
    if((istep_ct.lt.1).or.(istep_ct.gt.999)) then
       write(ilun_msgs,1001)
       call errmsg_exit('?nubeam_comp_exec: integer field value error in NUBEAM_REPEAT_COUNT: '//repeat_value)
    endif

    if(ixbrk.le.0) return

    zflt = ' '
    ileni = ilenr-ixbrk

    if((ileni.lt.1).or.(ileni.gt.20)) then
       write(ilun_msgs,1001)
       call errmsg_exit('?nubeam_comp_exec: unexpected floating pt field length in NUBEAM_REPEAT_COUNT: '//repeat_value)
    endif

    zflt(20-ileni+1:20) = repeat_value(ixbrk+1:ilenr)
    read(zflt,'(G20.0)') dt_step
    if(istat.ne.0) then
       write(ilun_msgs,1001)
       call errmsg_exit('?nubeam_comp_exec: float field parse error in NUBEAM_REPEAT_COUNT: '//repeat_value)
    endif
    if((dt_step.lt.0.001d0).or.(dt_step.gt.100.0d0)) then
       write(ilun_msgs,1001)
       call errmsg_exit('?nubeam_comp_exec: float field value error in NUBEAM_REPEAT_COUNT: '//repeat_value)
    endif

  end subroutine parse_repeat_value

  subroutine frantic_init_check

    !---------------------------------------
    integer :: iii
    real*8, parameter :: ZERO=0.0d0
    real*8, parameter :: ONE=1.0d0
    !---------------------------------------

    if(ss_out%nrho_gas.gt.0) then
       iii = ss_out%nrho_gas - 1
       if((inum_frantic.gt.0).and.(iii.ne.inum_frantic)) then
          write(0,*) ' %nubeam_comp_exec: FRANTIC zones already allocated, #zones = ',iii
          write(0,*) '  This is not consistent with FRANTIC_INIT #zones = ',inum_frantic
          write(0,*) '  The number of zones remains unchanged.'
       endif
       inum_frantic = iii
    else
       if(inum_frantic.eq.0) then
          inum_frantic = max(ss_out%nrho_nbi,ss_out%nrho_fus) - 1
          if(inum_frantic.gt.0) then
             write(0,*) ' %nubeam_comp_exec: #zones for FRANTIC set to match fast ion grid(s):',inum_frantic
          else
             inum_frantic=50
             write(0,*) ' %nubeam_comp_exec: #zones for FRANTIC set to default value: ',inum_frantic
          endif
       endif
    endif
  end subroutine frantic_init_check

end program nubeam_comp_exec
