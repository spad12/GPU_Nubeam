C----------------------------------------------------------------------
C  ORBALL -- DO ALL MONTE CARLO ORBIT LOOPS (PREV. PART OF FIMAIN)
C
C   MG  MARCH 2009 -- for MPI each process carries a single 
C                     sublist of particles  instead of both sublist and 
C                     whole list arrays.
C   DMC Apr 2007 -- adding ability to repeat the advance of all orbits
C      By saving and retrieving the ptcl variables.  ALSO: NLBOUT=T
C      I/O feature removed (was not working with MPI in any case).
C
C   DMC MAY 2001 -- commenting our RF resonant operator (DEFUNCT)
C      (The FPP code interface is being rewritten-- the code here
C       would also have to be redone.  To do this would require
C       evidence of use)...
C
C   DMC AUG 1988 -- SUPPORT MULTISPECIES BEAMS
C
      SUBROUTINE ORBALL(iorbtot,ierr)
C
      use nbi_com
      use nbi_random_pll
      use nbi_random
      use nbi_mpi_com
      use nbi_fbm_random
      use orbrzv
      use logmod
      use dep_mpi_split ! use this module to deposit CX tracks for statistics improvement
                        ! does not interfer with depall
C
      IMPLICIT NONE

      !INTEGER, PARAMETER :: R8=SELECTED_REAL_KIND(12,100)
      integer, intent(out) :: iorbtot   ! total no. of orbits calculated.
      integer, intent(out) :: ierr      ! status code (0=OK)
C
C  local stuff...
C
      integer isb,iorb,j,irecap,ind,jptr,ibflg,ixiav
      integer :: iersum,ib,ien,isplit,jsplit,indsave
      integer iuncon,itrap,jloc,idum
      integer :: inuse(mibs)    !original ptcl in use
      integer :: inuse_p(mibs)    !original ptcl in use on given processor
      REAL*8 zwght1,zang,zxhw,zphiav,zeiav,zxksav,xjadep,zrrf
      REAL*8 zadjust,zwght2,zw,zav,zvar2,znlimdt
      REAL*8 zpw, zfrnm, zprft, ztprft
      integer :: i,nuseycx,icx
      integer orbsteps
c-------------------------------
c  max energy for time step adjust normalization

      real*8, parameter :: zmaxe_dtc = 500.0d3

c-------------------------------
c  DMC Apr 2007 -- testing rerun of orbit loop, controlled from within
c  orball; results should be identical...
c
c  set the following variable to the number of times to repeat the calculation
c  (normal value: 1, just do the calculation once).
c
c  calculation may be redone for other reasons (e.g. adjust RF wave field).
c
      integer, parameter :: ipass_num = 3  ! >1 for debug ONLY
      real*8, parameter :: zpwr_errmax = 0.1_r8
      integer :: ipass
      logical :: irepeat_possible, irepeat, itmp_file
      logical :: key_myidd_0
c-------------------------------

#ifdef __DEBUG
      REAL*8 zwdep,zwdepp,zwold,zwoldp
#endif
C
      LOGICAL ILRECAP,ifirst,ilsplit,inew
C
C%      integer ibplot
C%      LOGICAL LYES,INWONL
C
C  MPI related stuff
C
      integer :: mpierr,jk,iseed,indnew
      integer inumproc, iscreen1,idt,idk,impitime
C
      character*200 :: zorball_tmp_filename
C
      real*8 :: ztim1,ztim2,zrandom,ztim3,ztim4
      real*8 :: ztime1,ztime2,ztime3,ztime4
      integer find_ptcl_ind
      integer, dimension(:), allocatable ::nuse_total! Total # of MC ptcls currently in storage
                                                 ! is using for a global control during deposition 
                                                 ! and orbiting
      integer :: iost
      real*8 :: wbav_c, p_inj_rs,fbmcxsum
      integer :: ncx1
      real*8 :: xncx0_use
      integer :: i1,j1
      integer :: gpulun
C
      if(allocated(nuse_total)) deallocate(nuse_total)
      allocate(nuse_total(mibs)); nuse_total = puseyp + 0.5_R8
C
C------------------------------------------------------------------
C
      ierr=0
      call cptimr8(ztim3)
	  call gpu_orbit_input(XI,ierr)
	  call cptimr8(ztim4)
	  print*, 'GPU Call took ',(ztim4-ztim3)
      itmp_file = .FALSE.

      impitime=1
      ttime31=0.0_r8
      ttime32=0.0_r8
#ifdef __MPI
      ttime31=MPI_WTIME()
#endif
      if(only_io) then
         call errorLog(
     >        'ORBALL:orball(nubeam): improper call, ONLY_IO = .TRUE.')
         if(myidd.eq.0) then
            write(nonlin,*) 
     >           ' ?orball(nubeam): improper call, ONLY_IO = .TRUE.'
         endif
         ierr=666
         go to 9999
      endif

      call cptimr8(ztim1)

C  sawtooth related setups (dmc Nov. 2005) (serial code)

      sawflag = sawflag.or.sawflag2  ! TRUE to use pre-calculated operator

C  pre_sawnbi_setup was called by nbstart to initialize calculation of an
C  operator in the upcoming step...

C
C  see if this is the first step (all ptcls new)
C
      ifirst=.TRUE.
      do isb=1,nsbeam
         ifirst = ifirst.and.(nuse_total(isb).eq.nnew(isb))
      enddo
C
C  CLEAR SUMS:  BEAM ION POPULATION ENERGY, MOMENTUM, AND N
C%      IBPLOT=NBPLOT
C%        IF(NBPLOT.GT.0) CALL NBPLOT_BDY(0)
C%      CALL UREAD('ORBALL:  PLOT ONLY NEWLY DEPOSITED ORBITS ? (Y/N)$')
C%      INWONL=LYES(0)
C%      IF(INWONL) NBPLOT=-1
C
      NORBTOT=0
C
      xnum_dt_adj0=0.0_R8
      dt_adj0_sums=0.0_R8
C
      iersum = 0
      stime9=0.0_r8
C
      iscreen1=1600+myidd

      DO 100 ISB=1,NSBEAM

 
C
         xncx0_orb(isb)=0._R8
         xncx0=0._R8
         inuse(isb) = nuse_total(isb)
         inuse_p(isb) = nuse(isb) !save nuse on previous step

         LSBEAM=ISB

         if(nlfbmfpp(LSBEAM)) go to 100
C
         CALL INISPC
C
         eirefs(isb)=min(zmaxe_dtc,einjs(isb))
         xmrefs(isb)=Raxis*abeams(isb)*1.6726E-24_R8*
     >      sqrt(eirefs(isb)/v2toev)
C
         IF(.NOT.NLHEAVY(LSBEAM)) CALL JBSHLD
C
         ZWGHT1=BNTOTS(LSBEAM)
C
         ENOLDS(LSBEAM)=0.0_R8
         BPHOLDS(LSBEAM)=0.0_R8
         BNTOTS(LSBEAM)=0.0_R8
         WNLIGHT(LSBEAM)=0.0_R8



C
C  check that all ORB outputs are zero now
C
         call nbi_zcheck_orb
         call nbi_zcheck_ntr
         
c  (parallelized) sawtooth operator...

         if(sawflag) then
            call sawnbi_exec
         endif

C
C  partial seed for ptcl RNG sequences
C
         zrandom=nbrandm(idum)
         ISEED=abs(100000000*zrandom)


C-----------------------------------------------
C  set up for possible recalculation of orbit loop
C  add logic here if recalculation due to RF is a possibility...

C  SEE ALSO the bottom of the loop where logic decides whether or not
C  to actually repeat the calculation

         irepeat = .FALSE.  ! repeat of calculation not decided yet...
         itmp_file = .FALSE.

         irepeat_possible = (ipass_num.gt.1)

C----------------
         if(irepeat_possible) then
            call orball_loclist_write  ! save local ptcl list in temporary file
            itmp_file = .TRUE.
         endif
         ipass = 0
C-----------------------------------------------
C  loop here if re-doing all orbits for a species
C   (e.g. to recalculate after an RF wave field renormalization)

 50      continue
         ipass = ipass + 1
         if(ipass.gt.1) then
            call nbi_zzero_orb       ! zero out ORB MPI-reduced outputs...
            call nbi_zzero_ntr       ! zero out ORB MPI-reduced outputs...
            call orball_loclist_read ! restore original ptcl list
         endif

C
         IORB=0
C
C      KEEP TRACK OF CURRENTLY STORED ENERGY AND PTCLS
C
         J=0
         NBSPLT=0

#ifdef __DEBUG
         zwdepp = 0
         zwdep = 0
         zwoldp = 0
         zwold = 0
#endif

C  CX RECAPTURE MONTE CARLO PTCLS COUNTER
         IRECAP=0
C  ORBIT SPLITTING COUNTERS
         ISPLIT=0
         JSPLIT=0
         ILSPLIT=.FALSE.

c  (parallelized) compression operator...
         if(NLCPRB) then
            call bcmprsn(ierr)
            if(ierr.ne.0) then
               orb_code_error(lsbeam) = orb_code_error(lsbeam) + 1
               go to 9999
            endif
         endif
C--------------------------------------------------------------
C  TOP OF ORBIT LOOP
C
 60      CONTINUE

C
         if(.not.ilsplit) then
            J=J+1
c            call warnLog('NOT ILSPLIT')
C  EXIT TEST
            IF(J.GT.(NUSE(ISB)-IRECAP-ISPLIT)) then
               GO TO 90
            endif

         endif
C
C
C  NEXT PTCL (FROM MAINSTREAM PTCL ARRAY)
C    (NOTE NBNDEX HAS BEEN REPLACED BY NBNDEXN)
c    ORIGINAL: IND=NBNDEX(J,LSBEAM)

         IND=NBNDEX(J,LSBEAM)
         ind=find_ptcl_ind(numprocd,myidd,ind) !find proper ptcl.ID#
         indsave=IND

         pinj_ay=pinjay(IND,LSBEAM)
         NBSCE=NBSCAY(IND,LSBEAM)
         NBIEN=NBIENAY(IND,LSBEAM)
         EINJ0=EINJAY(IND,LSBEAM)
         XION=XIAY(IND,LSBEAM)
         TH=THAY(IND,LSBEAM)
         VION=VAY(IND,LSBEAM)
         XKSID=XKSIDY(IND,LSBEAM)

         WGHT=WGHTAY(IND,LSBEAM)
         if(WGHT.le.0.0_R8) then
            call errorLog('ORBALL: WGHT=0.0')
            call errorLog('ORBALL: probable book keeping error !')
            call errorLog('lsbeam = ',lsbeam)
            call errorLog('ind = ',ind)
            call errorLog('nbndex(j,lsbeam) = ',nbndex(j,lsbeam))
            call errorLog('nuse_total(lsbeam) = ',nuse_total(lsbeam))
            call errorLog('nuse(lsbeam) = ',nuse(lsbeam))
            if(myidd.eq.0) then
               write(nonlin,*) ' ?? ORBALL lsbeam = ',lsbeam,' ind = '
     &              ,ind,nbndex(j,lsbeam)
               write(nonlin,*) ' ?? ORBALL nuse_total = ',
     &              nuse_total(lsbeam),nuse(lsbeam)
               write(nonlin,*) ' ?? ORBALL myidd = ',myidd,' J = ',j, 
     >              ' WGHT=0.0, probable book keeping error!'
               write(nonlin,*)NBNDEX(1:nuse(lsbeam),LSBEAM)
            endif
            call bad_exit
         endif

         tmfcur=tmfini
         GOOSE=GOOSAY(IND,LSBEAM)

         dt_adjust = dt_adj0s(lsbeam)
         if(dt_adjust.le.0.0_R8) dt_adjust=1.0_R8

         CXSKIP=CXPRAY(IND,LSBEAM)
         FPSKIP=FPPRAY(IND,LSBEAM)
         XISTATE=XISTAY(IND,LSBEAM)
C
C  launch time -- .GT.0 means a new orbit...
C
         BITIME(1)=TDEPAY(IND,LSBEAM)
         if(BITIME(1).eq.0.0_R8) then
c  OLD orbit
c
c  as a precaution (e.g. due to changed MHD equilibrium) reduce
c  goosing by 50% for old particles.  Reduce by more if the macro
c  beam timestep was shortened sufficiently (cf TFRACP from NBSTART).
c

            GOOSE=max(1.0001_R8,min(0.5_R8,TFRACP)*GOOSE)
            if(.not.ilsplit) then
               WGHT=WGHT*wghtakil(lsbeam) ! compensate for DEPALL->NBKILL RR
               WGHTAY(ind,lsbeam)=WGHT
               pinj_ay = pinj_ay*wghtakil(lsbeam)
               pinjay(IND,ISB)=pinj_ay
               
            endif
            inew=.FALSE.
#ifdef __DEBUG
            if(.not.ilsplit) then
               zwdepp = zwdep
               zwoldp = zwold
               zwold = zwold + WGHT
            endif
#endif
         else
c  NEW orbit
            WGHT=WGHT*wghtsave(lsbeam) ! rescale weight 
            WGHTAY(ind,lsbeam)=WGHT
            pinjay(IND,ISB)=pinj_ay
            inew=.TRUE.
#ifdef __DEBUG
            zwdepp = zwdep
            zwoldp = zwold
            zwdep = zwdep + WGHT
#endif
         endif
c     
         jloc=njavay(ind,lsbeam)
c
c mhr 19-july-1994
c
         SRD=0.0_R8
c
         indnew=int((xion+xksid)*2.E8_R8)
         call nbi_random_init_pll(ISEED,indnew,iscreen1)
c
         IF(NLHEAVY(LSBEAM)) THEN
            IF(BITIME(1).GT.0.0_R8) THEN
               XZBEAMI=1.0_R8  ! NEW ORBIT
               ZANG=XZBMAY(IND,LSBEAM)  ! RESTORE GYROPHASE
               COSPHO=COS(ZANG)
               SINPHO=SIN(ZANG)
               BITIME(1)=min(BITIME(1),0.999_R8*DELT) ! FORCE SOME STEPS
            ELSE
               XZBEAMI=XZBMAY(IND,LSBEAM)
            ENDIF
            CALL JBSHLD
         ENDIF


C
C%          IF(INWONL.AND.(BITIME(1).GT.0.0)) THEN
C%             NBPLOT=IBPLOT
C%          ENDIF
C
c-------------------------------------------
C  check for splitting based on weight/<avg weight> -- old ions only.
C
         ilsplit=.FALSE.
         wbav_c=wbav(jloc,lsbeam) 
         fact_wbav = 1
C if the code went from a period of time with low power to a period of time with high power, 
C low weight particles from the low power regime would persist; 
C if their presence influences wbav it could cause new particles from the high power 
C regime to be split; 
C the same might happen to old particles if transitioning from a period of high power to a 
C prolonged period of low power.
C
C 
         p_inj_rs=pinjs_res(lsbeam)   ! injected beam power without 
                                ! shine through power at the current time step
                                ! or at the time step when power wasn't equal zero
         if(.not.inew) then
C
C  old orbit eligible for splitting...
C
!     old paricles. Do not let them make a lot of splittings. Increase wbav.
            if(p_inj_rs.gt.0.0_R8.and.
     >           (pinjay(ind,isb).gt.p_inj_rs)) then
!     power drops
!     old paricles from high power regime
               fact_wbav = pinjay(ind,isb)/p_inj_rs
               wbav_c=wbav_c*fact_wbav
            endif
C     
            if(wght.gt.2*wbav_c) then
C
C  w/<w> >= 2
C
  
               if(nuse(isb).lt.ntotptp(isb)) then
C
C  slot available
C
                  ilsplit=.TRUE.
                  isplit=isplit+1
                  jsplit=jsplit+1
                  ! save for later...
                  WGHTAY(ind,lsbeam)=
     >                 WGHTAY(ind,lsbeam)-wbav_c ! save for later
                  WGHT=wbav_c   ! make slot for this now...
c
                  NUSE(ISB)=NUSE(ISB)+1
                  JPTR=NUSE(ISB)
                  IND=NBNDEX(NUSE(ISB),ISB)
                  ind=find_ptcl_ind(numprocd,myidd,ind) !find proper ptcl.ID#
c
                  pinjay(IND,LSBEAM)=pinj_ay !pass to "child" from "parent"
                  NBSCAY(IND,LSBEAM)=NBSCE
                  NBIENAY(IND,LSBEAM)=NBIEN
                  EINJAY(IND,LSBEAM)=EINJ0
               endif
            endif
         else
!     new paricles.Define the renormalization factor for wbav for time dependence of power included to
!
            if(pinjay(ind,isb).gt.0.0_R8.and.
     >           (p_inj_rs.gt.pinjay(ind,isb))) then
!     power up
               fact_wbav = p_inj_rs/pinjay(ind,isb)
            endif
C     

         endif
c-------------------------------------------
c  sums formerly calculated in serial DEPALL have been moved here inside
c  the parallel section...
c
         if(.not.nlfprod(lsbeam)) then
            if(nbsce.eq.0) then
               call errorLog('ORBALL:')
               call errorLog('beam source tracking algorithm error !')
               if(myidd.eq.0) then
                  write(nonlin,*) 
     >                 'orball: beam source tracking algorithm error!'
               endif
               orb_code_error(lsbeam) = orb_code_error(lsbeam) + 1
            else
               zw=WGHT*(delt-bitime(1))
               wts_ien(nbien,nbsce)=wts_ien(nbien,nbsce)+zw
               ein_ien(nbien,nbsce)=ein_ien(nbien,nbsce)+zw*einj0
               dei_ien(nbien,nbsce)=dei_ien(nbien,nbsce)+zw*einj0*einj0
            endif
         endif
c
c  nmc & wnmc sums moved to depall.for 09/04/09  MG
c
c         nmc(jloc,lsbeam)=nmc(jloc,lsbeam)+1.0_R8
c         wnmc(jloc,lsbeam)=wnmc(jloc,lsbeam)+WGHT
C
C------------------------
C  CLEAR SPLITTING FLAGS
C
         ILRECAP=.FALSE.
         if(.not.ilsplit) then
            JPTR=J
         endif
C
C  ORBIT
C
 65      CONTINUE
C
#ifdef __DEBUG
         if(.not.(ilsplit.or.ilrecap)) then
            call zbal_debug(iorb,zwdepp,zwoldp) 
         endif
#endif
         IORB=IORB+1

C%         write(nonlin,*) 'lsbeam, nuse(lsbeam), iorb: ',
C%     >        lsbeam,nuse(lsbeam),iorb

         CALL ORBIT(IORB,ierr,orbsteps)
        ! print*,orbsteps
         orb_code_error(lsbeam) = orb_code_error(lsbeam) + ierr


         if(xicur.eq.epsinv) then
            call infoLog('ORBALL: orbit never inside plasma,IORB=',iorb)
            if(myidd.eq.0) then
               write(nonlin,9011) myidd,iorb
            endif
 9011       format(
     >           ' %ORBALL:  In processor ',i5,': orbit # IORB=',i7,
     >           ' never inside plasma.')
            
         endif
C
C%          IF(INWONL) THEN
C%             IF(NBPLOT.GT.0) CALL NBPLOT_BDY(0)
C%             NBPLOT=-1
C%          ENDIF
C
C  dmc 27 Sep 1994 -- new radial current calculation, based on orbit
C  average position -- not-lost orbits only.  Lost orbits had XISTOP
C  set in subroutine ORBIT
C
         ibflg=nbflg
         if(nbflg.eq.1) then
            if(v2toev*vioncur*vioncur.lt.100.0_R8) ibflg=2
         endif
         if(ibflg.eq.1) then
C
C  use last inside-the-plasma orbit coordinates
C
            XISTOP=xjadep(xicur,thcur,rmjcur,ycur,vioncur,vpllcur,
     >         itrap,iuncon,zxksav,zeiav,zphiav,ixiav,zxhw)
C
C  dmc 18 Sep 1995 -- trapping fraction calculation
C
            call addtrap(wght,itrap,zeiav)
C
C  end dmc 18 Sep
C
         else if(ibflg.eq.2) then
C
            XISTOP=xicur                ! low energy
C
         endif
C
C  only for cx loss is radial current already counted; for normal orbits
C  and other losses add to orbit current here.
C
         if(nbicx.eq.0) then
            call radcur(xistate,xistop,wght,tirorbs)
         endif
C
         IF(IBFLG.EQ.1) THEN
C  ZJAV IS AVG RADIAL LOCATION NBNDEX CALC. SUBR. ORBIT
            JLOC=XJAV+0.5_R8
            JLOC=MAX(LCENTR,MIN(LEP1,JLOC))
            ZRRF=WGHT/WBAV_kill(JLOC,LSBEAM)
            IF(ZRRF.LE.0.33333333333333_R8) THEN
               wnlight(lsbeam)=wnlight(lsbeam) + 1.0_R8
            ENDIF
         ENDIF
C
C  PTCL SURVIVE TO NEXT STEP?
         IF(IBFLG.EQ.1) THEN
C  YES:  SAVE PARAMETERS
CFEB92            ZTSUM3=ZTSUM3+WGHT
c  (for next step compression operator)
c
            if(nlcprb) then
               call ptclsave(xion,th,vion,xksid,
     >            rmjionay(ind,lsbeam),xzionay(ind,lsbeam),
     >            tfluxay(ind,lsbeam),
     >            xmuay(ind,lsbeam),pphiay(ind,lsbeam),
     >            pmechay(ind,lsbeam))
            endif
c
c  traditional state arrays
c
            NJAVAY(IND,LSBEAM)=JLOC
            XIAY(IND,LSBEAM)=XION
            THAY(IND,LSBEAM)=TH
            VAY(IND,LSBEAM)=VION
            XKSIDY(IND,LSBEAM)=XKSID
            GOOSAY(IND,LSBEAM)=GOOSE
            CXPRAY(IND,LSBEAM)=CXSKIP
            FPPRAY(IND,LSBEAM)=FPSKIP
            WGHTAY(IND,LSBEAM)=WGHT
            XISTAY(IND,LSBEAM)=XISTOP
            TDEPAY(IND,LSBEAM)=0.0_R8 ! old orbit now
c
            IF(NLHEAVY(LSBEAM)) XZBMAY(IND,LSBEAM)=XZBEAMI
c
c  compute sawtooth operator -- shadow ptcl list
c
            call pre_sawnbi(IND)
c

C*********************************************************
C         
C***********************************************************

c
C  SUM OF SURVIVING PTCLS, ENERGY
            ENOLDS(LSBEAM)=ENOLDS(LSBEAM)+EP*V2TOEV*WGHTAY(IND,LSBEAM)
            BPHOLDS(LSBEAM)=BPHOLDS(LSBEAM)+BPHION*WGHTAY(IND,LSBEAM)
            BNTOTS(LSBEAM)=BNTOTS(LSBEAM)+WGHTAY(IND,LSBEAM)
C
         ELSE
C  NO:   DELETE; CREATE FREE PTCL SLOT

            WGHTAY(IND,LSBEAM)=0.0_R8

            CALL IDELET(NBNDEX(1,ISB),minb,NUSE(LSBEAM),JPTR)

C  DECREMENT CX-RECAPTURE ION COUNTER IF NEEDED
            IF(ILRECAP) THEN
               IRECAP=IRECAP-1
            else if(ilsplit) then
               isplit=isplit-1
            ELSE
               J=J-1
            ENDIF
         ENDIF
C
C  PROCESS PENDING CX-RECAPTURED IONS NOW IF SPACE PERMITS (AND THERE
C  ARE SUCH IONS)
C
         IF((NUSE(ISB).GE.ntotptp(ISB)).OR.
     +               (NBSPLT.EQ.0).OR.(ILSPLIT)) then
            GO TO 60
         ENDIF
        
C
C  INSERT CX-RECAPTURE ION AT END OF PTCLS LIST
C   KEEP TRACK OF NUMBER OF CX-RECAPTURE IONS

C
         ILRECAP=.TRUE.
         IRECAP=IRECAP+1
C
C  INSERT PREVIOUSLY SPLIT PTCL BACK INTO MAINSTREAM
         BITIME(1)=TBSA(NBSPLT)
         tmfcur=tmfini                  ! time-step magn. factor
         GOOSE=1.0001_R8  ! 1.0000 reserved for newly deposited ions

         dt_adjust = dt_adj0s(lsbeam)
         if(dt_adjust.le.0.0_R8) dt_adjust=1.0_R8

         CXSKIP=1.0_R8
         FPSKIP=1.0_R8
         pinj_ay=pinjbsa(NBSPLT)
         NBSCE=NBSBSA(NBSPLT)
         NBIEN=NBEBSA(NBSPLT)
         EINJ0=EINJBSA(NBSPLT)
         XION=RBSA(NBSPLT)
         TH=THBSA(NBSPLT)
         VION=VBSA(NBSPLT)
         XKSID=XKSBSA(NBSPLT)
         XISTATE=XISBSA(NBSPLT)
         SRD=0.0_R8
         WGHT=WTBSA(NBSPLT)

         NBSPLT=NBSPLT-1
C  CREATE NEW PTCL SLOT IN MAINSTREAM ARRAY
C ---->> Examine following 3 lines carefully

         NUSE(ISB)=NUSE(ISB)+1
         JPTR=NUSE(ISB)
         IND=NBNDEX(NUSE(ISB),ISB)
         ind=find_ptcl_ind(numprocd,myidd,ind) !find proper ptcl.ID#

         NBSCAY(IND,LSBEAM)=NBSCE
         NBIENAY(IND,LSBEAM)=NBIEN
         EINJAY(IND,LSBEAM)=EINJ0
         pinjay(IND,LSBEAM)=pinj_ay
C
C  FOLLOW ORBIT OF THIS ION
C
         GO TO 65
C
 90      CONTINUE

C
C---------------------------
C  DONE FOLLOWING ORBITS OF CURRENT SPECIES
C
         xnorbs(lsbeam) = iorb

C***************************************************
C  MPI Related code
C  (Buffer reduction and particle collection)
C***************************************************
         J_SPL=jsplit

#ifdef __MPI
      stime91=MPI_WTIME()
!
! that combines the values stored by each process into a single value 
! available to master processes.(see nubeam_buf_reduce for MPI_REDUCE)
! 
      puseyp(lsbeam)=nuse(lsbeam)
      call nubeam_buf_reduce !ntr% arrays will not be reduce now
      if(myidd.eq.0) then
         NUSE_total(LSBEAM)=puseyp(lsbeam) + 0.5_R8 !It is a global # MC in storage
                                  ! on master processor(!)
                                  ! because MPI_REDUCE not MPI_ALLREDUCE
                                  ! (see nubeam_buf_reduce).
                                  ! This value for a child processor will be 
                                  ! updated before the next entry to depall
                                  ! by broadcasting of the states.
      endif

      stime92=MPI_WTIME()

      stime9=stime9+(stime92-stime91)
#else
      puseyp(lsbeam)=nuse(lsbeam)
      NUSE_total(LSBEAM)=puseyp(lsbeam) + 0.5_R8
#endif

C---------------------------------------------------
C ** decide here whether to redo orbits **
C    (insert logic here for redo with RF wavefield renormalization)
C    NOTE: this needs MPI modifications so that all CPUs do the same
C    thing based on (presumably) a test of MPI-reduced RF results).
C

      irepeat = (ipass.lt.ipass_num)
C  (add RF test)

      ! do some post-processing on RF-related outputs
      if(orbrzv_rf_option.eq.1 .and. mcrf_on) then
        ! RF-power (watt)
        ! power conversion factor erg -> watt
        zpw = 1.0e-7_r8/(tbm2-tbm1) 
        orbrzv_Prftot(lsbeam) = orbrzv_Prftot(lsbeam)*zpw
        orbrzv_Prf(:,lsbeam) = orbrzv_Prf(:,lsbeam)*zpw

#ifndef __MPI
!#ifdef __DEBUG
        ! density profiles
        orbrzv_nrf(lsbeam,:,:) = orbrzv_nrf(lsbeam,:,:)/(tbm2-tbm1)
!#endif
#endif

        ! RF-power match test
        if(lsbeam .ne. orbrzv_isp_prfmax) then
          irepeat = .false.
        else if(orbrzv_toric_Prftot(lsbeam) <= 1.0e4_r8) then
          irepeat = .false.
        else
          zpw = orbrzv_Prftot(lsbeam)-orbrzv_toric_Prftot(lsbeam)
          zpw = abs(zpw/orbrzv_toric_Prftot(lsbeam))
          if(zpw < zpwr_errmax) irepeat = .false.
        endif

        ! in the case of repeat, 
        if(irepeat) then
          ! adjust toric field renormalization factor and
          zfrnm = sqrt(orbrzv_toric_Prftot(lsbeam)/
     >                 max(1.0_r8, orbrzv_Prftot(lsbeam)))

          ! scale up is tamed to avoid too sensitive change
          if(zfrnm .gt. 1.0_r8) zfrnm = 0.5_r8*(1.0_r8 + zfrnm)

          orbrzv_toric_frnm = zfrnm*orbrzv_toric_frnm

#ifndef __MPI
!#ifdef __DEBUG
          ! --------------------------------------------------------
          ! write RF-debug information
          do i = 1, mj
            rfdbg_pdiag(2:6,i) = rfdbg_pdiag(2:6,i)/
     >                          (rfdbg_pdiag(1,i)+epslon)
            rfdbg_pdiag(7,i) = rfdbg_pdiag(7,i)/(tbm2 - tbm1)
            rfdbg_pdiag(8:9,i) = rfdbg_pdiag(8:9,i)/
     >                          (rfdbg_pdiag(1,i)+epslon)
          enddo

          write(unit=orbrzv_out,fmt=*) '# ipass :', ipass
          write(unit=orbrzv_out,fmt=*) '# ion :', lsbeam,
     >       orbrzv_Prftot(lsbeam), orbrzv_toric_Prftot(lsbeam)
          write(unit=orbrzv_out,fmt=*) '# frnm :',orbrzv_toric_frnm
          do i = 1, mj
            write(unit=orbrzv_out,fmt=*) i,
     >       orbrzv_Prf(i,lsbeam),
     >       rfdbg_pdiag(1:7,i),
     >       orbrzv_nrf(1,1,i),
     >       orbrzv_nrf(1,2,i),
     >       orbrzv_nrf(1,3,i),
     >       rfdbg_pdiag(8,i), rfdbg_pdiag(9,i)
          enddo
          orbrzv_nrf  = 0.0_r8
          rfdbg_pdiag = 0.0_r8
          write(unit=orbrzv_out,fmt=*) ' '
          ! --------------------------------------------------------
!#endif
#endif

          ! zero out Prf variables
          orbrzv_Prftot(lsbeam) = 0.0_r8
          orbrzv_Prf(:,lsbeam)  = 0.0_r8
        endif
      else
        irepeat = .false.
      endif

      if(irepeat.or.(ipass.gt.1)) then
        key_myidd_0=.FALSE.
        if(myidd.eq.0) key_myidd_0=.TRUE.
#ifndef __MPI
!to allow  serial code when debugging MPI
        key_myidd_0=.TRUE.
#endif
         if(key_myidd_0) then
            write(nonlin,*) 
     >           ' specie  XI             TH'//
     >           '             V              Vpll/V   "CPU0 last ion":'
            
            if(iorb.eq.0) then
               write(nonlin,*) ' (species ',isb,'; CPU 0: no orbits).'
               
            else
               WRITE(NONLIN,8001) isb,XION,TH,VION,XKSID
               
            endif
         endif
      endif
C
      if(irepeat) then
         print *, "orball-repeat : ", myidd, irepeat, ipass
         go to 50
      endif

      ! at the final stage of this species....
      ! (moved from nbfinish.f90)
      if(orbrzv_rf_option.eq.1 .and. mcrf_on) then
        if(lsbeam.eq.orbrzv_isp_prfmax) then
          zprft  = orbrzv_Prftot(lsbeam)
          ztprft = orbrzv_toric_Prftot(lsbeam)
          if(min(zprft,ztprft) > 1.0e4_r8) then
            zfrnm = sqrt(ztprft/zprft)
            ! scale up is tamed to avoid too sensitive change
            if(zfrnm .gt. 1.0_r8) zfrnm = 0.5_r8*(1.0_r8 + zfrnm)
            orbrzv_toric_frnm = zfrnm*orbrzv_toric_frnm
          endif

          ! return updated frnm in this time step
          orbrzv_toric_ufrnm = orbrzv_toric_frnm
        endif
      endif



C******************************************************
C    MPI buffer reduction and particle collection ends 
C******************************************************


C------------------------------------------------
C  serial code...

         iorb = xnorbs(lsbeam) + 0.5_r8
         norbtot = norbtot + iorb

#ifdef __MPI
         call infoLog(' In Processor ',myidd)
         call infoLog(' %MC ptcl counts, specie #',LSBEAM)
         call infoLog(' ORIG=',inuse(lsbeam))
         call infoLog(' SPL=',JSPLIT)
         call infoLog(' TOT=',int(puseyp(lsbeam)))
#endif
         if(myidd.eq.0) then
            WRITE(NOUT,8303) myidd,LSBEAM,inuse(lsbeam),
     >           J_SPL,int(puseyp(lsbeam))
            
 8303       format('In Processor ,',i5,' %MC ptcl counts, specie #',i1,
     >           ': ORIG=',i5,' SPL=',i5,' TOT=',i5)
            
         endif
C
C%        CALL ANMODE
C
C  PTCL BALANCE - RELATED OUTPUTS
C
         ZWGHT2=BNTOTS(LSBEAM)
         SFDTBMC(LSBEAM)=(ZWGHT2-ZWGHT1)/DELT

C
#ifdef __DEBUG
         call infoLog('ORBALL:  dep sums: ',sfdepmc(lsbeam))
         call infoLog('         zwdep/delt= ',zwdep/delt)       
         call infoLog('ORBALL:  dep RR sums: ',
     >        sfdeprr(lsbeam)-sforbrr(lsbeam))
         call infoLog('    (zwold-zwght1)/delt',(zwold-zwght1)/delt)   
         if(myidd.eq.0) then
            write(nonlin,*) 'ORBALL: dep sums: ',
     >           sfdepmc(lsbeam),zwdep/delt
            
            write(nonlin,*) ' ORBALL: dep RR sums: ',
     >           sfdeprr(lsbeam)-sforbrr(lsbeam),(zwold-zwght1)/delt
         
         endif
#endif
C
C
C ncx0_orb # of CX tracks collected during orbiting. 
C May not be enough for desire statistics.
C Check if it needs to be improved CX statistics
C
         xncx0_use=dble(ncx0)
         if((xncx0_orb(lsbeam).gt.0).and.
     >        xncx0_use.gt.xncx0_orb(lsbeam)) then
C
            call cx_tracks_need
            if(ncx1.gt.0) then  !ncx1 #CX tracks per CPU
                  
               call cxnrm       ! make a renormalization of the NUTRAV outputs
               call depall_mpi_split(1,ncx1*numprocd,nuseycx)
!     parrallel part
!     allocate random sample arrays;
!     get sorted list of real*8 random numbers
               call nbi_fbm_random_fetch(nuseycx,ierr)
               call nbi_distrsample(nuseycx,ierr)
!
! nuseycx # CX tracks on each CPU to be gathered
               call infolog(' ORBALL -- start CX tracks gathering')
               do icx=1,nuseycx
!     find weight here
                  call beamcx0(icx,ierr)
               enddo
               call infolog(' ORBALL -- done CX tracks gathering')
            endif
         endif
#ifdef __MPI
! reduce all data assotiated with impoved CX statistics
         call nubeam_cxbuf_reduce

#endif

         CALL OUTPTB(IORB,ierr)
         
         orb_code_error(lsbeam) = orb_code_error(lsbeam) + ierr
C
         call orbsmoo                  ! radial smoothing
C
         IF(NBSPLT.GT.0) THEN
            call errorLog('ORBALL: error NBSPLT GT.0 AT END')
            call errorLog('OF TIMESTEP, SPLIT PARTICLES LOST ****')
            call errorLog('NBSPLT= ',NBSPLT)
            if(myidd.eq.0) then
               WRITE(NOUT,7999) NBSPLT
            endif
            
 7999       FORMAT(' ??? ERROR IN ORBALL: NBSPLT=',I2,', .GT.0 AT END'/
     >           '     OF TIMESTEP, SPLIT PARTICLES LOST ****')
            ierr=1
            orb_code_error(lsbeam) = orb_code_error(lsbeam) + ierr
         ENDIF
C
C  END OF BEAM SPECIES LOOP
C
         idum = orb_code_error(lsbeam) + 0.5_R8
         iersum = iersum + idum
         if(iersum.eq.0) then
               
         endif

 100  CONTINUE

#ifdef __MPI
      if(nlbout) then
         call infoLog('ORBALL: lost ptcl. gather')
         if(myidd.eq.0) then
            write(nonlin,*) ' %ORBALL: lost ptcl. gather'
         endif
         call orball_lost_gather
      else
         call infoLog('ORBALL: lost ptcl.  NOT gathered')
      endif
#endif


      
C
      NBS=0
      LSBEAM=0
      ABEAM=0.0_R8
      XZBEAMI=0.0_R8
C
C  update dt_adj0s(...)
C
      do isb=1,nsbeam
         if(nlfprod(isb)) then
            znlimdt=nptclf
         else if(nlheavy(isb)) then
            znlimdt=nptclh
         else
            znlimdt=nptcls
         endif

         znlimdt = max(10.0_R8,(0.01_R8*znlimdt)) - 0.001_R8
         if(xnum_dt_adj0(isb).gt.znlimdt) then
            zadjust = dt_adj0_sums(isb)/xnum_dt_adj0(isb)
            dt_adj0s(isb) = dt_adj0s(isb)*zadjust
         endif

         if(isb.eq.1) then
            dt_adj0=dt_adj0s(isb)
         else
            dt_adj0=min(dt_adj0,dt_adj0s(isb))
         endif
      enddo
C
      flag_dt_reset = ifirst    ! global dt adjust flag for next call...
C
C  EIN_IEN calculation renorm
C
      do ib=1,nbeam
         do ien=1,3
            zw=wts_ien(ien,ib)
            if(zw.eq.0.0_R8) then
               ein_ien(ien,ib)=0.0_R8
               dei_ien(ien,ib)=0.0_R8
            else
               zav=ein_ien(ien,ib)/zw
               ein_ien(ien,ib)=zav
               zvar2=dei_ien(ien,ib)/zw - zav*zav
               dei_ien(ien,ib)=sqrt(max(0.0_R8,zvar2)) ! RMS variance
            endif
         enddo
      enddo
C
C%      NBPLOT=IBPLOT
C
#ifndef __MPI
      ! For serial jobs only:

      if(norbtot.gt.0) then
         if(myidd.eq.0) then
            write(nonlin,*) 
     >        ' specie  XI             TH'//
     >        '             V              Vpll/V     "last ion":'
         endif
         
         do isb=1,nsbeam
            if(nuse(isb).gt.0) then
               ind=nbndex(nuse(isb),isb)
               ind=find_ptcl_ind(numprocd,myidd,ind) !find proper ptcl.ID#
               xion=xiay(ind,isb)
               th=thay(ind,isb)
               vion=vay(ind,isb)
               xksid=xksidy(ind,isb)
               
               if(myidd.eq.0) then
                  WRITE(NONLIN,8001) isb,XION,TH,VION,XKSID
               endif
               

            endif
         enddo
      endif
#endif

 8001 FORMAT(3x,i1,3x,4(1X,1PE14.7))
 
      nmcloss = nmcloss + xmcloss_o + 0.5_R8

!      moved to depall.for 09/04/09 MG
c      call nmcsum 

      iorbtot = norbtot
      call infoLog(' ORBALL: # of orbit = ',iorbtot)


      call cptimr8(ztim2)
	  print*, 'ORBALL Call took ',(ztim2-ztim1)

      ierr = iersum

C-------------------------------------------
C  clear sawtooth flags -- must be reset to get another sawtooth event

      sawflag = .FALSE.
      sawflag2 = .FALSE.

C----------------------------------------------------------------
C   Print execution times for various components 
C   after all species loops (MPI related)
C----------------------------------------------------------------

#ifdef __MPI
      ttime32=MPI_WTIME()
      if (myidd.eq.0) CPBORB = CPBORB + (ttime32-ttime31) 
#else
      CPBORB = CPBORB + (ztim2-ztim1)
#endif

#ifdef __MPI

      if (impitime.gt.0) then
   
       call warnLog(' ')
       call warnLog('IN NBSTART: myidd= ',myidd)
       call warnLog('Total time in nbstart= ',ttime12-ttime11)
       call warnLog('Total time for file read= ',stime2-stime1)
       call warnLog(' ')
       call warnLog('IN DEPALL: myidd= ',myidd)
        ztime1=ttime22-ttime21
        ztime2=stime4+stime5+stime6+stime7+stime8+stime10
       call warnLog('Total parallel processing time= ',ztime2)
       call warnLog('Total time for calculation= ',ztime1-ztime2)
       call warnLog('Total time in depall/add_track= ',stime7)
       call warnLog('Time for MPI dpt_gather= ',stime10)
       call warnLog('Total time in depall= ',ztime1)
       call warnLog(' ')
       call warnLog('IN ORBALL: myidd= ',myidd)
        ztime3=(ttime32-ttime31)-stime9
       call warnLog('Time for MPI BUF Reduce= ',stime9)
       call warnLog('Time for calculation= ',ztime3)
       call warnLog('Total time in orball= ',ttime32-ttime31)
       call warnLog('Total time in orball (using cptim)= ',
     >      ztim2-ztim1)
       call warnLog(' ')
        ztime4=(ttime12-ttime11)+(ttime22-ttime21)+
     +         (ttime32-ttime31) 
       call warnLog('IN NBSTART, DEPALL, ORBALL: myidd= ',myidd)
       call warnLog('Total time = ',ztime4)
       call warnLog(' ')
      endif

#else
#endif

      go to 9999

C -- allocation error --
 6100 continue
      ierr=777
      call errorLog('orball: allocation failure,too many particles?')
      if(myidd.eq.0) then
         write(nonlin,'(a)') '?orball: allocation failure, are '
     >        //'you using too many particles?'
      endif
      
 9999 continue



      if(itmp_file) then
         open(unit=lunres,file=zorball_tmp_filename,status='OLD')
         close(unit=lunres,status='DELETE')
      endif
      return

      CONTAINS
 
      subroutine cx_tracks_need

! find out how many tracks (ncx1) need to be evaluted on each CPU to improve CX statistics
! ncx0 can be adjusted here to be a multiple of numprocd = ncx0_use
      integer :: ncx_rs
      real *8 :: zrem

      if(xncx0_orb(lsbeam).lt.1._r8) then
         ncx1 = 0
         return
      endif
      ncx_rs = int(xncx0_use - xncx0_orb(lsbeam))
      
#ifdef __MPI
C
C--------------------------------------------------------
C
         
      if(ncx_rs>numprocd) then
         ncx1 = ncx_rs/numprocd
         if(mod(ncx_rs,numprocd).ne.0) then
            ncx1 = ncx1+1       ! round up
         endif
      else
         ncx1 = 1
      endif
      ncx0_use = int(xncx0_orb(lsbeam))+ncx1*numprocd
         
#else
      ncx1 = ncx_rs
      ncx0_use = int(xncx0_use)
#endif
      end subroutine cx_tracks_need




        subroutine orball_loclist_write
           character*6 zcpid
           integer :: inum, iost
           
           ! write ptcl arrays in tmp binary file -- do not include
           ! sawtooth variables; sawtooth operator already executed
           ! if needed.

           write(zcpid,'(i5)') myidd
           zorball_tmp_filename = trim(workpath)//
     >          trim(runid)//'_orball_cpu'//trim(adjustl(zcpid))//'.tmp'

           open(unit=lunres,file=zorball_tmp_filename,status='UNKNOWN',
     >          access='SEQUENTIAL',form='UNFORMATTED',iostat=iost)
           if(iost > 0) then
              call errorLog(" ?orball_loclist_write: file open failed")
              
              call bad_exit("orball_loclist_write : file open failed")
           endif

           inum = ntotptp(lsbeam)
           write(lunres) inum,nuse(lsbeam)
           ! kjm addup
           write(lunres) puseyp(lsbeam),nuse_total(lsbeam)

           write(lunres) nbndex(1:inum,lsbeam)

           write(lunres) nbscay(1:inum,lsbeam)
           write(lunres) nbienay(1:inum,lsbeam)
           write(lunres) njavay(1:inum,lsbeam)

           write(lunres) cxpray(1:inum,lsbeam)
           write(lunres) fppray(1:inum,lsbeam)
           write(lunres) goosay(1:inum,lsbeam)
           write(lunres) einjay(1:inum,lsbeam)
           write(lunres) pmechay(1:inum,lsbeam)
           write(lunres) pinjay(1:inum,lsbeam)
           write(lunres) pphiay(1:inum,lsbeam)
           write(lunres) rmjionay(1:inum,lsbeam)
           write(lunres) tdepay(1:inum,lsbeam)
           write(lunres) tfluxay(1:inum,lsbeam)
           write(lunres) thay(1:inum,lsbeam)
           write(lunres) vay(1:inum,lsbeam)
           write(lunres) wghtay(1:inum,lsbeam)
           write(lunres) xiay(1:inum,lsbeam)
           write(lunres) xistay(1:inum,lsbeam)
           write(lunres) xksidy(1:inum,lsbeam)
           write(lunres) xmuay(1:inum,lsbeam)
           write(lunres) xzbmay(1:inum,lsbeam)
           write(lunres) xzionay(1:inum,lsbeam)

           close(unit=lunres)
        end subroutine orball_loclist_write

        subroutine orball_loclist_read
           integer :: inum, iost

           open(unit=lunres,file=zorball_tmp_filename,status='OLD',
     >          access='SEQUENTIAL',form='UNFORMATTED',iostat=iost)
           if(iost > 0) then
             print *, " ?orball_loclist_read : file open failed"
             call bad_exit("orball_loclist_read : file open failed")
           endif

           read(lunres) inum,nuse(lsbeam)
           ntotptp(lsbeam) = inum
           ! kjm addup
           read(lunres) puseyp(lsbeam),nuse_total(lsbeam)

           read(lunres) nbndex(1:inum,lsbeam)

           read(lunres) nbscay(1:inum,lsbeam)
           read(lunres) nbienay(1:inum,lsbeam)
           read(lunres) njavay(1:inum,lsbeam)

           read(lunres) cxpray(1:inum,lsbeam)
           read(lunres) fppray(1:inum,lsbeam)
           read(lunres) goosay(1:inum,lsbeam)
           read(lunres) einjay(1:inum,lsbeam)
           read(lunres) pmechay(1:inum,lsbeam)
           read(lunres) pinjay(1:inum,lsbeam)
           read(lunres) pphiay(1:inum,lsbeam)
           read(lunres) rmjionay(1:inum,lsbeam)
           read(lunres) tdepay(1:inum,lsbeam)
           read(lunres) tfluxay(1:inum,lsbeam)
           read(lunres) thay(1:inum,lsbeam)
           read(lunres) vay(1:inum,lsbeam)
           read(lunres) wghtay(1:inum,lsbeam)
           read(lunres) xiay(1:inum,lsbeam)
           read(lunres) xistay(1:inum,lsbeam)
           read(lunres) xksidy(1:inum,lsbeam)
           read(lunres) xmuay(1:inum,lsbeam)
           read(lunres) xzbmay(1:inum,lsbeam)
           read(lunres) xzionay(1:inum,lsbeam)

           close(unit=lunres)
        end subroutine orball_loclist_read
      
      END
c=====================================================================
#ifdef __DEBUG
      subroutine zbal_debug(iorb,zwdep,zwold)

      use nbi_com
      use nbi_mpi_com
      use logmod
      implicit NONE

      integer, intent(in) :: iorb
      real*8, intent(in) :: zwdep,zwold

      !  debug: compute interim ptcl balance

      INTEGER, PARAMETER :: R8=SELECTED_REAL_KIND(12,100)
      real*8 :: zbal,zmax,bsumint,zsumth

      integer, parameter :: ifreq = 100
      integer, parameter :: imin = 100

      integer :: itest

      !-------------------------------

      if(iorb.lt.imin) return
      itest = iorb/ifreq
      itest = itest*ifreq
      if(itest.ne.iorb) return

      sfdtbmc(lsbeam)=bntots(lsbeam)-zwold
      zsumth=BSUMINT(SORBTHS(1:mj,LSBEAM))

      zbal = zwdep
     >        -BXFLWS(LSBEAM)-zsumth
     >        -BXFSHS(LSBEAM)
     >        -SFCXESC(LSBEAM)-SFCXRR(LSBEAM)-BXRPLS(LSBEAM)
     >        -SFDTBMC(LSBEAM)
      zbal=abs(zbal)
      zmax=max(1.0_R8,abs(zwdep))
      zmax=max(zmax,abs(BXFLWS(LSBEAM)))
      zmax=max(zmax,abs(zsumth))
      zmax=max(zmax,abs(BXFSHS(LSBEAM)))
      zmax=max(zmax,abs(SFCXESC(LSBEAM)))
      zmax=max(zmax,abs(SFCXRR(LSBEAM)))
      zmax=max(zmax,abs(BXRPLS(LSBEAM)))
      zmax=max(zmax,abs(SFDTBMC(LSBEAM)))

      call infoLog('orball: iorb=',iorb)
      call infoLog('orball: ratio = ',zbal/zmax)
      if(myidd.eq.0) then
         write(nonlin,*) ' iorb=',iorb,' ratio = ',zbal/zmax
      endif
      return
      end
#endif
