
	subroutine gpu_orbit_input(xizones,ierr)

!	This routine packages the variables that need to be passed to the gpu_orbit routines


      use nbi_com
      use nbi_random_pll
      use nbi_random
      use nbi_mpi_com
      use nbi_fbm_random
      use orbrzv
      use logmod
      use nbspline_mod, only: xi2,th2,hxi,hxii,hth,hthi,xspani, &
      thspani,nbsii,nbsjj,nbrk,xi_brk,nint,next,rspl,rsplx,zspl,zsplx,bspl,jspl,thmin,thmax,nxi,nth
      use nbatom_mod
      use xplasma_obj_instance
      use xplasma_calls, only: eq_rzdist

! Bfield Parameters

!	ZDXI = DXIZB(LCENTR)		!USE TRANSP DXI FOR ZONE BOUNDARIES
!	IXIMAX = NZNBLO + LCENTR
!	ZXIMAX = XIBLO(IXIMAX)
!	INXI = 1 + int(ZXIMAX/ZDXI + 0.5_R8) !number of XI zones
!   ITHETA = MTHETA	! number of theta zones
!   ZXI ! array of extrapolated xi values, zxi(mtbl)
!   ZTHETA ! array of theta values, ztheta(mtbl)
!   SPLG ! splg(4,mtbl) MHD G spline coefficients
!   SPLIMHD ! splimhd(4,mtbl) MHD I(xi) spline coeff
!   SPLIOT ! spliot(4,mtbl) MHD iota(bar) SPLINE COEFF
!   SPLPHI ! splphi(4,mtbl) MHD RADIAL ELECTRIC POTENTIAL
!   BSPL ! bspl(4,nth,nxi)
!   JSPL ! jspl(4,nth,nxi)
!	XI2 ! xi(nxi) from nbspline_mod, xigrid in gpu_orbit.cu
!	TH2 ! th(nth) from nbspline_mod thetagrid in gpu_orbit.cu
!	xizones ! XI(mjeqv,miz) from nbi_com
! 	FBZ	! fbz(mj,miz) ACTUAL/EXTERNAL TOR. B FIELD AT ZN/BDY J,IZ
! 	BTILTC ! btiltc(5,mimxbz) coarse B-field on 2d grid
! 	NTHZSM ! nthzsm(nznbmr)  # theta zones total inside indicated row
!	THBDY ! = [thbdy0,thbdy1]
!	dxi0  ! dxizc(mjeqv) DIFFERENTIAL ELEMENTS OF XI

	implicit none

	  integer, intent(out) :: ierr      ! status code (0=OK)
	  real*8, intent(in),dimension(mjeqv,miz) :: xizones ! xi from nbi_com
	  integer,dimension(32) :: gpuiparams ! integer parameters to be passed to gpu orball
	  real*8,dimension(15) :: gpudparams ! double parameters to be passed to gpu orball
	  real*8, dimension(8) :: spacingparams ! spline spacing params
	  real*8,dimension(2) :: thbdy
	  integer :: gpulun,i1,j1,iwarn,k1,l1

	  real*8,allocatable,dimension(:,:) :: Xi_map
	  real*8,allocatable,dimension(:,:) :: Theta_map
	  real*8,allocatable,dimension(:,:,:) :: psispline_out,gspline_out,phispline_out
	  real*8,allocatable,dimension(:,:) :: limiter_distance_map
	  real*8 :: vertdispl_temp
	  real*8 :: zjac(2,2)
	  real*8 max_energy
	  real*8 :: rmajor_temp
	  real*8 :: rho_out
	  real*8 :: theta_out
	  real*8 :: zdist_gc(1),zz_R(1),zz_Z(1)
	  real*8 zz_Rlim(1),zz_Zlim(1),zz_philim(1),zz_phi(1)
	  real*8 :: g, dpsi(1:2),dpdRR,dpdZZ,dpdRZ
	  real*8 :: temp_energy,zintrp
	  real*8 :: ztim1,ztim2,ztim3,ztim4
	  integer :: ind,indp1,irange
	  integer :: iopt

	  print*, "xi2(nbrk) = ", xi2(nxi)


! Setup to Fill the r,z -> Xi,Theta maps
	if(allocated(Xi_map)) deallocate(Xi_map)
	if(allocated(Theta_map)) deallocate(Theta_map)
	if(allocated(limiter_distance_map)) deallocate(limiter_distance_map)

	allocate(Xi_map(1:orbrzv_nr,1:orbrzv_nz),Theta_map(1:orbrzv_nr,1:orbrzv_nz))
	allocate(limiter_distance_map(1:orbrzv_nr,1:orbrzv_nz))

	zz_phi(1) = 0

! Use xplasma call xs_fxget1_r8 to generate the r,z -> xi,theta map
	do i1=1,orbrzv_nr
		do j1=1,orbrzv_nz
			rmajor_temp = dble(i1-1)*orbrzv_dr+orbrzv_rmin
			vertdispl_temp = dble(j1-1)*orbrzv_dz+orbrzv_zmin
			call xs_fxget1_r8(0.01_r8,0.0_r8,rmajor_temp,vertdispl_temp, &
																& rho_out,theta_out,iwarn)
			Xi_map(i1,j1) = rho_out
			Theta_map(i1,j1) = theta_out
			!print*,"Position:",i1-1,j1-1
			!print*,"r=",rmajor_temp,"z=",vertdispl_temp
		!	print*,"Xi=",rho_out,"Theta=",theta_out
			zz_R(1) = rmajor_temp/100._R8
			zz_Z(1) = vertdispl_temp/100._R8
			call eq_rzdist(1,zz_R,zz_Z,zz_phi,zdist_gc,iopt,zz_Rlim,zz_Zlim,zz_philim,ierr)
			limiter_distance_map(i1,j1) = zdist_gc(1)*100._R8

			!call nbmomry(Xi_map(i1,j1),Theta_map(i1,j1),0,rmajor_temp,vertdispl_temp,zjac)

			!rmajor_temp = (dble(i1-1)+0.5_r8)*orbrzv_dr+orbrzv_rmin
			!vertdispl_temp = (dble(j1-1)+0.5_r8)*orbrzv_dz+orbrzv_zmin

			!call EZspline_interp2_r8(orbrzv_g_spo, rmajor_temp, vertdispl_temp, g, ierr)
		  !call EZspline_gradient2_r8(orbrzv_psi_spo, rmajor_temp, vertdispl_temp, dpsi, ierr)
		  !call EZspline_derivative2_r8(orbrzv_psi_spo, 2, 0, rmajor_temp, vertdispl_temp, dpdRR, ierr)
 			!call EZspline_derivative2_r8(orbrzv_psi_spo, 0, 2, rmajor_temp, vertdispl_temp, dpdZZ, ierr)
      !call EZspline_derivative2_r8(orbrzv_psi_spo, 1, 1, rmajor_temp, vertdispl_temp, dpdRZ, ierr)
			!print*,"r=",rmajor_temp,"z=",vertdispl_temp
			!print*,"Psispline = ",orbrzv_psi_spo%fspl(1,i1,j1), &
			!orbrzv_psi_spo%fspl(2,i1,j1), &
			!orbrzv_psi_spo%fspl(3,i1,j1), &
			!orbrzv_psi_spo%fspl(4,i1,j1)
		!	print*,"gspline = ",orbrzv_g_spo%fspl(1,i1,j1), &
			!orbrzv_g_spo%fspl(2,i1,j1), &
			!orbrzv_g_spo%fspl(3,i1,j1), &
			!orbrzv_g_spo%fspl(4,i1,j1)
			!print*,"dPsi,g = ",dpsi(1),dpsi(2),g
			!print*,"ddPsi = ",dpdRR,dpdZZ,dpdRZ
			!print*,"Bfield = ", dpsi(2)/rmajor_temp*orbrzv_bpsign, &
			! -dpsi(1)/rmajor_temp*orbrzv_bpsign, &
			! g/(rmajor_temp*rmajor_temp)

			!print*,"------------------------------------------------------------"
		enddo
	enddo

	  max_energy = 0.0

! Find the maximum energy used for cross sections
	  do i1=1,nbnsvmx
	  	do j1=0,nsbeam
	  		max_energy = max(max_energy,bnsves(i1,j1))
	  	enddo
	  enddo

	  psispline_out = orbrzv_psi_spo%fspl(1,1,1)
	  gspline_out = orbrzv_g_spo%fspl(1,1,1)
	  phispline_out = orbrzv_phi_spo%fspl(1,1,1)



!	  do i1=1,ng
!	  	do j1=1,nfbznsi/100
!	  		do k1 = 1,51
!	  			do l1 = 1,nsbeam
!	  				temp_energy = max_energy*(((10.0_r8)**(4.0_r8*dble(k1-1)/512.0_r8-4.0_r8))-0.0001_r8)
!	  				irange = k1/(512/4)
!	  				call cxnsum_intrp1(temp_energy,l1,irange,ind,indp1,zintrp)
!	  				print*,"------------------------------------------------------------"
!	  				print*,"cxn_thcx_ha(",temp_energy,i1-1,j1-1,ind-1,l1-1,")"
!	  				print*,(1.0_R8-zintrp)*cxn_thcx_ha(i1,j1,ind,l1)+zintrp*cxn_thcx_ha(i1,j1,indp1,l1)
!	  				print*,"------------------------------------------------------------"
!	  			enddo
!	  		enddo
!	  	enddo
!	  enddo


	  gpuiparams(1) = mjeqv
	  gpuiparams(2) = miz
	  gpuiparams(3) = nsbeam
	  gpuiparams(4) = mibs
	  gpuiparams(5) = nxi
	  gpuiparams(6) = nth
	  gpuiparams(7) = mib
	  gpuiparams(8) = ledge
	  gpuiparams(9) = lcentr
	  gpuiparams(10) = nzones
	  gpuiparams(11) = mig
	  gpuiparams(12) = minb
	  gpuiparams(13) = nsnccw

	  if(nlsym2b) then
	  	gpuiparams(14) = 1
	  else
	  	gpuiparams(14) = 0
	  endif

	  gpuiparams(15) = mimxbz
	  gpuiparams(16) = mimxbzf
	  gpuiparams(17) = nznbmr
	  gpuiparams(18) = nznbmri
	  gpuiparams(19) = nznbmr+lcentr-1
	  gpuiparams(20) = nxtbzn
	  gpuiparams(21) = ndifbe
	  gpuiparams(22) = ng
	  gpuiparams(23) = orbrzv_nr
	  gpuiparams(24) = orbrzv_nz

	  gpuiparams(25) = nint

	  gpuiparams(26) = next
	  gpuiparams(26) = nbrk
	  gpuiparams(27) = nsjdotb

	  gpuiparams(28) = nbsii(1)
	  gpuiparams(29) = nbsjj(1)
	  gpuiparams(30) = orbrzv_bpsign
	  gpuiparams(31) = cxn_zmin
	  gpuiparams(32) = cxn_zmax

	  gpudparams(1) = xminbm
	  gpudparams(2) = thbdy0
	  gpudparams(3) = thbdy1
	  gpudparams(4) = fact_wbav
	  gpudparams(5) = fac_e
	  gpudparams(6) = orbrzv_rmin
	  gpudparams(7) = orbrzv_rmax
	  gpudparams(8) = orbrzv_zmin
	  gpudparams(9) = orbrzv_zmax
	  gpudparams(10) = fppcon
	  gpudparams(11) = cxpcon
	  gpudparams(12) = max_energy
	  gpudparams(13) = orbrzv_dr
	  gpudparams(14) = orbrzv_dz
	  gpudparams(15) = xbmbnd

	  spacingparams(3) = hxi(1)
	  spacingparams(4) = hxii(1)
	  spacingparams(1) = hth(1)
	  spacingparams(2) = hthi(1)
	  spacingparams(6) = xspani
	  spacingparams(5) = thspani



	  print*, "delt = ",delt,"fact_wbav = ",fact_wbav
	  print*, "zmin = ",orbrzv_zmin-orbrzv_dz,"rmin = ",orbrzv_rmin-orbrzv_dr

		call cptimr8(ztim1)
	  call setup_gpu_fields(orbrzv_psi_spo%fspl,orbrzv_g_spo%fspl, &
	  orbrzv_phi_spo%fspl,Xi_map,Theta_map,omegag,rhob,owall0, &
	  ovol02,bn0x2p,bv0x2p,be0x2p,xzbeams,bmvol,bn002,xninja, &
	  viona,vcxbn0,vbtr2p,wbav,xiblo,dxi,te,ti,einjs,cfa,dfa,efa,vpoh, &
	  xjbfac,vmin,velb_fi,difb_fi,velb_bi,difb_bi,fdifbe,edifbe,rspl,zspl, &
	  rsplx,zsplx,xi2,th2,nlfprod,limiter_distance_map,ympx, &
	  bnsvtot,bnsvexc,bnsviif,bnsvief, &
	  bnsvizf,bnsvcxf,bbnsvcx,bbnsvii,cxn_thcx_a,cxn_thcx_wa, &
	  cxn_thii_wa,cxn_thcx_ha,cxn_thii_ha,cxn_bbcx,cxn_bbii,btfus_dt, &
	  btfus_d3,btfus_ddn,btfus_ddp,btfus_td,btfus_tt,btfus_3d,bnsves, &
	  nbnsve,lbnsve,nbnsver,nthzsm,gpudparams,gpuiparams,	&
	  spacingparams,ierr)
	  call cptimr8(ztim2)

	    print*, 'GPU Setup Fields Call took ',(ztim2-ztim1)

    call cptimr8(ztim3)

	  call orbit_gpu(nbndex,nbscay,nbienay,pinjay, &
	  einjay,xiay,thay,vay,xksidy,wghtay,xzbeama, &
	  abeama,rmjionay,xzionay,cxpray,fppray,delt,ierr)

	  call cptimr8(ztim4)

	    print*, 'GPU Orbit Call took ',(ztim4-ztim3)

!      gpulun=50
!	  open(gpulun,FILE = 'DATA01.TXT')
! 105    FORMAT(12(1X,G10.4))
!	  do j1=1,mibs
!	  	do i1=1,minb
!	  		XION = XIAY(i1,j1)
!	  		TH = THAY(i1,j1)
!	  		XZBEAMI = 1.0_R8
!	  		XKSID = XKSIDY(i1,j1)
!	  		VION = VAY(i1,j1)
!	  		call nbigc(1)
!			write(gpulun,*) XZBEAMI,(i1-1),(j1-1)
!            write(gpulun,105) XKSIDY(i1,j1), &
!            VAY(i1,j1),PINJAY(i1,j1), &
!           EINJAY(i1,j1),XIAY(i1,j1), &
!            THAY(i1,j1),WGHTAY(i1,j1), &
!            RMJIONAY(i1,j1),XZIONAY(i1,j1), &
!            NBNDEX(i1,j1),NBSCAY(i1,j1), &
!           NBIENAY(i1,j1)
!		enddo
!	  enddo
! 	  print *, ' GPU ORBIT Test... '
!	  close(gpulun)

!      call orbit_gpu(NBNDEX,NBSCAY, &
!       	NBIENAY,PINJAY, &
!      	EINJAY,XIAY,THAY, &
!       	VAY,XKSIDY,WGHTAY, &
!       	XZIONAY,ABEAMA, &
!       	RMJIONAY,XZIONAY, &
!       	CXPRAY,FPPRAY, &
!       	fbz,bzxr,btiltc, thbdy, &
!       	bspl,jspl,xi2,th2, &
!		rspl,rsplx,zspl,zsplx, &
!       	xizones,thzons,XIBLO, &
!       	nthzsm,dxi,dxizc,dxizb, &
!       	PHIPRG,spliot,splg,splphi, &
!       	splimhd, &
!       	plflxi, omegag,gxtabl, &
!       	ti,qgeo,bpjxbs,bphjxbs, &
!       	tqjxbs,tqjxb_nb,pbephis, &
!       	tirtots,fpixbts, &
!       	gpudparams, &
!       	gpuiparams,spacingparams,ierr)
       	return
       	      contains
         subroutine cxnsum_intrp1(zeova,isbe,irng,ie,iep,zifac)


         REAL*8, intent(in) :: zeova      ! E/A for table
         integer, intent(in) :: isbe    ! table id (0 or by beam specie)
         integer, intent(inout) :: irng ! table energy range sector
         integer, intent(out) :: ie,iep ! bracketing energy indices
         REAL*8, intent(out) :: zifac     ! linear interpolation factor

!---------------

         integer inume,inumr,imin,imax,inum,inc
         REAL*8 zea,zemin,zemax

!---------------

         inumr=nbnsver(isbe)            ! no. of energy range sectors
         inume=nbnsve(isbe)             ! tot. no. energy grid pts

         zea=zeova
         if(zea.gt.bnsves(inume,isbe)) then
            zea=bnsves(inume,isbe)
         endif
!
!  guard low energy limit
!
         zea=max(bnsves(1,isbe),zea)
!
!  guard sector range
!
         irng=max(1,min(inumr,irng))
!
         do
            if(irng.eq.1) then
               imin=1
            else
               imin=lbnsve(irng-1,isbe)
            endif
            imax=lbnsve(irng,isbe)
!
            zemin=bnsves(imin,isbe)
            zemax=bnsves(imax,isbe)
!
            if(zea.lt.zemin) then
               irng=irng-1
               cycle
!
            else if(zea.gt.zemax) then
               irng=irng+1
               cycle
!
            else
!  in range...
               inum=imax-imin
               inc=min((inum-1),int(inum*(zea-zemin)/(zemax-zemin)))
!
               ie=imin+inc
               iep=ie+1
!
               zifac=(zea-bnsves(ie,isbe))/(bnsves(iep,isbe)-bnsves(ie,isbe))
!
               exit
!
            endif
         end do
!
         end subroutine cxnsum_intrp1

       	end

