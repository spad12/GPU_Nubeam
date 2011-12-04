#include "fpreproc/library_names.h"
subroutine nbspline(nonlin,zxi,inxi,inth,ierr)
 
  use nbspline_mod
  use xplasma_calls
 
 
!============
! idecl:  explicitize implicit REAL declarations:
      IMPLICIT NONE
      INTEGER, PARAMETER :: R8=SELECTED_REAL_KIND(12,100)
!============
  integer, intent(in) :: nonlin    ! LUN for messages
  integer, intent(in) :: inxi      ! no. of xi surfaces
  REAL*8, intent(in) :: zxi(inxi)    ! xi surfaces
  integer, intent(in) :: inth      ! number of th points wanted
 
  integer, intent(out) :: ierr     ! status code; 0=OK
 
  !  the following are for the Jacobian evaluation
  !  in the main code these are QGEO(LCENTR) and TFLUX
 
  !----------------------------------------------------
  !  set up beam code's R(theta,xi), Z(theta,xi) bicubic splines
  !  using xplasma information
  !
  !----------------------------------------------------
 
  REAL*8 zdum,d1mach,zdxi
  REAL*8 zcm,zcm3,zgauss
  integer i,j,imin
 
  REAL*8, dimension(:), allocatable :: bcr0,bcr1,bcr1a,bcrx
  REAL*8, dimension(:), allocatable :: bcz0,bcz1,bcz1a,bczx
 
  REAL*8 zxi_axis,zxi_bdy,zxi_bdyp,zxi_far
  REAL*8, dimension(:), allocatable :: zrho,zchi,zphi,zdpsidrho,zrhoj
  REAL*8, dimension(:,:), allocatable :: zans
 
  REAL*8, dimension(:), allocatable :: zwk
  REAL*8 :: zdelx_sm,zdelth_sm,zpio6
 
  integer :: ifcns(2)=(/0,0/)
  integer iwantv(6)
  integer iwantd(6)
  integer :: id_bmod(1)=(/0/)
  integer :: id_dbinvdt(1)=(/0/)
  integer :: id_psi=0,id_dgdt=0
  integer idum1,idum2
 
  REAL*8 ztrght2  ! legacy extrapolation function
 
  data iwantv /1,0,0,0,0,0/
  data iwantd /0,1,0,0,0,0/
 
  !----------------------------------------------------
 
  ierr = 0
 
  !  setup arrays for nbspline_mod -- indep. coordinates & spline arrays
  !  find break point at plasma boundary
 
  !  domain extrema...
 
  call eq_rholim(zxi_axis,zxi_bdy)
  zxi_bdyp=(1.0_R8+d1mach(4))*zxi_bdy
  if(zxi_bdyp.eq.zxi_bdy) zxi_bdyp=(1.0_R8+d1mach(4))*zxi_bdy
  zxi_far=zxi(inxi)
 
  !  will convert m to cm, T to gauss
 
  zcm=100.0_R8
  zcm3=zcm*zcm*zcm
  zgauss=1.0E4_R8
 
  !  allocate the grid arrays
 
  if(allocated(xi)) deallocate(xi)
  if(allocated(th)) deallocate(th)

  if(allocated(xi2)) deallocate(xi2)
  if(allocated(th2)) deallocate(th2)
 
  nxi=inxi
  nth=inth

  print*, "nxi = ",nxi,"nth = ",nth
 
  allocate(xi(nxi),th(nth))
  allocate(xi2(nxi),th2(nth))
 
  !  work arrays for fetching data from xplasma
  !  data fetched in rows of 1:nth
 
  allocate(zrho(nth),zchi(nth),zphi(nth))
  zphi=0
 
  allocate(zans(nth,2))
 
  !  set up the common grids
 
  xi=zxi
  xi2 = xi
  do i=1,nth
     zchi(i)=(thmin*(nth-i)+thmax*(i-1))/(nth-1)
     th(i)=zchi(i)
     th2(i) = zchi(i)
  enddo
 
  !  find the plasma boundary
  !  the grid extends into an extrapolated region
 
  zdum=abs(zxi_axis-zxi_bdy)
  do i=1,nxi
     if(abs(xi(i)-zxi_bdy).lt.zdum) then
        zdum=abs(xi(i)-zxi_bdy)
        imin=i
     endif
  enddo
 
  !  boundary...
 
  nbrk=imin
  xi_brk=xi(nbrk)
 
  !  number inside
 
  nint=nbrk
 
  !  number outside
 
  next=nxi-nbrk+1
 
  !  these quantities aid lookup and interpolation
 
  hxi=(xi(nxi)-xi(1))/(nxi-1)
  hxii=1/hxi
  hth=(thmax-thmin)/(nth-1)
  hthi=1/hth
 
  xspani=1/(xi(nxi)-xi(1))
  thspani=1/(th(nth)-th(1))
 
  !----------------------------------------------------
  !  allocate the bicubic spline arrays
 
  !  (R,Z)
 
  if(allocated(rspl)) deallocate(rspl)
  if(allocated(rsplx)) deallocate(rsplx)
  if(allocated(zspl)) deallocate(zspl)
  if(allocated(zsplx)) deallocate(zsplx)
 
  allocate(rspl(4,nth,nint),zspl(4,nth,nint))
  allocate(rsplx(4,nth,next),zsplx(4,nth,next))
 
  !  (1/B), "J"
 
  if(allocated(bspl)) deallocate(bspl)
  if(allocated(jspl)) deallocate(jspl)
 
  allocate(bspl(4,nth,nxi),jspl(4,nth,nxi))
 
  !----------------------------------------------------
  !  get XPLASMA ids
 
  !  required:
  call eq_gfnum('R',ifcns(1))
  call eq_gfnum('Z',ifcns(2))
  call eq_gfnum('BMOD',id_bmod(1))
  call eq_gfnum('PSI',id_psi)
 
  ierr=0
  if((ifcns(1).eq.0).or.(ifcns(2).eq.0)) then
     write(nonlin,*) ' ?nbspline: xplasma "R" and "Z" functions not found.'
     ierr=ierr+1
  endif
 
  if(id_bmod(1).eq.0) then
     write(nonlin,*) ' ?nbspline: xplasma "Bmod" function not found.'
     ierr=ierr+1
  endif
 
  if(id_psi.eq.0) then
     write(nonlin,*) ' ?nbspline: xplasma psi(rho) function not found.'
     ierr=ierr+1
  endif
 
  if(ierr.gt.0) return
 
  !----------------------------------------------------
  !  (R,Z) boundary conditions (d/dxi at domain extrema), periodic in theta
  !  enforce periodicity of BC
 
  !  more on boundary conditions:  splines other than (R,Z) use
  !  "not a knot" in xi and periodic in theta
 
  allocate(bcr0(nth),bcr1(nth),bcr1a(nth),bcrx(nth))
  allocate(bcz0(nth),bcz1(nth),bcz1a(nth),bczx(nth))
 
  !  (R,Z) BC on axis
 
  zrho=zxi_axis
  call eq_hrhochi(nth,zrho,zchi,2,ifcns,iwantd,nth,zans,ierr)
  if(ierr.ne.0) goto 9999
  bcr0=zans(1:nth,1)*zcm
  bcz0=zans(1:nth,2)*zcm
 
  !  (R,Z) BC at plasma bdy (from inside)
 
  zrho=zxi_bdy
  call eq_hrhochi(nth,zrho,zchi,2,ifcns,iwantd,nth,zans,ierr)
  if(ierr.ne.0) goto 9999
  bcr1=zans(1:nth,1)*zcm
  bcz1=zans(1:nth,2)*zcm
 
  !  (R,Z) BC at plasma bdy (from outside)
 
  zrho=zxi_bdyp
  call eq_hrhochi(nth,zrho,zchi,2,ifcns,iwantd,nth,zans,ierr)
  if(ierr.ne.0) goto 9999
  bcr1a=zans(1:nth,1)*zcm
  bcz1a=zans(1:nth,2)*zcm
 
  !  (R,Z) BC at domain bdy
 
  zrho=zxi_far
  call eq_hrhochi(nth,zrho,zchi,2,ifcns,iwantd,nth,zans,ierr)
  if(ierr.ne.0) goto 9999
  bcrx=zans(1:nth,1)*zcm
  bczx=zans(1:nth,2)*zcm
 
  !----------------------------------------------------
  !  (R,Z) data
 
  do i=1,nxi
     if(i.eq.1) then
        zrho=zxi_axis
     else if(i.eq.nbrk) then
        zrho=zxi_bdy
     else
        zrho=zxi(i)
     endif
     call eq_hrhochi(nth,zrho,zchi,2,ifcns,iwantv,nth,zans,ierr)
     if(ierr.ne.0) goto 9999
     if(i.eq.1) then
        rspl(1,1:nth,i)=zcm*sum(zans(1:nth,1))/nth
        zspl(1,1:nth,i)=zcm*sum(zans(1:nth,2))/nth
     else if(i.lt.nbrk) then
        rspl(1,1:nth,i)=zans(1:nth,1)*zcm
        zspl(1,1:nth,i)=zans(1:nth,2)*zcm
     else if(i.eq.nbrk) then
        rspl(1,1:nth,i)=zans(1:nth,1)*zcm
        zspl(1,1:nth,i)=zans(1:nth,2)*zcm
        rsplx(1,1:nth,i-nbrk+1)=zans(1:nth,1)*zcm
        zsplx(1,1:nth,i-nbrk+1)=zans(1:nth,2)*zcm
     else
        rsplx(1,1:nth,i-nbrk+1)=zans(1:nth,1)*zcm
        zsplx(1,1:nth,i-nbrk+1)=zans(1:nth,2)*zcm
     endif
  enddo
 
  !----------------------------------------------------
  !  setup the (R,Z) splines...
 
  !  R internal
 
  call r8mkbicub(th(1:nth),nth,xi(1:nint),nint,rspl,nth, &
       -1,bcr0,-1,bcr0,1,bcr0,1,bcr1,klinx,klinth,ierr)
  if(ierr.ne.0) goto 9999
 
  !  Z internal
 
  call r8mkbicub(th(1:nth),nth,xi(1:nint),nint,zspl,nth, &
       -1,bcz0,-1,bcz0,1,bcz0,1,bcz1,klinx,klinth,ierr)
  if(ierr.ne.0) goto 9999
 
  !  R external
 
  call r8mkbicub(th(1:nth),nth,xi(nbrk:nxi),next,rsplx,nth, &
       -1,bcr1a,-1,bcr1a,1,bcr1a,1,bcrx,klinxx,klinthx,ierr)
  if(ierr.ne.0) goto 9999
 
  !  Z external
 
  call r8mkbicub(th(1:nth),nth,xi(nbrk:nxi),next,zsplx,nth, &
       -1,bcz1a,-1,bcz1a,1,bcz1a,1,bczx,klinxx,klinthx,ierr)
  if(ierr.ne.0) goto 9999
 
  !----------------------------------------------------
  !  setup the (1/B) spline
  !  analytic continuation used for extrapolated region, following
  !  method of original MC code
 
  !  nint=nbrk
 
  do i=1,nint
     if(i.eq.1) then
        zrho=zxi_axis
     else if(i.eq.nbrk) then
        zrho=zxi_bdy
     else
        zrho=zxi(i)
     endif
     call eq_frhochi(nth,zrho,zchi,1,id_bmod,nth,zans,ierr)
     if(ierr.ne.0) goto 9999
     if(i.eq.1) then
        bspl(1,1:nth,i)=1/(zgauss*sum(zans(1:nth,1))/nth)
     else
        bspl(1,1:nth,i)=1/(zgauss*zans(1:nth,1))
     endif
  enddo
 
  !  analytic continuation... with smoothing...
 
  allocate(zwk(nth))
  zpio6=6.2831853071795862_R8/6
  zdelx_sm=1.0_R8/(10*zpio6)
 
  zdxi=xi(nint)-xi(nint-1)
  do i=nint+1,nxi
     do j=1,nth
        zwk(j)=ztrght2(xi(i)-1.0_R8,bspl(1,j,nint),zdxi, &
             bspl(1,j,nint-1),bspl(1,j,nint-2),bspl(1,j,nint-3))
     enddo
     zdelth_sm=zpio6*(1.0_R8-exp(-(xi(i)-xi(nint))/zdelx_sm))
     call thsmoo(nth/4)
     bspl(1,1:nth,i)=zwk
  enddo
  deallocate(zwk)
 
  !  setup the bicubic spline
 
  call r8mkbicub(th(1:nth),nth,xi(1:nxi),nxi,bspl,nth, &
       -1,bcz0,-1,bcz0,0,bcz0,0,bcz0,idum1,idum2,ierr)
  if(ierr.ne.0) go to 9999
 
  !----------------------------------------------------
  !  setup the "Jacobian" spline
  !  analytic continuation used for extrapolated region, following
  !  method of original MC code
 
  !  nint=nbrk
 
  !  defer axis calculation (0/0 limit)
 
  allocate(zdpsidrho(nint),zrhoj(nint))
  zrhoj(1)=zxi_axis
  zrhoj(nint)=zxi_bdy
  do i=2,nint-1
     zrhoj(i)=zxi(i)
  enddo

  call nbi_get_dpsidrho(nint,zrhoj,zdpsidrho)
 
  zdpsidrho=zgauss*zcm*zcm*zdpsidrho   ! T*m**2 -> gauss*cm**2
 
  do i=2,nint
     if(i.eq.nbrk) then
        zrho=zxi_bdy
     else
        zrho=zxi(i)
     endif
     call eq_getdetj(nth,zrho,zchi,zphi,zcm3,zans(1:nth,1),ierr)
     if(ierr.ne.0) go to 9999
     jspl(1,1:nth,i)=abs(zans(1:nth,1)/zdpsidrho(i))
  enddo
 
  !  numerical formulation for the axis
 
  jspl(1,1:nth,1)=sum(jspl(1,1:nth,2))/nth
 
  !  analytic continuation... with smoothing...
 
  allocate(zwk(nth))
  zpio6=6.2831853071795862_R8/6
  zdelx_sm=1.0_R8/(10*zpio6)
 
  zdxi=xi(nint)-xi(nint-1)
  do i=nint+1,nxi
     do j=1,nth
        zwk(j)=ztrght2(xi(i)-1.0_R8,jspl(1,j,nint),zdxi, &
             jspl(1,j,nint-1),jspl(1,j,nint-2),jspl(1,j,nint-3))
     enddo
     zdelth_sm=zpio6*(1.0_R8-exp(-(xi(i)-xi(nint))/zdelx_sm))
     call thsmoo(nth/4)
     jspl(1,1:nth,i)=zwk
  enddo
  deallocate(zwk)
 
  !  setup the bicubic spline
 
  call r8mkbicub(th(1:nth),nth,xi(1:nxi),nxi,jspl,nth, &
       -1,bcz0,-1,bcz0,0,bcz0,0,bcz0,idum1,idum2,ierr)
  if(ierr.ne.0) go to 9999
 
  !----------------------------------------------------
  !  setup splines for compression operator
  !  (if any) (removed DMC Nov 2001 -- redundant).
 
  !
  !  cleanup and exit
  !
 
  deallocate(zdpsidrho,zrhoj)
  deallocate(bcr0,bcr1,bcr1a,bcrx)
  deallocate(bcz0,bcz1,bcz1a,bczx)
  deallocate(zrho,zchi,zphi,zans)
 
  return
 
9999 continue
 
  !  unexpected error
 
  write(nonlin,*) ' ?? nbspline:  unexpected code error!'
  return
 
  contains
    subroutine thsmoo(iextnd)
 
      integer, intent(in) :: iextnd
 
      !  add hoc poloidal smooth of radial extrapolations for bspl, jspl
 
      REAL*8 :: zwk1(1-iextnd:nth+iextnd),zwk2(1-iextnd:nth+iextnd)
      REAL*8 :: zth(1-iextnd:nth+iextnd)
      REAL*8 :: zeps(1-iextnd:nth+iextnd),zdel(1-iextnd:nth+iextnd)
 
      integer iex,inum
      REAL*8 :: zdum = 0.0_R8
      REAL*8 zsm
      integer idrop,iblrs
 
      !----------------------
 
      zeps = 1.0E30_R8
      zdel = zdelth_sm
 
      zth(1:nth)=th(1:nth)
      zwk1(1:nth)=zwk(1:nth)
 
      do iex=1,iextnd
         zth(1-iex)=zth(1)-(zth(nth)-zth(nth-iex))
         zth(nth+iex)=zth(nth)+(zth(1+iex)-zth(1))
         zwk1(1-iex)=zwk1(nth-iex)
         zwk1(nth+iex)=zwk1(1+iex)
      enddo
 
      inum=nth+2*iextnd
      call r8filtr6(zth,zwk1,zwk2,inum,zeps,inum,zeps,0,zdel,0,zdum, &
           0,zdum,zdum,zdum,zsm,idrop,iblrs)
 
      zwk(1:nth)=zwk2(1:nth)
 
    end subroutine thsmoo
end subroutine nbspline
! 11Sep2004 fgtok -s r8_precision.sub nubeam_test_r8.sub "r8con.csh conversion"
! 11Sep2004 fgtok
