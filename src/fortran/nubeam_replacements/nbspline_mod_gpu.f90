module nbspline_mod

  save

  !  module to hold single precision R(theta,xi) Z(theta,xi)
  !  splines -- interior and exterior regions

  !  setup routine:  nbspline.f90
  !  evaluation routine:  nbmomry.f90

  integer nxi,nth
  real*8, dimension(:), allocatable :: xi   ! radial coordinate
  real*8, dimension(:), allocatable :: th   ! poloidal coordinate

  ! setup seconday xi and theta variables for use in gpu_orbit_init
  real*8, dimension(:), allocatable :: xi2 ! radial coordinate
  real*8, dimension(:), allocatable :: th2 ! poloidal coordinate

  !  (array dimensioning added to pass LF95 argument checks w/vectorized
  !  spline routines)

  real*8 hxi(1),hxii(1),hth(1),hthi(1),xspani,thspani   ! spacing parameters
  integer :: nbsii(1),nbsjj(1)

  integer nbrk                 ! index xi(nbrk) ~ bdy
  real*8 xi_brk                  ! bdy xi value

  ! it is assumed that one of the xi values will be at the boundary

  integer nint                 ! no. of points inside bdy, counting bdy
  integer next                 ! no. of points outside bdy, counting bdy

  ! spline data arrays: (R,Z) internal & external

  real*8, dimension(:,:,:), allocatable :: rspl,rsplx,zspl,zsplx

  ! spline data arrays: (1/B) and [J] with analytic continuation

  real*8, dimension(:,:,:), allocatable :: bspl,jspl

  ! even spacing flags

  integer klinx,klinth,klinxx,klinthx

  real*8, parameter :: thmin=0.0d0
  real*8, parameter :: thmax=6.2831853071795862d0

end module nbspline_mod
