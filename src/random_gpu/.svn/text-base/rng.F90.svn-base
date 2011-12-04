#include "config_random.h"
module rngf77
  ! interface for the original f77 routines
  use rngdef
  implicit none
  interface
     function random(ri,ra)
       use rngdef
       implicit none
       real(double) :: random
       integer, intent(inout) :: ri
       real(double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random
     function srandom(ri,ra)
       use rngdef
       implicit none
       real(single) :: srandom
       integer, intent(inout) :: ri
       real(double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom
     subroutine random_array(y,n,ri,ra)
       use rngdef
       implicit none
       integer, intent(in) :: n
       real(double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array
     subroutine srandom_array(y,n,ri,ra)
       use rngdef
       implicit none
       integer, intent(in) :: n
       real(single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array
     subroutine rand_batch(ri,ra)
       use rngdef
       implicit none
       integer, intent(inout) :: ri
       real(double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch
     subroutine random_init(seed,ri,ra)
       use rngdef
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init
     subroutine decimal_to_seed(decimal,seed)
       use rngdef
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed
     subroutine string_to_seed(string,seed)
       use rngdef
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed
     subroutine set_random_seed(time,seed)
       use rngdef
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed
     subroutine seed_to_decimal(seed,decimal)
       use rngdef
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal
     subroutine next_seed3(n0,n1,n2,seed)
       use rngdef
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed3
     subroutine next_seed(n0,seed)
       use rngdef
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed
  end interface
end module rngf77

subroutine rng_init(seed,state)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init(seed,state%index,state%array)
  return
end subroutine rng_init

subroutine rng_step_seed(seed,n0,n1,n2)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed3(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed

function rng_print_seed(seed)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed
  call seed_to_decimal(seed,rng_print_seed)
  return
end function rng_print_seed

subroutine rng_number_d0(state,x)
  use rngf77
  implicit none
  real(double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0

subroutine rng_number_s0(state,x)
  use rngf77
  implicit none
  real(single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(single), parameter :: ulps = 2.0_single**(-23)
  real(double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5_single)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0

subroutine rng_number_d1(state,x)
  use rngf77
  implicit none
  real(double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1

subroutine rng_number_s1(state,x)
  use rngf77
  implicit none
  real(single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(single), parameter :: ulps = 2.0_single**(-23)
  real(double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5_single)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5_single)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1

subroutine rng_number_d2(state,x)
  use rngdef
  implicit none
  real(double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1(state,x(:,i))
  end do
end subroutine rng_number_d2

subroutine rng_number_s2(state,x)
  use rngdef
  implicit none
  real(single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1(state,x(:,i))
  end do
end subroutine rng_number_s2

subroutine rng_number_d3(state,x)
  use rngdef
  implicit none
  real(double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2(state,x(:,:,i))
  end do
end subroutine rng_number_d3

subroutine rng_number_s3(state,x)
  use rngdef
  implicit none
  real(single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2(state,x(:,:,i))
  end do
end subroutine rng_number_s3

subroutine rng_gauss_d1(state,x)
  use rngf77
  implicit none
  real(double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1
  end interface
  integer :: i
  real(double), parameter :: pi=__DPI
  real(double) :: theta,z
  call rng_number_d1(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0_double*x(i)-1.0_double)
     z=sqrt(-2.0_double*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0_double*x(ubound(x,1))-1.0_double)
  z=sqrt(-2.0_double*log(random(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1

subroutine rng_gauss_d0(state,x)
  use rngdef
  implicit none
  real(double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1
  end interface
  real(double), dimension(1) :: y
  call rng_gauss_d1(state, y)
  x=y(1)
end subroutine rng_gauss_d0

subroutine rng_gauss_s1(state,x)
  use rngf77
  implicit none
  real(single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1
  end interface
  integer :: i
  real(single), parameter :: pi=__SPI
  real(single) :: theta,z
  call rng_number_s1(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0_single*x(i)-1.0_single)
     z=sqrt(-2.0_single*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0_single*x(ubound(x,1))-1.0_single)
  z=sqrt(-2.0_single*log(srandom(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1

subroutine rng_gauss_s0(state,x)
  use rngdef
  implicit none
  real(single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1
  end interface
  real(single), dimension(1) :: y
  call rng_gauss_s1(state, y)
  x=y(1)
end subroutine rng_gauss_s0

subroutine rng_gauss_d2(state,x)
  use rngdef
  implicit none
  real(double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1(state,x(:,i))
  end do
end subroutine rng_gauss_d2

subroutine rng_gauss_s2(state,x)
  use rngdef
  implicit none
  real(single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1(state,x(:,i))
  end do
end subroutine rng_gauss_s2

subroutine rng_gauss_d3(state,x)
  use rngdef
  implicit none
  real(double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3

subroutine rng_gauss_s3(state,x)
  use rngdef
  implicit none
  real(single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3

subroutine rng_set_seed_time(seed,time)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed(ltime,seed)
  end if
end subroutine rng_set_seed_time

subroutine rng_set_seed_int(seed,n)
  use rngdef
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int

subroutine rng_set_seed_string(seed,string)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string
