! f90 interface to random number routine
! Charles Karney <karney@pppl.gov> 1999-09-30 16:58 -0400

#include "config_random.h"

module rngdef
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=selected_real_kind(6)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef

module rng
  ! the main module 
  use rngdef, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0(state,x)
       use rngdef
       implicit none
       real(double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0
     subroutine rng_number_d1(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1
     subroutine rng_number_d2(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2
     subroutine rng_number_d3(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3
     subroutine rng_number_s0(state,x)
       use rngdef
       implicit none
       real(single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0
     subroutine rng_number_s1(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1
     subroutine rng_number_s2(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2
     subroutine rng_number_s3(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3
  end interface
  interface rng_gauss
     ! Gaussian random numbers
     subroutine rng_gauss_d0(state,x)
       use rngdef
       implicit none
       real(double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0
     subroutine rng_gauss_d1(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1
     subroutine rng_gauss_d2(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2
     subroutine rng_gauss_d3(state,x)
       use rngdef
       implicit none
       real(double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3
     subroutine rng_gauss_s0(state,x)
       use rngdef
       implicit none
       real(single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0
     subroutine rng_gauss_s1(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1
     subroutine rng_gauss_s2(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2
     subroutine rng_gauss_s3(state,x)
       use rngdef
       implicit none
       real(single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3
  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time(seed,time)
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time
     subroutine rng_set_seed_int(seed,n)
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int
     subroutine rng_set_seed_string(seed,string)
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string
  end interface

  interface
     function rng_print_seed(seed)
       ! return the seed in a string representation
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed
     end function rng_print_seed
     subroutine rng_init(seed,state)
       ! initialize the state from the seed
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init
     subroutine rng_step_seed(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed
  end interface
end module rng
