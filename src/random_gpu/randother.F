#include "config_random.h"
! Version 2.1 of random number routines
! Author: Charles Karney <karney@princeton.edu>
! Date: 1999-10-27 14:10 -0400
!
      subroutine random_gauss(y,n,ri,ra)
      implicit none
      integer ri
      REAL8 ra(0:100-1)
      integer n
      REAL8 y(0:n-1)
      integer i
      REAL8 pi,theta,z
      REAL8 random
      external random_array,random
      data pi/__PI/
#ifndef HIPREC
      Real ys(0:n-1)
      Real spi,stheta,sz
      Real srandom
      external srandom_array,srandom
      data spi/__RPI/
#else /* HIPREC */
      entry srandom_gauss(y,n,ri,ra)
#endif /* HIPREC */
      if(n.LE.0)return
      call random_array(y,n,ri,ra(0))
      do i=0,int(n/2)*2-1,2
      theta=pi*(TWO*y(i)-ONE)
      z=sqrt(-TWO*log(y(i+1)))
      y(i)=z*cos(theta)
      y(i+1)=z*sin(theta)
      end do
      if(mod(n,2).EQ.0)return
      theta=pi*(TWO*y(n-1)-ONE)
      z=sqrt(-TWO*random(ri,ra(0)))
      y(n-1)=z*cos(theta)
      return
#ifndef HIPREC
      entry srandom_gauss(ys,n,ri,ra)
      if(n.LE.0)return
      call srandom_array(ys,n,ri,ra(0))
      do i=0,int(n/2)*2-1,2
      stheta=spi*(2.0*ys(i)-1.0)
      sz=sqrt(-2.0*log(ys(i+1)))
      ys(i)=sz*cos(stheta)
      ys(i+1)=sz*sin(stheta)
      end do
      if(mod(n,2).EQ.0)return
      stheta=spi*(2.0*ys(n-1)-1.0)
      sz=sqrt(-2.0*srandom(ri,ra(0)))
      ys(n-1)=sz*cos(stheta)
      return
#endif /* not HIPREC */
      end
!
      subroutine random_isodist(v,n,ri,ra)
      implicit none
      integer ri
      REAL8 ra(0:100-1)
      integer n
      REAL8 v(0:3*n-1)
      integer i
      REAL8 pi,costheta,phi
      external random_array
      data pi/__PI/
#ifndef HIPREC
      Real vs(0:3*n-1)
      Real spi,scostheta,sphi
      external srandom_array
      data spi/__RPI/
#else /* HIPREC */
      entry srandom_isodist(v,n,ri,ra)
#endif /* HIPREC */
      if(n.LE.0)return
      call random_array(v(n),2*n,ri,ra(0))
      do i=0,n-1
      costheta=TWO*v(n+2*i)-ONE
      phi=pi*(TWO*v(n+2*i+1)-ONE)
      v(3*i)=cos(phi)*sqrt(ONE-costheta**2)
      v(3*i+1)=sin(phi)*sqrt(ONE-costheta**2)
      v(3*i+2)=costheta
      end do
      return
#ifndef HIPREC
      entry srandom_isodist(vs,n,ri,ra)
      if(n.LE.0)return
      call srandom_array(vs(n),2*n,ri,ra(0))
      do i=0,n-1
      scostheta=2.0*vs(n+2*i)-1.0
      sphi=spi*(2.0*vs(n+2*i+1)-1.0)
      vs(3*i)=cos(sphi)*sqrt(1.0-scostheta**2)
      vs(3*i+1)=sin(sphi)*sqrt(1.0-scostheta**2)
      vs(3*i+2)=scostheta
      end do
      return
#endif /* not HIPREC */
      end
!
      subroutine random_cosdist(v,n,ri,ra)
      implicit none
      integer ri
      REAL8 ra(0:100-1)
      integer n
      REAL8 v(0:3*n-1)
      integer i
      REAL8 pi,costheta2,phi
      external random_array
      data pi/__PI/
#ifndef HIPREC
      Real vs(0:2*n-1)
      Real spi,scostheta2,sphi
      external srandom_array
      data spi/__RPI/
#else /* HIPREC */
      entry srandom_cosdist(v,n,ri,ra)
#endif /* HIPREC */
      if(n.LE.0)return
      call random_array(v(n),2*n,ri,ra(0))
      do i=0,n-1
      costheta2=v(n+2*i)
      phi=pi*(TWO*v(n+2*i+1)-ONE)
      v(3*i)=cos(phi)*sqrt(ONE-costheta2)
      v(3*i+1)=sin(phi)*sqrt(ONE-costheta2)
      v(3*i+2)=sqrt(costheta2)
#ifndef HIPREC
      end do
      return
      entry srandom_cosdist(vs,n,ri,ra)
      if(n.LE.0)return
      call srandom_array(vs(n),2*n,ri,ra(0))
      do i=0,n-1
      scostheta2=vs(n+2*i)
      sphi=spi*(2.0*vs(n+2*i+1)-1.0)
      vs(3*i)=cos(sphi)*sqrt(1.0-scostheta2)
      vs(3*i+1)=sin(sphi)*sqrt(1.0-scostheta2)
      vs(3*i+2)=sqrt(scostheta2)
#endif /* not HIPREC */
      end do
      return
      end
!
