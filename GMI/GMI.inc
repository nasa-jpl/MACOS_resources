C***********************************************************************
C    +----------------------------------------------------------------+
C    |  Copyright (C) 1998-2006, California Institute of Technology.  |
C    |  U.S. Government Sponsorship is acknowledged.		      |
C    +----------------------------------------------------------------+
C***********************************************************************


	Integer, parameter ::
     &             numseg    =  6,   ! max # of segments
     &             numacf    =  6,
     &             numSAF    =  0,   ! max # of sub-aperture flats (ACFs = SAFs)
     &             mzern     = 12,
!    &             mgrid     = 256,
     &             mgrid     = 99,
     &             mgrid2    = mgrid*mgrid,
     &             mpzern    = numseg*mzern,
!    &             mpdm      = numseg,
     &             mpdm      = (numseg+2)*2,    ! segments+m2+m3 in double-pass
!    &             mprb      = (numseg+6+2)*6,       ! Last 2 are FSM
!    &             mprb      = (numseg+3+4+2)*6,     ! Last 2 are FSM
     &             mprb      = (numseg+55)*6,   ! make mprb large enough to include all elts in Rx -jzlou     
     &             mpgrid    = mgrid2*((numseg+1)*2+numacf), ! support segments+SM+ACFs and double-pass 
     &             mpfa      = 7,
     &             mprad     = numseg,
     &             mpimg     = 100,
     &             mpflg     = 2000,
     &             msenmea   = 2  ! total sensing spots; eash sensing spot has 4 measuremens  

	Integer :: mPix2
