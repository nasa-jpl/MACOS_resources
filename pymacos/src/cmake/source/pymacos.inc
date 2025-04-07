	Integer, parameter :: &
                   numseg    = 36, &  ! max # of segments
                   numSAF    =  6, &  ! max # of sub-aperture flats (ACFs = SAFs)
                   mzern     = 12, &
                   mgrid     = 99, &
                   mgrid2    = mgrid*mgrid, &
                   mpzern    = ((numseg+numSAF*2+3)*6+2)*mzern, &
                   mpdm      = numseg, &
                   mprb      = (numseg+numSAF*2+3)*6, &      ! Last 2 are FSM
                   mpgrid    = mgrid2*(numseg+numSAF+3*2), & ! 3 = SM,TM,FSM
                   mpfa      = 7, &
                   mprad     = numseg, &
                   mpimg     = 10000, &
                   mpflg     = 1000

	Integer :: mPix2
