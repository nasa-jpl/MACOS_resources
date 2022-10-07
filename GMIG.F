! MATLAB provided header: [Matlab]\extern\include
#include <fintrf.h>

C***********************************************************************
C    +----------------------------------------------------------------+
C    |  Copyright (C) 1998-2006, California Institute of Technology.  |
C    |  U.S. Government Sponsorship is acknowledged.		      |
C    +----------------------------------------------------------------+
C***********************************************************************

C***********************************************************************
C  jwstG.F - MEXFUNCTION gateway function, f90 version of jwstG.f
C***********************************************************************


	SUBROUTINE mexfunction(NLHS, PLHS, NRHS, PRHS)

C   [PIX,RealEF,ImagEF,OPD,OPDMask,SPOT,WFE,CENT,MetMeas,USER]=...
C      GMI(prb,pzern,pgrid,pdm,pfa,prad,pimg,pflg,InfFcnZern,InfFcnGrid);

	use param_mod
        use src_mod
        use elt_mod
	use math_mod
        use smacos_mod
        use GMI_mod   ! (G)eneric (M)ACOS (I)nterface

	IMPLICIT NONE

! ------------------------------------------------------------------------------
C  This is the SMACOS user COMMON
! ------------------------------------------------------------------------------
        INTEGER npts,NoiseSeed(2),iSpot
        COMMON /UserCommon4/npts,NoiseSeed,iSpot

! ------------------------------------------------------------------------------
!  These are local variables
! ------------------------------------------------------------------------------

	INCLUDE 'GMI.inc'

	LOGICAL :: ifInitOptics,ifprb,ifpzern,ifpgrid,ifpdm,ifpfa,
     &	  	   ifprad,ifpimg,ifpflg,ifMetGMIG,ifPfGMIG

	INTEGER :: i,M,N,mxGetN,mxGetM,mxGetString,mxIsChar,
     &	  	   NLHS,NRHS,nPix,nPos,nProc,myProc,myPart,
     &	  Nprb,Npzern,Npgrid,Npdm,Npfa,Nprad,Npimg,Npflg,
     &	  model_size_in,gsStatus,szUSER,m_err_jwstG,dim1

	MWPOINTER :: PLHS(*),PRHS(*),mxGetPr,mxCreateDoubleMatrix,
     &	  prbP,pzernP,pgridP,pdmP,pfaP,pradP,pimgP,pflgP,
     &	  InfFcnZernP,InfFcnGridP,PixArrayP,OPDP,spotP,metMeasP,
     &	  WFEP,CENTP,USERP,msp,dPosP,dAngleP,mxCreateDoubleScalar,
     &    StrCmdP,RealEFdArrayP,ImagEFdArrayP,OPDMaskP

	REAL*8, dimension(:),   allocatable, save :: OPD,PIX,USER,
     &                          RealEFd,ImagEFd,OPDMask
        REAL*8, dimension(:,:), allocatable, save :: SPOT

	REAL*8 prb(mprb),pzern(mpzern),pgrid(mpgrid),pdm(mpdm),
     &	  pfa(mpfa),prad(mprad),WFError,CENT(2),
     &	  pimg(mpimg),pflg(mpflg),InfFcnZern(15),InfFcnGrid(mgrid,
     &    mgrid),msbuf(1)

        REAL*8, dimension(:,:), allocatable, save :: dPos, dAngle

	Integer, save :: curr_model_size=-1
        Logical, save :: first_entry=.true.

        CHARACTER*100 fname
        CHARACTER*3   StrCmd

        MWPOINTER :: mxGetNumberOfElements

C        REAL*8 D_CEIL

! ------------------------------------------------------------------------------
! Initialize flags
! ------------------------------------------------------------------------------

	DATA ifInitOptics/.FALSE./

! ------------------------------------------------------------------------------
! Check for proper number of arguments
! ------------------------------------------------------------------------------

! ------------------------------------------------------------------------------
! INTERFACE definition
! ------------------------------------------------------------------------------

 	IF (NRHS .NE. 13)
     &	  CALL mexErrMsgTxt('GMI requires 13 input arguments')
	IF (NLHS .NE. 10)
     &	  CALL mexErrMsgTxt('GMI requires 10 output argument')


! ------------------------------------------------------------------------------
! Check for process number argument
! ------------------------------------------------------------------------------
        M = mxGetM(PRHS(13)); N = mxGetN(PRHS(13))
        print*,'GMIG: M, N =',M,N
        if (M /= 1 .or. N /= 1) then
          CALL mexErrMsgTxt
     &      ('nProc argument must be a scalar value')
        end if
        msp = mxGetPr(PRHS(13))
        !Call mxCopyPtrToReal8(msp,msbuf,1)
        Call mxCopyPtrToReal8(msp,msbuf,mxGetNumberOfElements(PRHS(1)))
        nProc=msbuf(1)
	print*,'---GMI: nProc = ',nProc

        print*,'----GMI: **(elt_mod) mPolySide =', mPolySide 

! ------------------------------------------------------------------------------
! Check for model size
! ------------------------------------------------------------------------------
        M = mxGetM(PRHS(12)); N = mxGetN(PRHS(12))
        if (M /= 1 .or. N /= 1) then
          CALL mexErrMsgTxt
     &      ('model_size argument must be a scalar value')
        end if
        msp = mxGetPr(PRHS(12))
        Call mxCopyPtrToReal8(msp, msbuf, mxGetNumberOfElements(PRHS(12)))
        model_size_in = msbuf(1)
        print*,'---GMI: model_size_in = ', model_size_in


	! Initialize or reset SMACOS model
        If (first_entry .or.
     &      model_size_in /= curr_model_size) then
          curr_model_size = model_size_in
          call macos_init_all(curr_model_size)

	  mPix2=mPix*mPix

          ! Parameters have been defined after init call
          szUSER=max(mElt,15)
          allocate(OPD(mRay),SPOT(2,mRay),PIX(mPix*mPix),USER(szUSER),
     &             OPDMask(mRay),RealEFd(mPix*mPix),ImagEFd(mPix*mPix),
     &             dPos(3,mElt),dAngle(4,mElt),stat=m_err_jwstG)
         if (m_err_jwstG /= 0) then
            call macos_memory_failure
     &                ('GMIG: allocate failed!')
          end if
        End If


! ------------------------------------------------------------------------------
! Check the dimensions of input matrices
! ------------------------------------------------------------------------------

	M = mxGetM(PRHS(1))
	N = mxGetN(PRHS(1))
	Nprb=MAX(M,N)
	IF ((Nprb.LE.mprb).AND.(MIN(M,N).LE.1)) THEN
          ifprb = .NOT.(Nprb.LE.1)
	ELSE
          print*,'** GMIG: matlab Nprb =',Nprb
          print*,'** GMIG: GMI mprb =',mprb
          print*,'** GMIG: Need Nprb <= mprb' 
	  CALL mexErrMsgTxt('prb must be a 1x1 or mprb x 1 matrix')
	END IF

	M      = mxGetM(PRHS(2))
	N      = mxGetN(PRHS(2))
	Npzern = MAX(M,N)
	IF ((Npzern.LE.mpzern).AND.(MIN(M,N).LE.1)) THEN
          ifpzern = .NOT.(Npzern.LE.1)
	ELSE
	  CALL mexErrMsgTxt('pzern must be scalar, empty or a '//
     &	                    'mpzern x 1 matrix')
	END IF


	M      = mxGetM(PRHS(3))
	N      = mxGetN(PRHS(3))
	Npgrid = MAX(M,N)
	IF ((Npgrid.LE.mpgrid).AND.(MIN(M,N).LE.1)) THEN
          ifpgrid = .NOT.(Npgrid.LE.1)
	ELSE
          print*,'*** GMIG: mpgrid = ',mpgrid
	  CALL mexErrMsgTxt('pgrid must be a 1x1 or mpgrid x 1 matrix')
	END IF


	M    = mxGetM(PRHS(4))
	N    = mxGetN(PRHS(4))
	Npdm = MAX(M,N)
	print*,'** prhs4: M,N,Npdm,mpdm = ',M,N,Npdm,mpdm
	IF ((Npdm.LE.mpdm).AND.(MIN(M,N).LE.1)) THEN
	  ifpdm = .NOT.(Npdm.LE.1)
	ELSE
	  CALL mexErrMsgTxt('pdm must be a 1x1 or mpdm x 1 matrix')
	END IF


	M    = mxGetM(PRHS(5))
	N    = mxGetN(PRHS(5))
	Npfa = MAX(M,N)
	IF (((Npfa.EQ.mpfa).OR.(Npfa.EQ.3).OR.(Npfa.LE.1)).AND.
     &    (MIN(M,N).LE.1)) THEN
          ifpfa = .NOT.(Npfa.LE.1)
	ELSE
	  CALL mexErrMsgTxt('pfa must be scalar, empty or a '//
     &	                    'mpfa x 1 matrix')
	END IF
        !
        ! ----------------------------------------------------------------------
        !
	M = mxGetM(PRHS(6))
	N = mxGetN(PRHS(6))
	Nprad=MAX(M,N)
	IF (((Nprad.EQ.mprad).OR.(Nprad.LE.1)).AND.(MIN(M,N).LE.1)) THEN
	  ifprad = .NOT.(Nprad.LE.1)
!	  IF (Nprad.LE.1) THEN
!	    ifprad=.FALSE.
!	  ELSE
!	    ifprad=.TRUE.
!	  END IF
	ELSE
	  CALL mexErrMsgTxt('prad must be a 1x1 or mprad x 1 matrix')
	END IF
        !
        ! ----------------------------------------------------------------------
        !
        !
        !    Npimg   : =0 Sets: Wavelen = lambda   and  Flux = FluxNom (in-file value)
        !            : =1 Sets: Wavelen = pimg(1)  and  Flux = 1d0
        !            : =2 Sets: Wavelen = pimg(1)  and  Flux = pimg(2)
        !            : >2 Sets: Incoherent superposition of pixelated Intensity values from each wavelength
        !                       where Wavelen = pimg(iLambda)   with iLambda = odd
        !                             Flux    = pimg(iFlux)     iFlux   = iLambda + 1
        !
	M = mxGetM(PRHS(7))
	N = mxGetN(PRHS(7))
	Npimg=MAX(M,N)
	ifpimg = (Npimg.NE.0)    ! permit empty vector, i.e. "[]" and if empty then ifpimg=.FALSE.
	IF (ifpimg) THEN
	  IF ((Npimg.LE.mpimg).AND.(MIN(M,N).EQ.1)) THEN   ! check input size
	    IF ((Npimg.GE.2).AND.                          ! Npimg = 2,4,6, ...
     &            (MOD(Npimg,2).NE.0)) THEN       ! check input size
	      CALL mexErrMsgTxt('pimg: Input tuple must be scalar '//
     &	                        'or of even length.')
	    END IF
	  ELSE
	    CALL mexErrMsgTxt('pimg must be empty, scalar or '//
     &	                      'mpimg x 1 matrix')
	  END IF
	ELSE
	  CALL DZERO(pimg,mpimg)        ! set to zero
	END IF
        !
        ! ----------------------------------------------------------------------
        !
	M = mxGetM(PRHS(8))
	N = mxGetN(PRHS(8))
	Npflg=MAX(M,N)
	IF ((Npflg.LE.mpflg).AND.(Npflg.GE.1)) THEN
	  ifpflg = .NOT.(Npflg.LE.1)
!	  IF (Npflg.LE.1) THEN
!	    ifpflg=.FALSE.
!	  ELSE
!	    ifpflg=.TRUE.
!	  END IF
	ELSE
	  CALL mexErrMsgTxt('pflg must be mpflg x 1 or smaller')
	END IF
        !
        ! ----------------------------------------------------------------------
        !
	IF (ifpdm) THEN
	  M = mxGetM(PRHS(9))
	  N = mxGetN(PRHS(9))
	  IF ((MAX(M,N).NE.15).OR.(MIN(M,N).NE.1)) THEN
	    CALL mexErrMsgTxt('InfFcnZern must be 15x1')
	  END IF

	  M = mxGetM(PRHS(10))
	  N = mxGetN(PRHS(10))
	  IF ((M.NE.mgrid).OR.(N.NE.mgrid)) THEN
	    CALL mexErrMsgTxt('InfFcnGrid must be mgrid x mgrid')
	  END IF
	END IF

        gsStatus=mxGetString(prhs(11),fname,100)
        if (gsStatus.ne.0) then
          call mexErrMsgTxt('prescription filename too long')
        endif

! ------------------------------------------------------------------------------
! Assign pointers to input arguments
! ------------------------------------------------------------------------------

	IF (ifprb)   prbP   = mxGetPr(PRHS(1))
	IF (ifpzern) pzernP = mxGetPr(PRHS(2))
	IF (ifpgrid) pgridP = mxGetPr(PRHS(3))
	IF (ifpdm) THEN
	  pdmP        = mxGetPr(PRHS(4))
	  InfFcnZernP = mxGetPr(PRHS(9))
	  InfFcnGridP = mxGetPr(PRHS(10))
	END IF
	IF (ifpfa)  pfaP  = mxGetPr(PRHS(5))
	IF (ifprad) pradP = mxGetPr(PRHS(6))
	IF (ifpimg) pimgP = mxGetPr(PRHS(7))
	IF (ifpflg) pflgP = mxGetPr(PRHS(8))

! ------------------------------------------------------------------------------
! Get values for input arguments
! ------------------------------------------------------------------------------

#if 0
        IF (ifprb)   CALL mxCopyPtrToReal8(prbP, prb, Nprb)
	IF (ifpzern) CALL mxCopyPtrToReal8(pzernP, pzern, Npzern)
	IF (ifpgrid) CALL mxCopyPtrToReal8(pgridP, pgrid, Npgrid)
	!print*,'**** GMIG_met: Npgrid =',Npgrid
	IF (ifpdm) THEN
	  CALL mxCopyPtrToReal8(pdmP, pdm, Npdm)
	  CALL mxCopyPtrToReal8(InfFcnZernP, InfFcnZern, 15)
	  CALL mxCopyPtrToReal8(InfFcnGridP, InfFcnGrid, mgrid2)
	END IF
	IF (ifpfa)  CALL mxCopyPtrToReal8(pfaP, pfa, Npfa)
	IF (ifprad) CALL mxCopyPtrToReal8(pradP, prad, Nprad)
	IF (ifpimg) CALL mxCopyPtrToReal8(pimgP, pimg, Npimg)
	IF (ifpflg) CALL mxCopyPtrToReal8(pflgP, pflg, Npflg)
#endif
        IF (ifprb)   CALL mxCopyPtrToReal8(prbP, prb, mxGetNumberOfElements(PRHS(1)))
        IF (ifpzern) CALL mxCopyPtrToReal8(pzernP, pzern, mxGetNumberOfElements(PRHS(2)))
        IF (ifpgrid) CALL mxCopyPtrToReal8(pgridP, pgrid, mxGetNumberOfElements(PRHS(3)))
        IF (ifpdm) THEN
          CALL mxCopyPtrToReal8(pdmP, pdm, mxGetNumberOfElements(PRHS(4)))
          CALL mxCopyPtrToReal8(InfFcnZernP, InfFcnZern, mxGetNumberOfElements(PRHS(9)))
          CALL mxCopyPtrToReal8(InfFcnGridP, InfFcnGrid, mxGetNumberOfElements(PRHS(10)))
        END IF
        IF (ifpfa)  CALL mxCopyPtrToReal8(pfaP, pfa, mxGetNumberOfElements(PRHS(5)))
        IF (ifprad) CALL mxCopyPtrToReal8(pradP, prad, mxGetNumberOfElements(PRHS(6)))
        IF (ifpimg) CALL mxCopyPtrToReal8(pimgP, pimg, mxGetNumberOfElements(PRHS(7)))
        IF (ifpflg) CALL mxCopyPtrToReal8(pflgP, pflg, mxGetNumberOfElements(PRHS(8)))
        

	IF (ifpgrid.AND..NOT.ifprb) THEN
	  DO i=1,mprb
	    prb(i)=0d0
	  END DO
	END IF


! ------------------------------------------------------------------------------
! Parallel PIX support
! ------------------------------------------------------------------------------
        CALL FKPROCS(nProc,myProc,myPart,Npimg)
! ------------------------------------------------------------------------------

! ------------------------------------------------------------------------------
! Compute optical performance
! ------------------------------------------------------------------------------
	CALL GMI_DVR(PIX,RealEFd,ImagEFd,OPD,OPDMask,SPOT,WFError,
     &          ifMetGMIG,ifPfGMIG,cent,USER,prb,pzern,pgrid,pdm,
     &          pfa,prad,pimg,pflg,
     &		ifprb,ifpzern,ifpgrid,ifpdm,ifpfa,ifprad,ifpflg,
     &		ifInitOptics,nPix,Npimg,InfFcnZern,InfFcnGrid,
     &          fname,curr_model_size,myProc,nProc)

        print*,'************** -->GMIG: After GMI_DVR'


! ------------------------------------------------------------------------------
! Parallel PIX support - final PIX compose
! ------------------------------------------------------------------------------
        !CALL PIPEIO(myPart,myProc,PIX,nPix)
! ------------------------------------------------------------------------------

! ------------------------------------------------------------------------------
! Create matrices for return arguments
! ------------------------------------------------------------------------------

        print*,
     &    '**-->GMIG: nPix, nGridPts, nMetMeas,mElt =',
     &    nPix,nGridPts,nMetMeas,mElt
        If (iSpot==0) iSpot = 1  ! make sure iSpot positive
        print*,'iSpot =',iSpot
	PLHS(1) = mxCreateDoubleMatrix(nPix,nPix,0)
	PLHS(2) = mxCreateDoubleMatrix(nPix,nPix,0)
	PLHS(3) = mxCreateDoubleMatrix(nPix,nPix,0)
	PLHS(4) = mxCreateDoubleMatrix(nGridPts,nGridPts,0)
        PLHS(5) = mxCreateDoubleMatrix(nGridPts,nGridPts,0)
        print*,'**-->GMIG: Aft PLHS(5)'
#if 1 
	PLHS(6) = mxCreateDoubleMatrix(2,iSpot,0)
        print*,'**-->GMIG: Aft PLHS(6)'
	PLHS(7) = mxCreateDoubleMatrix(1,1,0)
        print*,'**-->GMIG: Aft PLHS(7)'
	PLHS(8) = mxCreateDoubleMatrix(2,1,0)
	PLHS(9) = mxCreateDoubleMatrix(max(nMetMeas,1),1,0)
	PLHS(10) = mxCreateDoubleMatrix(mElt,1,0)
#endif
        print*,'**-->GMIG: After return array memeory alloc'

! ------------------------------------------------------------------------------
! Assign pointers to return arguments
! ------------------------------------------------------------------------------

	PixArrayP = mxGetPr(PLHS(1))
	RealEFdArrayP = mxGetPr(PLHS(2))
	ImagEFdArrayP = mxGetPr(PLHS(3))
	OPDP      = mxGetPr(PLHS(4))
        OPDMaskP  = mxGetPr(PLHS(5))
	spotP     = mxGetPr(PLHS(6))
	WFEP      = mxGetPr(PLHS(7))
	CENTP     = mxGetPr(PLHS(8))
	metMeasP   = mxGetPr(PLHS(9))
	USERP     = mxGetPr(PLHS(10))

        print*,'After getting return pointers'
! ------------------------------------------------------------------------------
! Save return values
! ------------------------------------------------------------------------------
!	USER(8)  = RptElt(1,10)-rptNom(1,10)
!	USER(9)  = RptElt(2,10)-rptNom(2,10)
!	USER(10) = RptElt(3,10)-rptNom(3,10)
!	USER(11) = KrElt(10)
!	USER(12) = KcElt(10)
!	USER(13) = pMon(1,10)
!	USER(14) = pMon(2,10)
!	USER(15) = pMon(3,10)
!	USER(16) = rptNom(1,10)
!	USER(17) = rptNom(2,10)
!	USER(18) = rptNom(3,10)

        dim1 = nPix*nPix
	CALL mxCopyReal8ToPtr(PIX,PixArrayP,dim1)
	CALL mxCopyReal8ToPtr(RealEFd,RealEFdArrayP,dim1)
	CALL mxCopyReal8ToPtr(ImagEFd,ImagEFdArrayP,dim1)

        dim1 = nGridPts*nGridPts
	CALL mxCopyReal8ToPtr(OPD,OPDP,dim1)
	CALL mxCopyReal8ToPtr(OPDMask,OPDMaskP,dim1)

        dim1 = iSpot*2
	CALL mxCopyReal8ToPtr(SPOT,spotP,dim1)

	CALL mxCopyReal8ToPtr(WFError,WFEP,1)
	CALL mxCopyReal8ToPtr(CENT,CENTP,2)
        CALL mxCopyReal8ToPtr(metMeasBuf,metMeasP,max(nMetMeas,1))
	CALL mxCopyReal8ToPtr(USER,USERP, mElt)
        print*,'After copying return pointers'

	first_entry=.false.
	RETURN
	END SUBROUTINE

C***********************************************************************


