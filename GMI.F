C***********************************************************************
C    +----------------------------------------------------------------+
C    |  Copyright (C) 1998-2009, California Institute of Technology.  |
C    |  U.S. Government Sponsorship is acknowledged.                  |
C    +----------------------------------------------------------------+
C***********************************************************************

C**************************************************************************
C Adapted from Sigrist's GMI.F 
C**************************************************************************

#include "realtype.h"

        MODULE GMI_mod
        use param_mod
        use src_mod
        use elt_mod
        use macos_mod
	use math_mod
        use traceutil_mod, only : LBWKnoPrint,CRIncidDir,opdRayMask
        use smacos_mod

        IMPLICIT NONE

        REAL*8, dimension(:,:), allocatable, save :: psiNom,
     &          vptNom,rptNom,pNom,xNom,yNom,zNom,ObsTypeNom
        REAL*8, dimension(:,:,:), allocatable, save :: SrfMetPosNom

#if 0
	REAL*8, dimension(:),   allocatable, save :: OPD,PIX,USER,
     &                                               RealEFd,ImagEFd
        REAL*8, dimension(:,:), allocatable, save :: SPOT
#endif

        SREAL,  dimension(:,:), allocatable, save :: PixArray
        REAL*8, dimension(:,:), allocatable, save :: OPDMat,RaySpot


        ! These are the SMACOS call-line variables:
        CHARACTER*256     :: command   ! length matching SMACOS
        INTEGER,PARAMETER :: MARG=9

        CHARACTER*256     :: CARG(MARG)
        REAL*8            :: DARG(MARG)
        INTEGER           :: IARG(MARG)
        LOGICAL           :: LARG
        SREAL             :: RARG(MARG),RMSWFE


        CONTAINS
        
 
        SUBROUTINE GMI_DVR(PIX,RealEF,ImagEF,OPD,OPDMask,
     &     SPOT,WFError,ifMet,ifSpf,
     &     cent,USER,prb,pzern,pgrid,pdm,pfa,prad,pimg,pflg,
     &     ifprb,ifpzern,ifpgrid,ifpdm,ifpfa,ifprad,ifpflg,
     &     ifInitOptics,nPix,Npimg,InfFcnZern,InfFcnGrid,fname,
     &     modelsize,ip,np)

        IMPLICIT NONE

        ! Dynamic array arguments
        REAL*8, dimension(:) :: PIX,RealEF,ImagEF,OPD,OPDMask
        REAL*8, dimension(:,:) :: SPOT

        ! Non-dynamic array arguments
        REAL*8, dimension(:) :: prb,pzern,pgrid,pdm,
     &          pfa,prad,pimg,pflg,cent,USER,InfFcnZern
        REAL*8, dimension(:,:) :: InfFcnGrid
        REAL*8 :: WFError
        Logical :: ifprb,ifpzern,ifpgrid,ifpdm,ifpfa,ifprad,
     &             ifpimg,ifpflg,ifInitOptics,ifMet,ifSpf,
     &             rxStopSetSave
        Integer :: nPix,Npimg,modelsize,ip,np
        character fname*(*)

        ! These are the SMACOS user common
        INTEGER npts,NoiseSeed(2),iSpot
        REAL*8 CntrSpot(3),RefSpot(2)
        REAL*8 xcent,ycent,xLocal(3),yLocal(3),zLocal(3)
        COMMON /UserCommon4/npts,NoiseSeed,iSpot
        COMMON /UserCommon8/xcent,ycent,xLocal,yLocal,zLocal,
     &                      CntrSpot,RefSpot
        !
        ! local variables
        !

	! Dynamic local arrays
        REAL*8, dimension(:), allocatable, save :: KrNom, nObsNom,
     &                                     IndRefNom, KcNom
        REAL*8, dimension(:,:,:), allocatable, save ::
     &          ObsVecNom, TEltNom

        INCLUDE 'GMI.inc'

        LOGICAL :: ifMask,MsgsOn,ifPIXSpotDetCheck,ifPIXflip,
     &             ifOPDCalc,ifSysCalib,ifMetCalc,ifSpfCalc

        INTEGER :: i,j,k,kk,iElt,iSeg,iSegmin,iNode,jNode,nzern,ifOPD,
     &    ifPIX,ifSPOT,cPix,cGrid,P,StopType,ifShotNoise,ifPIXElt,ifFDP,
     &    numgridSrf,StartSeed,ipflg,igridSrf,jgridSrf,gridSrf(mprb),
     &    lgridSrf(mElt),izernSrf,jzernSrf,zernSrf(mprb),lzernSrf(mElt),
     &    idmSrf,jdmSrf,dmSrf(mprb),ldmSrf(mElt),LensElt,Segsrf(mprb),
     &    minrbSrf,nFSM,iFSM(4),irbSrf,jrbSrf,rbSrf(mprb),iRptSrf,
     &    jRptSrf,RptSrf(mElt),ifPupilImg,DBias,ifFEX,numspot(numseg),
     &    wlens,m_err_jwst,NumChkThr,NumOverThreshold,ifRetRefSrf,iWF

        Integer, save :: ifRetUserSrf

        REAL*8 :: zEltNom,
     &    ToutNom(7,7),meanspotx(numseg),
     &    ChfRayPosNom(3),ChfRayDirNom(3),xGridNom(3),yGridNom(3),
     &    zGridNom(3),FluxNom,WavelenNom,QE,meanspoty(numseg),
     &    LensData(5)

        REAL*8 :: DMlim,Tfsm(6,2),pv(2),pixelSize,oldRptElt(mprb),
     &    newRptElt(mprb),EltPix(2),WinCenPix(2),StopVec(3),
     &    transMaskThreshold,rotMaskThreshold,sigJitterX,sigJitterY,
     &    sigCrosstalk,sigReadNoise

        Character :: fnameNom*100

        Logical :: SystemCalib_Done
	Logical, save :: first_entry = .TRUE.

C  Run parameters
        DATA
     &    EltPix         /0d0,0d0/,
     &    WinCenPix      /-0.5d0,-0.5d0/,
     &    LensElt	     /59/,
     &	  LensData       /177.07d0,221.82d0,262.17d0,321.96d0,696.15d0/

        SAVE

	LOGICAL, PARAMETER :: ReleaseMemory = .FALSE.,
     &                        WriteToFile   = .FALSE.,
     &                        Debug         = .FALSE.

        SystemCalib_Done=.FALSE.

	IF (Debug) THEN
	  P=6
	ELSE
	  IF (WriteToFile) THEN
	    P=7
	    OPEN(UNIT=P, STATUS='SCRATCH')
	  END IF
	END IF
	MsgsOn = (Debug .OR. WriteToFile)  ! Messages On/Off

	IF (first_entry.OR.ReleaseMemory) THEN
          allocate(PixArray(mPix,mPix),
     &             OPDMat(mpts, mpts), RaySpot(mRay,2),
     &             psiNom(3,mElt),vptNom(3,mElt),rptNom(3,mElt),
     &             pNom(3,mElt),xNom(3,mElt),yNom(3,mElt),
     &             zNom(3,mElt),KrNom(mElt),KcNom(mElt),
     &             IndRefNom(mElt),nObsNom(mElt),ObsTypeNom(1,mElt),
     &             ObsVecNom(6,mObs,mElt),
     &             SrfMetPosNom(3,mMetPos,mMetSrf),
     &             TEltNom(6,6,mElt), stat=m_err_jwst)
          IF (m_err_jwst /= 0) THEN
            call macos_memory_failure('jwst: allocate failed!')
            stop "*** Memory allocation failed ***"
          END IF
        END IF

! ----------------------------------------------------------------------
! Initialisation
! ----------------------------------------------------------------------

        print*,'-->GMI: check-point 1, beginning of GMI'

        ! --------------------------------------------------------------
        ! define flag parameters  (extract or use default values.
        ! --------------------------------------------------------------
        CALL ExtractFlagParameters

        print*,'-->GMI: check-point 2, after ExtractFlagParameters()'


        ! --------------------------------------------------------------
        ! Initialization section performed first call only
        ! --------------------------------------------------------------
        IF (ifInitOptics) THEN
          IF (fnameNom .NE. fname) ifInitOptics=.FALSE.
        END IF

        IF (.NOT.ifInitOptics) CALL ObtainNominalSettings ! Rx load here

        print*,'-->GMI: check-point 3, after ObtainNominalSettings()'

        ! Added 10/10/2018 -jzlou
        ! Save aperture stop setting after loading Rx, to be restored after reset 
        ! operations, since reset operations below would unset ifStopSet value.
        rxStopSetSave = ifStopSet


        ! --------------------------------------------------------------
        ! Part of initialisation
        ! --------------------------------------------------------------
	!
	! set array sizes
	!
	nPix     = cPix
	nGridPts = cGrid-1
	npts     = nGridPts-1

	!
	! Reset nominal prescription if not first run
	!
	IF (ifInitOptics) CALL SetToNominalSettings

	!
	! Reset counters
	!
        command='RESET'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &              OPDMat,RaySpot,RMSWFE,PixArray)

        ! Added 10/10/2018 -jzlou
        ! Restore aperture stop setting to saved value, see above
        ifStopSet = rxStopSetSave 

        print*,'-->GMI: check-point 4, after reset, ifStopSet =',
     &         ifStopSet

! ----------------------------------------------------------------------
! Begin normal computation here
! ----------------------------------------------------------------------

        ! --------------------------------------------------------------
        ! Apply perturbation to Optical System
        ! --------------------------------------------------------------
	CALL ApplyPerturbationToOpticalSystem

        print*,'-->GMI: check-point 5, after apply perturbs'


        ! --------------------------------------------------------------
        ! Update Source
        ! --------------------------------------------------------------
        !IF ((ifFEX.GE.1).OR.ifpfa) THEN
        IF ((ifFEX.GE.1).OR.ifpfa.OR.(StopType>-9999)) THEN
          !
          ! Set detector sub-window
          !
#if 0
           command='WINDOW'
           CARG(1)='TOUT'
           DARG(1)=pixelSize
           DARG(2)=EltPix(1)  ! EltPix(1:2) is set to (0,0) in previous DATA stmt , 
           DARG(3)=EltPix(2)  ! which are pixel coords of element vertex
	   WinCenPix(1:2)=0d0 ! updated by jzlou
           DARG(4)=WinCenPix(1)
           DARG(5)=WinCenPix(2)
           CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                 OPDMat,RaySpot,RMSWFE,PixArray)
#endif

          !IF ((ifFEX.GE.1) .OR. (StopType>-1d20)) THEN
          IF (StopType>-9999) THEN
            !
            !  Set stop
            !
            print*,'--> GMI: setting STOP now ...'
            print*,'    GMI: ifFEX =', ifFEX
            print*,'    GMI: StopType =',StopType 
            command='STOP'
            IF (StopType.EQ.0) THEN
              CARG(1)='OBJECT'
              CARG(2)='YES'
              DARG(1)=StopVec(1)
              DARG(2)=StopVec(2)
              DARG(3)=StopVec(3)
            ELSE IF (StopType.EQ.1) THEN
              CARG(1)='ELT'
              IARG(1)=INT(StopVec(1))
              DARG(1)=StopVec(2)
              DARG(2)=StopVec(3)
            END IF
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
          END IF
           
          !  Set Pixel Field Point (PFP, MACOS pg. 104)
          IF (ifpfa) THEN
            IF (pfa(1).EQ.0) THEN
              IF (MsgsOn) WRITE(P,*)'Finding Field Point =',
     &          pfa(2),pfa(3)
              command='PFP'
              IARG(1)=nElt
              CALL DZERO(DARG,9)
              DARG(1)=pixelSize
              DARG(2)=pfa(2)
              DARG(3)=pfa(3)
              CARG(1)='YES'
              CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                    OPDMat,RaySpot,RMSWFE,PixArray)
            ELSE IF (pfa(1).EQ.1) THEN
              IF (MsgsOn) WRITE(P,*)'Perturbing Source by ',
     &          pfa(2),pfa(3)
              DO iNode=1,6
                DARG(iNode)=0.d0
              END DO
              command='PERTURB'
              IF (MsgsOn) WRITE(P,*)'GLOBAL'
              CARG(1)='GLOBAL'
              IARG(1)=0
              DARG(1)=pfa(2)
              DARG(2)=pfa(3)
              CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &          OPDMat,RaySpot,RMSWFE,PixArray)
            ELSE IF (pfa(1).EQ.2) THEN
              IF (MsgsOn)
     &          WRITE(P,*)'Re-Define ChfRayDir: ',pfa(2),pfa(3),pfa(4)
              IF (MsgsOn)
     &          WRITE(P,*)'Re-Define ChfRayPos: ',pfa(5),pfa(6),pfa(7)
              DO iNode = 1,3
                 ChfRayDir(iNode) = pfa(iNode+1)
                 ChfRayPos(iNode) = pfa(iNode+4)
              END DO
!	      CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
!     &	        OPDMat,RaySpot,RMSWFE,PixArray)
            END IF
          END IF ! end of ifpfa check

          ! --------------------------------------
          ! Optionally perform system optimization.
          ! Elements to calibrate are specified in
          ! the MACOS optical prescription.
          ! Added by jzlou, 06/15/2007
          ! --------------------------------------
          IF (ifSysCalib) THEN
            !print*,'***GMI: calling MACOS optimizer ...'
            CALL SystemCalib
            IF (ifpfa) THEN
              ! STOP was set earlier but turned off by the
              ! optimizer. Since STOP is not changed, just
              ! set the STOP flag back on
              ifStopSet=.TRUE.
            END IF
            SystemCalib_Done=.TRUE.
          END IF


          ! -----------------------------
          ! Find exit pupil
          ! -----------------------------
	  IF (ifFEX.EQ.1) THEN
            command='FEX'
            IARG(1)=nElt-1
            CARG(1)='YES'
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
	  ELSEIF (ifFEX.EQ.2) THEN
	    IF (MsgsOn) WRITE(P,*)'***Exit Pupil read in and set******'
	    KrElt(nElt-1) = USER(1)
	    zElt(nElt-1) = -USER(1)
	    psiElt(1,nElt-1) = USER(2)
	    psiElt(2,nElt-1) = USER(3)
	    psiElt(3,nElt-1) = USER(4)
	    vptElt(1,nElt-1) = USER(5)
	    vptElt(2,nElt-1) = USER(6)
	    vptElt(3,nElt-1) = USER(7)
	    rptElt(1,nElt-1) = USER(5)
	    rptElt(2,nElt-1) = USER(6)
	    rptElt(3,nElt-1) = USER(7)
	  END IF

        END IF !of ifFEX

        ! --------------------------------------
        ! Optionally perform system optimization.
        ! Elements to calibrate are specified in
        ! the MACOS optical prescription.
        ! Added by jzlou, 06/15/2007
        ! --------------------------------------
        IF (ifSysCalib .AND. (.NOT.SystemCalib_Done)) THEN
          CALL SystemCalib
          SystemCalib_Done=.TRUE.
        END IF

        ! -----------------------------
        ! Output Exit Pupil parameters
        ! -----------------------------
        IF (ifRetRefSrf.EQ.1) THEN
	  USER(1) = KrElt(nElt-1)
	  USER(2) = psiElt(1,nElt-1)
	  USER(3) = psiElt(2,nElt-1)
	  USER(4) = psiElt(3,nElt-1)
	  USER(5) = vptElt(1,nElt-1)
	  USER(6) = vptElt(2,nElt-1)
	  USER(7) = vptElt(3,nElt-1)
	END IF

        ! ------------------------------
        ! Output User Surface parameters
        ! ------------------------------
        If (ifRetUserSrf.GT.0) Then
          If (MsgsOn) Then
           print*,'**** GMI: ifRetUserSrf =',ifRetUserSrf
           print*,'**** GMI: KrElt(ifRetUserSrf) =',
     &            KrElt(ifRetUserSrf)
           print*,'**** GMI: psiElt(1:3,ifRetUserSrf) =',
     &            psiElt(1:3,ifRetUserSrf)
           print*,'**** GMI: vptElt(1:3,ifRetUserSrf) =',
     &            vptElt(1:3,ifRetUserSrf)
          End If
          USER(9) = KrElt(ifRetUserSrf)
          USER(10:12) = psiElt(1:3,ifRetUserSrf)
          USER(13:15) = vptElt(1:3,ifRetUserSrf)
        End If


	! ------------------------------------------------------
        ! Optionally perform system (1-D) metrology measurement.
        ! Metrology definition is contained in the MACOS optical 
        ! prescription.
        ! Added by jzlou, 02/21/2012
        ! ------------------------------------------------------
	ifMet=.FALSE.
	IF (ifMetCalc) THEN
          CALL MetCalc
	  ! metrology measurements are in the 1-D array 
          ! metMeasBuf(1:nMetMeas) 
	  ifMet=.TRUE.
        END IF

	ifSpf=.FALSE.
        IF (ifSpfCalc) THEN
          CALL SpfCalc
          ! Steward Platform measurements are in the 1-D array
          ! pfMeasBuf(1:nPfMeas)
          ifSpf=.TRUE.
        END IF

	 IF (ifFDP>0) THEN
	   CALL ExecFDPCmd
	 END IF


        ! --------------------------------------------------------------
        !  Set wavelengths and generate image
        ! --------------------------------------------------------------
        IF (Npimg.LT.2) THEN
	  IF (MsgsOn) WRITE(P,*)'%#%#%#%#% Npimg =',Npimg
          IF ((Npimg.EQ.0).OR.(pimg(1).LE.0d0)) THEN
	    IF (MsgsOn)
     &	      WRITE(P,*)'WavelenNom =',WavelenNom,' FluxNom =',FluxNom
            pimg(1) = WavelenNom
            pimg(2) = FluxNom
          ELSE IF (Npimg.EQ.1) THEN
            pimg(2) = FluxNom
          END IF
	  Npimg = 2
	END IF

        ! --------------------------------------------------------------
        !  Compose and determine PIX values
        ! --------------------------------------------------------------
        IF (ifPIX.EQ.1) THEN
          CALL ComposePIX
        ELSE
          PIX = 0d0
        END IF

        ! --------------------------------------------------------------
        !  Determine OPD, SPOT and centroid values
        ! --------------------------------------------------------------
        IF (ifOPDCalc.OR.(ifSPOT.NE.0)) THEN
          !
          ! define wavelength
          !
          Wavelen = pimg(1)
          Flux    = pimg(1)
          command = 'MODIFY'
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                OPDMat,RaySpot,RMSWFE,PixArray)
          !
          ! SPOT and centroid calculations
          !
          IF (ifSPOT.EQ.0) THEN
            SPOT = 0d0
          ELSE
            command = 'SPC'
            CARG(1) = 'ELT'
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &	      OPDMat,RaySpot,RMSWFE,PixArray)

            command='SPOT'
            CARG(1)='BEA'
            IF (ifSPOT.NE.0) THEN
              IARG(1)=ifSPOT
              print*,
     &        '* Calc SPOT at elt ',ifSPOT,' using Beam Coord'
            ELSE
              IARG(1)=nElt
              print*,
     &        '* Calc SPOT at elt ',nElt,' using Beam Coord'
            END IF
            IF (MsgsOn) WRITE(P,*)'SPOT diagram at Element ',IARG(1)
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &	      OPDMat,RaySpot,RMSWFE,PixArray)

!NS         DO j=1,nRay
!NS           DO i=1,2
!NS             SPOT(i,j)=RaySpot(j,i)
!NS           END DO
!NS         END DO
            SPOT = TRANSPOSE(RaySpot(1:nRay,1:2))

            cent(1)=xcent
            cent(2)=ycent
          END IF
          !
          ! Generate WFE and OPD map output
          !
          IF (.NOT.ifOPDCalc) THEN
            OPD = 0d0
          ELSE
            command='OPD'
            IF (MsgsOn) WRITE(P,*)'OPD Elt =',ifOPD
            !WRITE(P,*)'OPD Elt =',ifOPD
            IF (ifOPD.LE.0) THEN
              IARG(1) = nElt+ifOPD
            ELSE
              IARG(1) = ifOPD
            END IF
	    !print*,'** OPD at element: ',IARG(1) 
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
            WFError=RMSWFE

            k=0
            DO j=1,nGridPts
              DO i=1,nGridPts
                k=k+1
                OPD(k)=OPDMat(i,j)
	        OPDMask(k)=opdRayMask(i,j)
              END DO
            END DO
	  END IF

	END IF !of (ifOPD.OR.ifSPOT) check

! ----------------------------------------------------------------------
!  Exit: Reset flag and return
! ----------------------------------------------------------------------
        IF (ReleaseMemory) THEN
          deallocate(PixArray,
     &      OPDMat, RaySpot,psiNom,vptNom,rptNom,pNom,xNom,yNom,
     &      zNom,KrNom,KcNom,IndRefNom,nObsNom,ObsTypeNom,ObsVecNom,
     &      TEltNom, stat=m_err_jwst)
          IF (m_err_jwst /= 0) THEN
             call macos_memory_failure ('jwst: deallocate failed!')
             stop "*** Memory deallocation failed ***"
          END IF
        END IF

        ifInitOptics = .TRUE.
        first_entry  = .FALSE.

        IF (WriteToFile .AND.(.NOT. Debug)) close(P)
        RETURN

        ! ----- Subroutines contained in subroutine GMI_DVR-------------
        CONTAINS
        ! --------------------------------------------------------------

        SUBROUTINE ObtainNominalSettings
          !
          ! Load data and set nominal parameters
          !
          command='OLD'
          CARG(1)=fname
          print*,'GMI: loading Rx ',fname
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &    	      OPDMat,RaySpot,RMSWFE,PixArray)
          !
          ! Turn off plotting
          !
          command='NONE'
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &    	      OPDMat,RaySpot,RMSWFE,PixArray)
          !
          ! Turn off gaps if number of rays is small
          !
          IF (mdttl.LT.512) THEN
            gap=0d0
          END IF
          !
          ! Save nominal element data
          !
          fnameNom   = fname
          WavelenNom = Wavelen
          FluxNom    = Flux
          zEltNom    = zElt(nElt-1)
          DO i=1,3
            ChfRayPosNom(i) = ChfRayPos(i)
            ChfRayDirNom(i) = ChfRayDir(i)
            xGridNom(i)     = xGrid(i)
            yGridNom(i)     = yGrid(i)
            zGridNom(i)     = zGrid(i)
          END DO
          DO iElt=1,nElt
            KrNom(iElt)        = KrElt(iElt)
            KcNom(iElt)        = KcElt(iElt)
            nObsNom(iElt)      = nObs(iElt)
            ObsTypeNom(1,iElt) = ObsType(1,iElt)
            DO i=1,3
              psiNom(i,iElt)      = psiElt(i,iElt)
              vptNom(i,iElt)      = vptElt(i,iElt)
              rptNom(i,iElt)      = rptElt(i,iElt)
              pNom(i,iElt)        = pMon(i,iElt)
              xNom(i,iElt)        = xMon(i,iElt)
              yNom(i,iElt)        = yMon(i,iElt)
              zNom(i,iElt)        = zMon(i,iElt)
              ObsVecNom(i,1,iElt) = ObsVec(i,1,iElt)
              CALL DEQUATE(TEltNom(1,1,iElt),TElt(1,1,iElt),36)
            END DO
            CALL DEQUATE(ToutNom,Tout,49)
	    If (nMetPos(iElt)>0) Then
              Do i=1,nMetPos(iElt)
                SrfMetPosNom(1:3,i,iEltToMetSrf(iElt))
     &            =SrfMetPos(1:3,i,iEltToMetSrf(iElt))
              End Do
            End If 
          END DO
        END SUBROUTINE ObtainNominalSettings
         
        ! --------------------------------------------------------------
         
        SUBROUTINE SetToNominalSettings

          Flux         = FluxNom
          Wavelen      = WavelenNom
          zElt(nElt-1) = zEltNom
          DO i=1,3
            ChfRayPos(i) = ChfRayPosNom(i)
            ChfRayDir(i) = ChfRayDirNom(i)
            xGrid(i)     = xGridNom(i)
            yGrid(i)     = yGridNom(i)
            zGrid(i)     = zGridNom(i)
          END DO
          DO iElt=1,nElt
            KrElt(iElt)        = KrNom(iElt)
            KcElt(iElt)        = KcNom(iElt)
            nObs(iElt)         = nObsNom(iElt)
            ObsType(1,iElt)    = ObsTypeNom(1,iElt)
            DO i=1,3
              psiElt(i,iElt)   = psiNom(i,iElt)
              vptElt(i,iElt)   = vptNom(i,iElt)
              rptElt(i,iElt)   = rptNom(i,iElt)
              pMon(i,iElt)     = pNom(i,iElt)
              xMon(i,iElt)     = xNom(i,iElt)
              yMon(i,iElt)     = yNom(i,iElt)
              zMon(i,iElt)     = zNom(i,iElt)
              ObsVec(i,1,iElt) = ObsVecNom(i,1,iElt)
            END DO
            CALL DEQUATE(TElt(1,1,iElt),TEltNom(1,1,iElt),36)
            If (nMetPos(iElt)>0) Then
	      Do i=1,nMetPos(iElt)
	        SrfMetPos(1:3,i,iEltToMetSrf(iElt))
     &            =SrfMetPosNom(1:3,i,iEltToMetSrf(iElt))
	      End Do
	    End If
          END DO
          CALL DEQUATE(Tout,ToutNom,49)

        END SUBROUTINE SetToNominalSettings
         
        ! --------------------------------------------------------------
         
        SUBROUTINE ExtractFlagParameters

          IF (ifpflg) THEN
            ifFEX =pflg(1); ifPupilImg =pflg(2); cGrid=pflg(3)
            cPix=pflg(4); DMlim=pflg(5); ifOPD=pflg(6);
            ifShotNoise=pflg(7); sigReadNoise=pflg(8)
            sigJitterX=pflg(9); sigJitterY=pflg(10)
            sigCrosstalk=pflg(11); StartSeed=pflg(12)
            transMaskThreshold=pflg(13); rotMaskThreshold=pflg(14)
            pixelSize=pflg(15); nzern=pflg(16)
            QE=pflg(17); DBias=pflg(18); wlens=pflg(19)
            ifPIX=pflg(20); ifRetRefSrf=pflg(21)
            ifSPOT=pflg(22)
            ifPIXflip = (INT(pflg(23)).EQ.1)
            ifPIXSpotDetCheck  = (INT(pflg(24)).EQ.1)
	    ifSysCalib = (INT(pflg(25)).EQ.1)

            ifOPDCalc = (INT(ifOPD).NE.9999)

	    ifPIXElt=pflg(26)
	    ifMetCalc=pflg(27)
	    ifSpfCalc=pflg(28)

            ifRetUserSrf=pflg(29) 
	
            ipflg = 30

            ! ------------------------------------
            ! Stop vector
            ! ------------------------------------
            ipflg = ipflg + 1; i=pflg(ipflg)
!           print*,'**GMI Stop-vector: ipflg, pflg(ipflg) =',
!    &             ipflg,pflg(ipflg)
	    If (i /= -9999) Then
              ! STOP defined in param
              StopType = pflg(ipflg)
              print*,' * ipflg,StopType,pflg(ipflg) =',
     &                   ipflg,StopType,pflg(ipflg)
              IF (.TRUE.) 
     &         WRITE(*,*)'**ipflg,StopType =',ipflg,StopType
              DO i=1,3
                ipflg = ipflg + 1
                StopVec(i) = pflg(ipflg)
                IF (.TRUE.) 
     &           WRITE(*,*)'ipflg,StopVec =',ipflg,StopVec(i)
              END DO
	    Else
              ! i = pflg(ipflg) = -9999
              WRITE(*,*)
     &        '**STOP not defined,ipflg,pflg(ipflg) =',
     &        ipflg,pflg(ipflg)
	      StopType = -(9999+1)  ! StopType value for STOP undefined
              ipflg = ipflg + 3  ! skip next three values 
	    End If

            print*,'** GMI: after STOP: ipflg =', ipflg
            !STOP

            ! ------------------------------------
            ! FSM
            ! ------------------------------------
            ipflg = ipflg + 1
            nFSM = pflg(ipflg)
	    If (nFSM/=9999 .and. nFSM>9) Then
              IF (nFSM.GT.4) THEN
                WRITE(0,*)'============= too many FSMs!!! ============='
                RETURN
              END IF
              DO i=1,nFSM
                ipflg = ipflg + 1
                iFSM(i) = pflg(ipflg)
              END DO

              DO j=1,2
                DO i=1,6
                  ipflg = ipflg + 1
                  Tfsm(i,j) = pflg(ipflg)
                  IF (MsgsOn) WRITE(P,*)'Tfsm(',i,',',j,') =',Tfsm(i,j)
                END DO
              END DO
	    Else
	      ! FSM not defined
	      iFSM=-1
	    End If

	    ! ------------------------------------
            ! FDP command
	    ! ------------------------------------ 
	    ipflg = ipflg + 1
            ifFDP = pflg(ipflg)
            If (ifFDP==9999 .OR. ifFDP<1) ifFDP=-1  ! don't run FDP 


            ! ------------------------------------
            ! GridSrf (pgrid)
            ! ------------------------------------
            lgridSrf = 0
!NS         DO i=1,mElt
!NS           lgridSrf(i)=0
!NS         END DO
            ipflg = ipflg + 1
            igridSrf = pflg(ipflg)
	    If (igridSrf /= 9999) Then
              IF (MsgsOn) WRITE(P,*)'igridSrf =',igridSrf
              ipflg = ipflg + 1
              jgridSrf = pflg(ipflg)
              IF (MsgsOn) WRITE(P,*)'jgridSrf =',jgridSrf
              DO i=1,igridSrf*jgridSrf
                ipflg = ipflg + 1
                gridSrf(i) = pflg(ipflg)
                IF (MsgsOn) WRITE(P,*)'gridSrf(',i,') =',gridSrf(i)
                lgridSrf(INT(pflg(ipflg)))=1
              END DO
	    End If

            ! ------------------------------------
            ! ZernSrf (pzern)
            ! ------------------------------------
            lzernSrf = 0
!NS         DO i=1,mElt
!NS           lzernSrf(i)=0
!NS         END DO
            ipflg = ipflg + 1
            izernSrf = pflg(ipflg)
	    If (izernSrf /= 9999) Then
              IF (MsgsOn) WRITE(P,*)'izernSrf =',izernSrf
              ipflg = ipflg + 1
              jzernSrf = pflg(ipflg)
              IF (MsgsOn) WRITE(P,*)'jzernSrf =',jzernSrf
              DO i=1,izernSrf*jzernSrf
                ipflg = ipflg + 1
                zernSrf(i) = pflg(ipflg)
                IF (MsgsOn) WRITE(P,*)'zernSrf(',i,') =',zernSrf(i)
                lzernSrf(INT(pflg(ipflg)))=1
              END DO
	    End If

            ! ------------------------------------
            ! Read in surfaces to apply pdm
            ! ------------------------------------
            ldmSrf = 0
!NS         DO i=1,mElt
!NS           ldmSrf(i)=0
!NS         END DO
            ipflg = ipflg + 1
            idmSrf = pflg(ipflg)
	    If (idmSrf /= 9999) Then
              IF (MsgsOn) WRITE(P,*)'idmSrf =',idmSrf
              ipflg = ipflg + 1
              jdmSrf = pflg(ipflg)
              IF (MsgsOn) WRITE(P,*)'jdmSrf =',jdmSrf
              DO i=1,idmSrf*jdmSrf
                ipflg = ipflg + 1
                dmSrf(i) = pflg(ipflg)
                IF (MsgsOn) WRITE(P,*)'dmSrf(',i,') =',dmSrf(i)
                ldmSrf(INT(pflg(ipflg)))=1
              END DO

              ! ------------------------------------
              ! renumber DM surfaces by seg. number
              ! ------------------------------------
              DO j=1,jdmSrf
                iSegmin = 1000
                DO i=1,idmSrf
                  iSegmin = min(iSegmin,dmSrf((j-1)*idmSrf+i))
                END DO
                iSegmin = iSegmin - 1
                IF (MsgsOn) WRITE(P,*)'iSegmin =',iSegmin
                DO i=1,idmSrf
                  segSrf((j-1)*idmSrf+i) = dmSrf((j-1)*idmSrf+i)
     &                                         - iSegmin + (j-1)*idmSrf
                  IF (MsgsOn) WRITE(P,*)
     &             'segSrf(',(j-1)*idmSrf+i,') =',segSrf((j-1)*idmSrf+i)
                END DO
              END DO
	    End If

            ! ------------------------------------
            ! Read in surfaces to apply prb to
            ! ------------------------------------
            ipflg = ipflg + 1
            irbSrf = pflg(ipflg)
            IF (MsgsOn) WRITE(P,*)'irbSrf =',irbSrf
	    If (irbSrf /= 9999) Then
              ipflg = ipflg + 1
              jrbSrf = pflg(ipflg)
              IF (MsgsOn) WRITE(P,*)'jrbSrf =',jrbSrf
              DO i=1,irbSrf*jrbSrf
                ipflg = ipflg + 1
                rbSrf(i) = pflg(ipflg)
                IF (MsgsOn) WRITE(P,*)'rbSrf(',i,') =',rbSrf(i)
              END DO
	    Else
	      jrbSrf=0 
	    End If

            ! ------------------------------------
            ! Read in surfaces to change RptElt
            ! ------------------------------------
            ipflg = ipflg + 1
            iRptSrf = pflg(ipflg)
            IF (MsgsOn) WRITE(P,*)'iRptSrf =',iRptSrf
	    If (iRptSrf /= 9999) Then
              ipflg = ipflg + 1
              jRptSrf = pflg(ipflg)
              IF (MsgsOn) WRITE(P,*)'jRptSrf =',jRptSrf
              DO i=1,iRptSrf*jRptSrf
                ipflg = ipflg + 1
                RptSrf(i) = INT(pflg(ipflg))
                IF (MsgsOn) WRITE(P,*)'RptSrf(',i,') =',RptSrf(i)
              END DO
              IF (mprb.LT.iRptSrf*3) THEN
                WRITE(0,*)'newRptElt is not big enough'
                RETURN
              END IF
              DO i=1,iRptSrf*3
                ipflg = ipflg + 1
                newRptElt(i) = pflg(ipflg)
                IF (MsgsOn) WRITE(P,*)'newRptElt(',i,') =',newRptElt(i)
              END DO
	    End If

            ! ------------------------------------
            ! Read in Exit Pupil parameters
            ! ------------------------------------
            IF (ifFEX.EQ.2) THEN
              DO i=1,7
                ipflg = ipflg + 1
                USER(i) = pflg(ipflg)
                IF (MsgsOn) WRITE(P,*)'USER(',i,') =',USER(i)
              END DO
            END IF
          ELSE
	    ! .not. ifpflg
            IF (MsgsOn) WRITE(P,*)' into else IF (ifpflg) ',ifpflg
            ifFEX              = 1
            ifPupilImg         = 0
            cGrid              = 66
            cPix               = 64
            DMlim              = 10d0
            ifOPD              = 0
            ifShotNoise        = 0
            sigReadNoise       = 0
            sigJitterX         = 0
            sigJitterY         = 0
            sigCrosstalk       = 0
            StartSeed          = 0
            transMaskThreshold = 1d22
            rotMaskThreshold   = 1d22
            pixelSize          = 1.6536132D-02
            nzern              = 8
            QE                 = 1.d0
            DBias              = 0
            ifSPOT             = 1
            ifPIX              = 1
            ifPIXflip          = .FALSE.
            ifRetRefSrf        = 0
            ifPIXSpotDetCheck  = .FALSE.
          END IF
        END SUBROUTINE ExtractFlagParameters
         
        ! --------------------------------------------------------------
        
 
	SUBROUTINE ApplyPerturbationToOpticalSystem

        ! ------------------------------------
        ! Implement Zernike deformations
        ! ------------------------------------
	IF (MsgsOn) WRITE(P,*)'ifpzern =',ifpzern
	IF (ifpzern) THEN
	  iSeg = 0
	  DO j=1,jzernSrf
	    k = 0
	    DO  i=1,izernSrf
	      iSeg = iSeg + 1
	      iElt = zernSrf(iSeg)
	      IF (iElt.GT.0) THEN
		IF (MsgsOn)
     &		  WRITE(P,*)'zern app. to zernSrf ',iSeg,', Elt ',iElt
		SrfType(iElt)=8
		DO iNode=4,nzern+3   ! skip tip, tilt, piston
		  k=k+1
		  ZernCoef(iNode,iElt)=pzern(k)
		  IF (MsgsOn)
     &		    WRITE(P,*)'ZernCoef(',iNode,iElt,') =',pzern(k)
		END DO
	      ELSE IF (iElt.LT.0) THEN
		IF (MsgsOn) WRITE(P,*)'negative value for element'
	      ELSE
	      END IF
	    END DO
	  END DO
	ELSE
	  iSeg = 0
	  DO j=1,jzernSrf
	    DO  i=1,izernSrf
	      iSeg = iSeg + 1
	      iElt = zernSrf(iSeg)
	      DO iNode=1,15
		ZernCoef(iNode,iElt)=0.d0
	      END DO
	      SrfType(iElt)=2
	    END DO
	  END DO
	END IF !of ifpzern check

	print*,'GMI-applyPert: check-point 1'
	
 	print*,'Gridded surfaces: jgridSrf =',jgridSrf 
 	print*,'Gridded surfaces: igridSrf =',igridSrf 
        print*,'GMI: mgrid = ',mgrid
!	print*,'Gridded surfaces: gridSrf =',
!    &    gridSrf(1:jgridSrf*igridSrf)
 
	!  Implement Gridded surfaces
        IF (MsgsOn) WRITE(P,*)'ifpgrid =',ifpgrid
        IF (ifpgrid) THEN
          iSeg = 0
          DO j=1,jgridSrf
            k = 0
            DO  i=1,igridSrf
              iSeg = iSeg + 1
              iElt = gridSrf((j-1)*igridSrf+i)
	      !print*,'Gridded surfaces: iElt =',iElt
	      !print*,'Gridded surfaces:   mgrid =',mgrid
	      !stop
              IF (iElt.GT.0) THEN
                IF (MsgsOn)
     &		  WRITE(P,*)'grid app. to gridSrf ',iSeg,', Elt ',iElt
                If (SrfType(iElt) /= 11) Then
                  ! updated on 05/04/2016 -jzlou
                  ! if Rx SrfType(iElt) is not AsGrData (11), change SrfType(iElt) 
                  ! to GridData (9), otherwise keep SrfType(iElt) as AsGrData 
                  SrfType(iElt)=9
                End If
	        iEltToGridSrf(iElt)=iSeg
		DO jNode=1,mgrid
                  DO iNode=1,mgrid
                    k=k+1
                    GridMat(iNode,jNode,iSeg)=pgrid(k)
		  END DO
                END DO
              ELSE IF (iElt.LT.0) THEN
                IF (MsgsOn) WRITE(P,*)'negative value for element'
		DO jNode=1,mgrid
		  DO iNode=1,mgrid
		    k=k+1
		  END DO
		END DO
		iSeg = iSeg - 1
              ELSE
		iSeg = iSeg - 1
              END IF
            END DO
          END DO
	ELSE
#if 1
	  iSeg = 0
	  DO j=1,jgridSrf
	    DO  i=1,igridSrf
	      iSeg = iSeg + 1
	      iElt = gridSrf((j-1)*igridSrf+i)
	      DO jNode=1,mgrid
		DO iNode=1,mgrid
		  GridMat(iNode,jNode,iSeg)=0.d0
		END DO
	      END DO
	    END DO
	  END DO
#endif
        END IF !of ifpgrid check

	print*,'GMI-applyPert: check-point 2'
 
	! Implement DM (radius of curvature) deformations
        ! This section is for (using grid surface as an example)
        !  1) param.dmSrf and param.gridSrf are both defined in Init file; 
        !  2) 'pgrid' is NOT defined (must be 0d0 in GMI call);
        ! so the dm grid surface will be defined by pdm(k)*InfFncGrid below; -jzlou
        ! 'ifpdm' will be true if the argument 'pdm' to GMI call is not 0d0, in that 
        ! case, pzern and pgrid should both be 0d0. -jzlou
	! 
	! jzlou's extension:
        ! When param.zernSrf and param.gridSrf arrays are both empty in Init file,
        ! and param.dmSrf is defined, lzernSrf and lgridSrf will then be false below,
        ! and ldmSrf will be true; in that case, a perturbation to KrElt(iElt) of
        ! amount pdm(k) will be applied. 
	! 
	IF (MsgsOn) WRITE(P,*)'ifpdm =',ifpdm
	IF (ifpdm) THEN
	  iSeg = 0
	  DO j=1,jdmSrf
	    k = 0
	    DO  i=1,idmSrf
	      iSeg = iSeg + 1
	      iElt =dmSrf(iSeg)
	      IF (iElt.GT.0) THEN
		k = k + 1
		IF (lgridSrf(iElt).EQ.1) THEN
		  IF (MsgsOn)
     &		    WRITE(P,*)'iElt=',iElt,' gridSrf ',segSrf(iSeg),
     &		    ' pdm(',k,') =',pdm(k)
		  SrfType(iElt)=9
		  DO jNode=1,mgrid
		    DO iNode=1,mgrid
		      IF (ifpgrid) THEN
			GridMat(iNode,jNode,segSrf(iSeg))=
     &			  pdm(k)*InfFcnGrid(iNode,jNode) +
     &			  GridMat(iNode,jNode,segSrf(iSeg))
		      ELSE
			GridMat(iNode,jNode,segSrf(iSeg))=
     &			  pdm(k)*InfFcnGrid(iNode,jNode)
		      END IF
		    END DO
		  END DO
		ELSEIF (lzernSrf(iElt).EQ.1) THEN
		  IF (MsgsOn) WRITE(P,*)'pdm(',k,') =',pdm(k)
		  SrfType(iElt)=8
		  DO iNode=1,15
		    IF (ifpzern) THEN
		      ZernCoef(iNode,iElt)=pdm(k)*InfFcnZern(iNode)+
     &		        ZernCoef(iNode,iElt)
		    ELSE
		      ZernCoef(iNode,iElt)=pdm(k)*InfFcnZern(iNode)
		    END IF
		    IF (MsgsOn) WRITE(P,*)'ZernCoef(',iNode,iElt,') =',
     &		      ZernCoef(iNode,iElt)
		  END DO
		ELSEIF (ldmSrf(iElt).EQ.1) THEN
	           If (.true.) WRITE(P,*)'pdm(',k,') =',pdm(k),
     &                         ' applied to KrElt at Elt ',iElt
	           KrElt(iElt)=KrElt(iElt)+pdm(k)
	        ELSE
		  WRITE(0,*)'Error: ',
     &		      'dmSrf does not overlap zernSrf or gridSrf'
		  RETURN
		END IF
	      END IF
	    END DO
	  END DO
	END IF
	!STOP ! test only

        print*,'GMI-applyPert: check-point 3'

	print*,'*** ifprb =',ifprb
        print*,'*** jrbSrf =', jrbSrf
        print*,'*** irbSrf =',irbSrf
	 
	! Apply rigid body perturbations
        IF (ifprb) THEN
	  iSeg = 0
	  DO j=1,jrbSrf-1
            k=0
	    kk=0   ! Counter for alternate rotation points
            DO i=1,irbSrf
	      iSeg = iSeg + 1
	      iElt = rbSrf(iSeg)
	      IF (EltGrp(0,abs(iElt))==0) THEN
                ! Single element perturb
                IF (iElt.LT.0) THEN
                  ! Check threshold for masking and mask out
                  DARG(1:6)=prb(k+1:k+6)
                  k=k+6

                  ifMask = ((ABS(DARG(1)).GE.rotMaskThreshold).OR.
     &                     (ABS(DARG(2)).GE.rotMaskThreshold).OR.
     &                     (ABS(DARG(4)).GE.transMaskThreshold).OR.
     &                     (ABS(DARG(5)).GE.transMaskThreshold).OR.
     &                     (ABS(DARG(6)).GE.transMaskThreshold))

!NS               DO iNode=1,6
!NS                 k=k+1
!NS                 DARG(iNode)=prb(k)
!NS                 IF (ABS(prb(k)).GE.transMaskThreshold) ifMask=.TRUE.
!NS               END DO
!NS               DO iNode=5,6
!NS                 IF (ABS(DARG(iNode)).GE.rotMaskThreshold)ifMask=.TRUE.
!NS               END DO

	          IF (ifMask) THEN
                    nObs(-iElt)       = 1
                    ObsType(1,-iElt)  = 1
                    ObsVec(1,1,-iElt) = 8000d0
                    ObsVec(2,1,-iElt) = 0d0
                    ObsVec(3,1,-iElt) = 0d0
                  ELSE
                    command='PERTURB'
                    CALL ApplyPerturbationToOpticalSystem_Prb
                  END IF
                !ELSE IF (iElt.GT.0) THEN   ! Perturb elements
                ELSE IF (iElt.GE.0) THEN   ! Perturb elements including src
                  DARG=prb(k+1:k+6)
                  k=k+6
                  command='PERTURB'
	          If (iElt==9999) Then
	            print*,'prb(1:20) =',prb(1:20)
	            print*,' **GMIG: B4 calling ApplyPrb, DARG =',DARG(1:6)
	            stop
	          End If 
                  print*,'**B4 calling ApplyPert, iElt =',iElt
                  CALL ApplyPerturbationToOpticalSystem_Prb
                ELSE
                  k=k+6
                END IF

	      ELSE IF (EltGrp(0,abs(iElt))>0) THEN
                ! Group elements perturb
                DARG=prb(k+1:k+6)
	        !print*,'prb(k+1:k+6) =',prb(k+1:k+6)
	        print*,'*** GPER on iElt =',abs(iElt)
                k=k+6
                command='GPERTURB'
                CALL ApplyPerturbationToOpticalSystem_Prb
              END IF
	    END DO ! irbSrf loop
          END DO ! jrbSrf loop

          print*,'GMI-applyPert: check-point 4'
	  
	  If (nFSM /= 9999) Then 
	    !  Actuate FSM in tilt
	    DO i=1,nFSM
	      IF (MsgsOn) WRITE(P,*)'prb(',6*irbSrf+1,') =',prb(6*irbSrf+1)
	      IF (MsgsOn) WRITE(P,*)'prb(',6*irbSrf+2,') =',prb(6*irbSrf+2)
	      pv(1)=prb(6*irbSrf+1)
	      pv(2)=prb(6*irbSrf+2)
              CALL DMPROD(DARG,Tfsm,pv,6,2,1)
              command='PERTURB'
              CARG(1)='GLOBAL'
              IARG(1)=iFSM(i)
              CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
	    END DO
	  End If
        END IF ! of perturbations
        END SUBROUTINE ApplyPerturbationToOpticalSystem
         
        ! --------------------------------------------------------------
         
        SUBROUTINE ApplyPerturbationToOpticalSystem_Prb
        IMPLICIT NONE
        INTEGER :: jElt, iTmp
        REAL*8  :: Q(3,3),dQ(3,3),tmpA(3),tmpD(3)


          jElt = ABS(iElt)
          iTmp = (jrbSrf-1)*irbSrf+i

	  print*,'***In ApplyPerturbationToOpticalSystem_Prb, jElt =',jElt

          !command='PERTURB'  ! command defined in calling routine
          IF (MsgsOn) WRITE(P,*)'rbSrf(',iTmp,
     &			        ') = ',rbSrf(iTmp)

	  IF (rbSrf(iTmp).EQ.0) THEN
	    IF (MsgsOn) WRITE(P,*)j,i,'GLOBAL'
	    CARG(1)='GLOBAL'

	  ELSEIF (rbSrf(iTmp).EQ.1) THEN
	    IF (MsgsOn) WRITE(P,*)j,i,'ELEMENT'
            CARG(1)='ELEMENT'

	  ELSEIF (rbSrf(iTmp).EQ.2) THEN
            tmpD = RptElt(:,jElt)-newRptElt(kk+1:kk+3)    ! = RptElt_old - RptElt_new
            tmpA = DARG(1:3)
            RptElt(:,jElt) = newRptElt(kk+1:kk+3)
            kk=kk+3
	    CARG(1)='GLOBAL'  ! actually ELEMENT but not yet implemented -> but makes no sense anyway

	  ELSEIF (rbSrf(iTmp).EQ.3) THEN
            tmpD = RptElt(:,jElt)-newRptElt(kk+1:kk+3)    ! = RptElt_old - RptElt_new
            tmpA = DARG(1:3)
            RptElt(:,jElt) = newRptElt(kk+1:kk+3)
            kk=kk+3
	    CARG(1)='GLOBAL'
	  END IF
           
          ! Perturb Element
	 !print*,'**** Perturb elt =',jElt
         !print*,'**** DARG =',DARG(1:6)
          !if (jElt<0) STOP

          IARG(1)=jElt
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                OPDMat,RaySpot,RMSWFE,PixArray)
           
	  !print*,'**** Perturb elt =',jElt
#if 0
          ! Update RptElt
	  IF ((rbSrf(iTmp).EQ.2) .OR. (rbSrf(iTmp).EQ.3)) THEN
	    CALL Qform(Q,dQ,tmpA)
            !RptElt(:,jElt) = RptElt(:,jElt) + MATMUL(Q,tmpD)
	    CALL DMPROD(dQ(1:3,1),Q,tmpD,3,3,1)
            RptElt(:,jElt) = RptElt(:,jElt) + dQ(:,1)
	  END IF
#endif
        END SUBROUTINE ApplyPerturbationToOpticalSystem_Prb
         
        ! --------------------------------------------------------------

	SUBROUTINE SystemCalib
        command = 'CALIB'
	CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &              OPDMat,RaySpot,RMSWFE,PixArray)	
	END SUBROUTINE SystemCalib

	! --------------------------------------------------------------
	
	SUBROUTINE MetCalc
        command = 'METcalc'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &              OPDMat,RaySpot,RMSWFE,PixArray)
        END SUBROUTINE MetCalc

	SUBROUTINE SpfCalc
        command = 'SPFcalc'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &              OPDMat,RaySpot,RMSWFE,PixArray)
        END SUBROUTINE SpfCalc

        ! --------------------------------------------------------------

	SUBROUTINE ExecFDPCmd
        command = 'FDP'
	IARG(1) = ifFDP
	CARG(1) = 'YES'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &              OPDMat,RaySpot,RMSWFE,PixArray)
        !
        If (.TRUE.) Then
          ! align pupil direction with chief ray direction
          ! updated 07/28/2009 -jzlou
          command = 'RAY'
          IARG(1) = 1
          LBWKnoPrint=.TRUE.
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                OPDMat,RaySpot,RMSWFE,PixArray)
          LBWKnoPrint=.FALSE.
          psiElt(1:3,nElt-1)=-CRIncidDir(1:3,nElt-1);
        EndIf
        END SUBROUTINE ExecFDPCmd

        ! --------------------------------------------------------------
         
        SUBROUTINE ComposePIX

        IMPLICIT NONE
        INTEGER  :: k,nimg,strt,end
	REAL*8 :: ddv(3),prop_dist,KrElt_save
	REAL*8, parameter :: zEps=1d-16 

        command = 'COMPOSE'
        !IARG(1) = nElt
        IARG(1) = ifPIXElt
	!print*,' ***** GMI.F: ifPIXElt =',ifPIXElt
        IARG(2) = nPix
        RARG(1) = pixelSize

        ! To adjust diffraction propagation distance if needed,
        ! say after the focal plane is moved. This is useful 
        ! when focal plane is moved but XP is not reset.
        ! added by jzlou, 04/2009
	KrElt_save=KrElt(IARG(1)-1)
	ddv = vptElt(1:3,IARG(1))-vptElt(1:3,IARG(1)-1)
	prop_dist = sqrt(DDOTC(ddv,ddv))
        if (abs(prop_dist-abs(zElt(IARG(1)-1)))>zEps) then
            if (KrElt(IARG(1)-1)>0d0) then
              KrElt(IARG(1)-1)=prop_dist 
            else
              KrElt(IARG(1)-1)=-prop_dist
	    end if
	end if
	!print*,'********* COMPOSE: KrElt_save=',KrElt_save
	!print*,'********* COMPOSE: KrElt(IARG(1)-1)=',KrElt(IARG(1)-1)

        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &              OPDMat,RaySpot,RMSWFE,PixArray)
	nimg=(Npimg/2)/np
        If (ip<np) Then
            strt=(ip-1)*nimg+1; end=ip*nimg
        Else
            strt=(ip-1)*nimg+1; end=Npimg/2
        End If

          !DO k=1,Npimg/2
	  DO k=strt,end
            Wavelen = pimg(2*k-1)
            Flux    = pimg(2*k)
            command = 'MODIFY'
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)

            ! ----------------------------------
            ! check if spots outside of detector
            ! ----------------------------------
            IF (ifPIXSpotDetCheck) THEN
              nGridPts = 15
              npts     = nGridPts-1

              command='SPOT'
              IF (ifPupilImg.NE.0) THEN
                IARG(1)=ifPupilImg
                CARG(1)='BEA'
              ELSE
                IARG(1)=nElt-2
                CARG(1)='BEA'
              END IF
              IF (MsgsOn)WRITE(P,*)'SPOT diagram at Element ',IARG(1)
              CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                    OPDMat,RaySpot,RMSWFE,PixArray)

              numspot   = 0
              meanspotx = 0.d0
              meanspoty = 0.d0
!NS           DO i=1,numseg
!NS             numspot(i)=0
!NS             meanspotx(i)=0.d0
!NS             meanspoty(i)=0.d0
!NS           END DO

              iSpot=0
              DO j=2,nRay
                IF (LRayOK(j).AND.LRayPass(j)) THEN
                    iSPOT=iSpot+1
!               print *,j,RayToSegMap(j,1),RayToSegMap(j,2),RaySpot(iSpot,1),
!     &             RaySpot(iSpot,2)
                meanspotx(RayToSegMap(j,1)) =
     &                  meanspotx(RayToSegMap(j,1)) + RaySpot(iSpot,1)
                meanspoty(RayToSegMap(j,1)) =
     &                  meanspoty(RayToSegMap(j,1)) + RaySpot(iSpot,2)
                numspot(RayToSegMap(j,1)) = numspot(RayToSegMap(j,1))+1
                END IF
              END DO

#if 0
	      ! numseg is hardwired for JWST
              DO i=1,numseg/2
                meanspotx(i) = meanspotx(i+1)/dble(numspot(i+1))
                meanspoty(i) = meanspoty(i+1)/dble(numspot(i+1))
                IF (MsgsOn) WRITE(P,*)i,meanspotx(i),meanspoty(i)
              END DO
#endif

              IF (ifprb) THEN
                minrbSrf=100
                DO i=1,irbSrf
                  iElt = rbSrf(i)
                  IF (iElt.LT.0) THEN
                    IF (-iElt.LT.minrbSrf) minrbSrf=-iElt
                  END IF
                END DO
                IF (MsgsOn) WRITE(P,*)'MINIMUM rbSrf =',minrbSrf
                NumChkThr        = 0
                NumOverThreshold = 0
                DO i=1,irbSrf
                  iElt = rbSrf(i)
                  IF (iElt.LT.0) THEN   ! Check threshold for masking
                  NumChkThr = NumChkThr + 1
                    IF ((abs(meanspotx(-iElt-minrbSrf+1)).GT.
     &                      transMaskThreshold).OR.
     &                  (abs(meanspoty(-iElt-minrbSrf+1)).GT.
     &                      transMaskThreshold)) THEN
                      IF (MsgsOn) 
     &                  WRITE(P,*)-iElt,' is over threshold!!!'
                      NumOverThreshold = NumOverThreshold + 1
                      nObs(-iElt)       = 1
                      ObsType(1,-iElt)  = 1
                      ObsVec(1,1,-iElt) = 8000d0
                      ObsVec(2,1,-iElt) = 0d0
                      ObsVec(3,1,-iElt) = 0d0
                    END IF
                  END IF
                END DO
              END IF

              nGridPts = cGrid-1
              npts     = nGridPts-1

              ! ----------------------------------
              ! Compose detector image
              ! ----------------------------------
!             IF (MsgsOn) WRITE(P,*)'NumChkThr =',NumChkThr
!	      IF (MsgsOn) WRITE(P,*)'NumOverThreshold = ',NumOverThreshold
	      IF (NumOverThreshold.LT.NumChkThr) THEN
                command='ADD'
                CARG(1)='NO'   ! no plotting
                IARG(2)=nPix
                RARG(1)=pixelSize
                CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                      OPDMat,RaySpot,RMSWFE,PixArray)
              ELSE
                WRITE(P,*)'Wavelength(',2*k-1,') = ',Wavelen,
     &                ' skipped due to no spots falling on the detector'
              END IF
              !
              ! take out obscurations for any segments that were out of range of detector
              !
              DO i=1,irbSrf
                iElt = rbSrf(i)
                IF (iElt.LT.0) THEN   ! Check threshold for masking
                  IF ((abs(meanspotx(-iElt-minrbSrf+1)).GT.
     &                    transMaskThreshold).OR.
     &                (abs(meanspoty(-iElt-minrbSrf+1)).GT.
     &                    transMaskThreshold)) THEN
                    nObs(-iElt)       = nObsNom(-iElt)
                    ObsType(1,-iElt)  = ObsTypeNom(1,-iElt)
                    ObsVec(1,1,-iElt) = ObsVecNom(1,1,-iElt)
                    ObsVec(2,1,-iElt) = ObsVecNom(2,1,-iElt)
                    ObsVec(3,1,-iElt) = ObsVecNom(3,1,-iElt)
                  END IF
                END IF
              END DO

            ELSE ! NO Spot check
              ! ----------------------------------
              ! Compose detector image
              ! ----------------------------------
              command='ADD'
              CARG(1)='NO'   ! no plotting
              IARG(2)=nPix
              RARG(1)=pixelSize
              CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                    OPDMat,RaySpot,RMSWFE,PixArray)

            END IF !of ifPIXSpotDetCheck check
          END DO ! loop over wavelengths

          ! ------------------------------------------------------------
          ! plot current composed image, WHY? To update PixArray ???
          ! ------------------------------------------------------------
          command='DADD'
          !CARG(1)='YES'
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                OPDMat,RaySpot,RMSWFE,PixArray)


          ! ------------------------------------------------------------
          !  Data manipulation and output generation
          ! ------------------------------------------------------------

          ! ------------------------------------
          ! Set seed for random number generator
          ! ------------------------------------
          IF (StartSeed.GT.0) THEN
            IF (MsgsOn)WRITE(P,*)'Seed set to ',StartSeed
            command='SEED'
            IARG(1)=StartSeed
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
          END IF

          ! ------------------------------------
          ! Add jitter
          ! ------------------------------------
          IF ((sigJitterX.GT.0d0).OR.(sigJitterY.GT.0d0)) THEN
            command='GBLUR'
            RARG(1)=sigJitterX
            RARG(2)=sigJitterY
            CARG(1)='NO'
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
          END IF

          ! ------------------------------------
          ! Add photon noise
          ! ------------------------------------
          IF (ifShotNoise.NE.0) THEN
            IF (MsgsOn)WRITE(P,*)'Photon noise added to image'
            command='NOISE'
            CARG(1)='PHOTON'
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
          END IF

          ! ------------------------------------
          ! Add pixel crosstalk
          ! ------------------------------------
          IF (sigCrosstalk.GT.0d0) THEN
            command='BLUR'
            RARG(1)=sigCrosstalk
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
          END IF

          ! ------------------------------------
          ! Scale to electrons and add detector bias
          ! ------------------------------------
          IF ((QE.NE.1.d0).OR.(DBias.NE.0)) THEN
            IF (MsgsOn)
     &        WRITE(P,*)'Scaling to electrons (QE =',QE,'),bias =',DBias
            DO j=1,nPix
              DO i=1,nPix
                PixArray(i,j)=PixArray(i,j)/QE+DBias
              END DO
            END DO
          END IF

          ! ------------------------------------
          ! Add read noise
          ! ------------------------------------
          IF (sigReadNoise.GT.0d0) THEN
            IF (MsgsOn)
     &        WRITE(P,*)'Adding read noise with std =',sigReadNoise
            command='NOISE'
            CARG(1)='GAUSS'
            DARG(1)=sigReadNoise
            CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,
     &                  OPDMat,RaySpot,RMSWFE,PixArray)
          END IF

          ! ------------------------------------
          ! Set PIX values
          ! ------------------------------------
          IF (ifPIXflip) THEN
            k=0
            DO j=1,nPix
              DO i=1,nPix
                k=k+1
                PIX(k)=PixArray(i,nPix-j+1)
              END DO
            END DO
          ELSE
            k=0
            DO j=1,nPix
              DO i=1,nPix
                k=k+1
                PIX(k)=PixArray(i,j)
              END DO
            END DO
          END IF

	  KrElt(IARG(1)-1)=KrElt_save

	  ! ------------------------------------
          ! Set complex E field values
          ! ------------------------------------
	  k=0; iWF=1
          DO j=1,nPix
            DO i=1,nPix
	      k=k+1
              RealEF(k)=DREAL(WFElt(i,j,iWF))
              ImagEF(k)=DIMAG(WFElt(i,j,iWF))
	    END DO
	  END DO
        END SUBROUTINE ComposePIX

        END SUBROUTINE GMI_DVR

	END MODULE GMI_mod

