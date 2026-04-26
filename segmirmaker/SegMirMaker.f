
C       SegMirMaker.f          SegMirMaker Version 0.6            2026
C       (Modernized from SMPGe.for, SMPG v0.4, dcr 5/7/92)

C	dFile generator: 3DOF/6DOF prescriptions, conic or FreeForm parent

C	Segmented Mirror Prescription Generator

C***********************************************************************
C  MODULE: parent surface state
C
C  Captures parent prescription data (conic + Mon + FF + grid) used by
C  SurfCoordFF and the per-segment writer.  For the legacy "no parent
C  file" path, parentIsFF=.FALSE. and the FF/Mon/grid sentinels are 0;
C  FreeFormSrf then reduces to the same conic solve as SMPGe used.
C***********************************************************************

	MODULE segmir_parent_mod
	  USE elt_mod, ONLY: mFFCoef, mMonCoef
	  USE param_mod, ONLY: mGridMat
	  IMPLICIT NONE
	  LOGICAL :: parentIsFF = .FALSE.
	  INTEGER :: parentElt  = 0
	  REAL*8  :: Kc_p, Kr_p, f_p, e_p
	  REAL*8  :: psi_p(3), pv_p(3)
	  REAL*8  :: lMon_p
	  REAL*8  :: pMon_p(3), xMon_p(3), yMon_p(3), zMon_p(3)
	  REAL*8  :: MonCoef_p(mMonCoef)
	  REAL*8  :: lFF_p
	  REAL*8  :: pFF_p(3), xFF_p(3), yFF_p(3), zFF_p(3)
	  REAL*8  :: FFCoef_p(mFFCoef)
	  INTEGER :: FFZernTypeL_p
	  REAL*8  :: FFZernCoef_p(mFFCoef)
	  INTEGER :: MonZernTypeL_p
	  REAL*8  :: MonZernCoef_p(mMonCoef)
	  INTEGER :: nGridMat_p
	  REAL*8  :: GridSrfdx_p
	  REAL*8  :: pData_p(3), xData_p(3), yData_p(3), zData_p(3)
	  REAL*8, ALLOCATABLE :: GridMat_p(:,:)
	  CHARACTER*24 :: GridFile_p

	CONTAINS

C***********************************************************************
C  FmtD -- format a REAL*8 with up to 16 significant digits, compact.
C  Writes 1P,ES23.15 then trims trailing zeros from the mantissa, so
C  1.2d0  -> '1.2E+00', 0.02d0 -> '2.0E-02', 1.234567890123456d0
C  keeps all 16 digits.
C***********************************************************************
	FUNCTION FmtD(x) RESULT(s)
	  IMPLICIT NONE
	  REAL*8, INTENT(IN) :: x
	  CHARACTER(LEN=24) :: s
	  CHARACTER(LEN=24) :: buf
	  INTEGER :: ie, i
	  WRITE(buf,'(1P,ES23.15E2)') x
	  buf = ADJUSTL(buf)
	  ie = INDEX(buf,'E')
	  IF (ie.EQ.0) THEN
	    s = TRIM(buf)
	    RETURN
	  END IF
	  i = ie - 1
	  DO WHILE (i.GT.1 .AND. buf(i:i).EQ.'0')
	    i = i - 1
	  END DO
	  IF (buf(i:i).EQ.'.') i = i + 1
	  s = buf(1:i) // buf(ie:LEN_TRIM(buf))
	END FUNCTION FmtD

	END MODULE segmir_parent_mod

C***********************************************************************

	PROGRAM SegMirMaker

	USE smacos_mod, ONLY: SMACOS
	USE math_mod,   ONLY: DORTHOGANALIZE
	USE src_mod,    ONLY: Aperture
	USE segmir_parent_mod

C  Declarations

	IMPLICIT NONE

C    The main sizing variables are set according to the maximum number 
C    of rings of the segmented array (mRing). In particular:
C	   mR2 = 2 x mRing
C	  mSeg = 3 x (mRing + 1) x mRing + 1
C    Other arrays are set according to the number of measurements mMeas
C	mState = 6 x mSeg
C	 mMeas = 6 x mSeg

	INTEGER mRing,mR2,mSeg,mState,mMeas
	PARAMETER (mRing=8, mR2=(2*mRing), 
     &	mSeg=(3*(mRing+1)*mRing+1), mMeas=(6*mSeg), mState=(6*mSeg))

	INTEGER nRing,nElt,SegCoord(3,mSeg),iFirst,iLast,
     &  SegCrdToNum(-mR2:mR2,-mR2:mR2),NumToRing(0:mSeg),
     &	MeasToSeg(2,mMeas),iMeasConfig,iDOF
	REAL*8 NumToPos(3,mSeg),NumToRadius(mSeg),pSeg(3,mSeg),
     &	TbSeg(3,3,mSeg)

	CHARACTER*8 Cinteger
	CHARACTER*20 GridTypeStr
	INTEGER i,j,k,ip3,jp3,iRay,iElt,iEm1,iSeg,iRing,iLeg,jSeg,nR2,
     &	iAdj,iMeas,nMeas,nState,iEltPrt,iPrt
	REAL*8 radhat(3),tanhat(3),normhat(3),SegSize,MirApDiam,width,
     &	length,w2,L2,S2,Pz(3,3),dpin(3),pinij(3),dzddeli(3),dzddelj(3),
     &	dzdthi(3),dzdthj(3),zij(3),rhoi(3),rhoj(3),rhoix(3,3),dzdxi(6),
     &	dzdxj(6),xhat(3),yhat(3),zhat(3),xseg(3)
	REAL*8 pv(3),prot(3),pin(3),ihat(3),f,e,psi(3),SegXgrid(3),
     &  pr(3),rhat(3),L,rho(3),th,PI,ds,rs,qs,ps,ths,D1(100),D2(100),
     &  D3(3,3),D4(3,3),S1,DOT,MAG,TElt(6,6),xSi,xSim1,ySi,ySim1,
     &  dx,dy,twodx,SIN60,COS60,Hw1(6),Hw2(6),
     &	SegYgrid(3),xs(3),ys(3),zs(3),standoff,gap

C  SMACOS call buffers + parent-prescription state
C  (SMACOS signature uses CHARACTER(len=256) for command/CARG; mirror it
C  here so this .f file needs no CPP.)
	CHARACTER(len=256) :: smCommand,smCARG(9)
	REAL*8  :: smDARG(9),smRARG(9),smRMSWFE
	INTEGER :: smIARG(9),parentEltIn(1),parentEltDef(1)
	LOGICAL :: smLARG
	INTEGER :: modelSize
	CHARACTER(len=256) :: parentPresc
	CHARACTER(len=256) :: smLine,fchk
	CHARACTER(len=8)   :: ans
	LOGICAL :: parentFileExists
	INTEGER :: lp

C  OPDMat / RaySpot / PixArray live in macos_mod only when CMACOS is
C  defined.  The smacos library is built without CMACOS, so we declare
C  local dummies for the SMACOS('OLD',...) call -- the OLD dispatcher
C  does not touch them.
	REAL*8, ALLOCATABLE :: OPDMat(:,:), RaySpot(:,:), PixArray(:,:)

C  Formats

 500	FORMAT(1P,'    iElt= ',i4/' EltName=  Seg',A/' EltType=  5'/
     &	'    fElt= ',d17.9/'    eElt= ',d17.9/'  psiElt= ',3d17.9/
     &	'  VptElt= ',3d17.9/'  RptElt= ',3d17.9/'  IndRef=  1d0'/
     &	'    zElt= ',d17.9/'PropType= 1'/' nECoord=',i2)
 501	FORMAT(1P,'    TElt= ',6d17.9)
 502	FORMAT(1P,10x,6d17.9)
 503	FORMAT(//'SegCoord= ',i5,2(5x,i5))
 504	FORMAT(10x,i5,2(5x,i5))
 505	FORMAT(/' Default mirror aperture diameter is ',1P,d17.9)
 506	FORMAT(//' MACOS Segmented Mirror Prescription Generator'/
     &	' Version 0.6, April 26, 2026'/)
 507	FORMAT(/'Default segment size is ',1P,d17.9)
 508	FORMAT(/' nMeas=',i6,';'/' nState=',i6,';'/)
 509	FORMAT(' Hx(',i6,',',i6,':',i6,')=[',6d17.9,'];')
 512	FORMAT(' Hx(',i6,',',i6,':',i6,')=[',3d17.9,'];')
 511	FORMAT(' Hw(',i6,',',i6,':',i6,')=[',6d17.9,'];')
 510	FORMAT(' MeasToSeg(1:2,',i6,')=[',i6,';',i6,'];')
 570	FORMAT('             nSeg=  ',i0)
 574	FORMAT('         SegCoord=',3(2x,i3))
 575	FORMAT(18x,3(2x,i3))

 901	format('   iSeg=',i4,' jSeg=',i4/1P,'     pi=',3d17.9/
     &	'     pj=',3d17.9/'     pr=',3d17.9/'   rhoi=',3d17.9/
     &	'   rhoj=',3d17.9/'dzddeli=',3d17.9/'dzddelj=',3d17.9/
     &	' dzdthi=',3d17.9/' dzdthj=',3d17.9/)

C  Constants

	DATA PI/3.141592653589793/SIN60/8.660254037844386d-01/
	DATA COS60/0.5d0/

C  Initialize arrays

	CALL ZERO(Pz,9)
	Pz(1,1)=1d0
	Pz(2,2)=1d0
	DO 8 j=-mR2,mR2
	  DO 7 i=-mR2,mR2
	    SegCrdToNum(i,j)=0
  7	  CONTINUE
  8	CONTINUE

C  Set up data file

	WRITE(*,506)

C  Initialize MACOS arrays so SMACOS / FreeFormSrf are usable
	modelSize = 512
	CALL macos_init_all(modelSize)
	ALLOCATE(OPDMat(1,1),RaySpot(1,1),PixArray(1,1))
	OPDMat=0d0; RaySpot=0d0; PixArray=0d0

C  Parent-prescription phase (first, so dialog defaults can derive from
C  it).  Blank = legacy conic-only path (parentIsFF=.FALSE., zero
C  sentinels).  Non-blank = load MACOS .in file and extract parent.

	CALL ZeroParent()
	parentPresc = ' '
	WRITE(*,'(/1X,A)',ADVANCE='NO')
     &	  'Enter parent prescription file [none]: '
	READ(*,'(A)') parentPresc
	IF (LEN_TRIM(parentPresc).GT.0 .AND.
     &	    parentPresc(1:4).NE.'none' .AND.
     &	    parentPresc(1:4).NE.'NONE') THEN
C  Strip a trailing .in / .IN if the user typed it -- SMACOS OLD
C  appends .in internally, so leaving it produces foo.in.in.
	  lp = LEN_TRIM(parentPresc)
	  IF (lp.GT.3 .AND. (parentPresc(lp-2:lp).EQ.'.in'
     &	             .OR.parentPresc(lp-2:lp).EQ.'.IN')) THEN
	    parentPresc(lp-2:lp) = '   '
	  END IF
C  Verify the .in file exists before calling SMACOS; otherwise OLD
C  fails silently and LoadParent would copy zeroed parent data.
	  fchk = TRIM(parentPresc)//'.in'
	  INQUIRE(FILE=TRIM(fchk), EXIST=parentFileExists)
	  IF (.NOT.parentFileExists) THEN
	    WRITE(*,*) ' SegMirMaker: parent file ',TRIM(fchk),
     &	      ' not found; using legacy conic dialog.'
	  ELSE
	    smCommand='OLD'
	    smCARG(1)=parentPresc
	    CALL SMACOS(smCommand,smCARG,smDARG,smIARG,smLARG,smRARG,
     &	                OPDMat,RaySpot,smRMSWFE,PixArray)
	    parentEltDef(1) = 1
	    CALL IACCEPT(parentEltIn,parentEltDef,1,
     &	      'Enter parent element number:')
	    parentElt = parentEltIn(1)
	    CALL LoadParent(parentElt)
	  END IF
	END IF

	CALL SetOutFile

C  Dialog for parameters of base mirror

	parentEltDef(1) = 3
	CALL IACCEPT(parentEltIn,parentEltDef,1,
     &	  'Enter number of segment DOFs (3 or 6):')
	iDOF = parentEltIn(1)
	IF (iDOF.NE.3) iDOF=6
	parentEltDef(1) = 1
	CALL IACCEPT(parentEltIn,parentEltDef,1,
     &	  'Enter starting segment number:')
	iEltPrt = parentEltIn(1)
	parentEltDef(1) = 1
	CALL IACCEPT(parentEltIn,parentEltDef,1,
     &	  'Enter number of rings:')
	nRing = parentEltIn(1)
	CALL CACCEPT(GridTypeStr,'Hex',
     &	  'Enter ray grid type (Circular, Square, Hex, Pie, Flower):')
	CALL ZERO(prot,3)
	IF (parentIsFF) THEN
	  CALL EQUATE(pv,pv_p,3)
	  CALL EQUATE(D1,psi_p,3)
	  f = f_p
	  e = e_p
	ELSE
	  CALL ZERO(pv,3)
	  D1(1)=0d0; D1(2)=0d0; D1(3)=1d0
	  f = 18d0
	  e = 0d0
	END IF

C  Sync _p from working defaults so they serve as DACCEPT defaults.
C  In the no-parent case this also gives _p the canonical values.
	Kc_p = -e*e
	Kr_p = -(1d0+e)*f
	f_p  = f
	e_p  = e
	CALL EQUATE(pv_p,pv,3)
	CALL EQUATE(psi_p,D1,3)

	CALL DACCEPT(psi,psi_p,3,
     &	'Enter mirror principal axis direction (x,y,z):')
	D1(1)=1d0; D1(2)=0d0; D1(3)=0d0
	CALL DACCEPT(SegXgrid,D1,3,
     &	'Enter SegXgrid (x,y,z):')
C  Build tangent-plane triad: zs=psi (preserved), xs=SegXgrid projected
C  into tangent plane, ys=zs x xs.  DORTHOGANALIZE normalizes its 1st
C  arg and overwrites its 3rd arg with X cross Y, then rebuilds Y.
C  Passing (psi, SegXgrid, ys) keeps psi's direction.
	CALL DORTHOGANALIZE(psi,SegXgrid,ys)
	xs=SegXgrid
	zs=psi

C  When psi points away from world +z (e.g., a mirror whose surface
C  faces -z, with light arriving from -z), the DORTHOGANALIZE-derived
C  ys ends up anti-aligned with world +y, which places iSeg=1 at
C  lower-right physically instead of the conventional upper-left in
C  the viewer's frame (looking from the source side toward the mirror).
C  Rotate the in-plane basis 180 deg about zs (negate both xs and ys)
C  so segment numbering starts at upper-left and traverses clockwise,
C  matching the convention used for back-facing mirrors.  Frame stays
C  right-handed.  No-op for nominal +z-facing mirrors.
	IF (zs(3) .LT. 0d0) THEN
	  xs(1) = -xs(1); xs(2) = -xs(2); xs(3) = -xs(3)
	  ys(1) = -ys(1); ys(2) = -ys(2); ys(3) = -ys(3)
	END IF

	ihat(1)=-psi(1)
	ihat(2)=-psi(2)
	ihat(3)=-psi(3)

	CALL DACCEPT(f,f_p,1,'Enter mirror focal length:')
	CALL DACCEPT(e,e_p,1,'Enter mirror eccentricity:')
	S1=0d0
	CALL DACCEPT(gap,S1,1,'Enter inter-segment gap:')
	parentEltDef(1) = 1
	S1=f/2d0
	CALL DACCEPT(standoff,S1,1,
     &	  'Enter standoff distance for segment definition plane:')

	CALL IACCEPT(parentEltIn,parentEltDef,1,
     &	  'Enter measurement configuration (1=inner, 2=all):')
	iMeasConfig = parentEltIn(1)

C  Write user's (possibly adjusted) values back to _p for SurfCoordFF.
	Kc_p = -e*e
	Kr_p = -(1d0+e)*f
	f_p  = f
	e_p  = e
	CALL EQUATE(pv_p,pv,3)
	CALL EQUATE(psi_p,psi,3)

C  Enter mirror size parameters

 21	CONTINUE
	CALL IACCEPT(i,1,1,
     &	'Enter size option (1=spec seg size, 2=spec aperture diam):')
	IF (i.EQ.1) THEN
	  IF (parentIsFF .AND. Aperture.GT.0d0) THEN
	    S1 = Aperture/DBLE(2*nRing+1)
	  ELSE
	    S1 = 0.03d0
	  END IF
	  CALL DACCEPT(SegSize,S1,1,
     &	  'Enter segment size (side-to-side):')
	  width=SegSize
	  length=SegSize/DCOS(PI/6d0)
	  w2=width/2d0
	  L2=length/2d0
	  S2=length/4d0
	  dx=w2
	  dy=L2+S2
	  twodx=width
	  IF (nRing.EQ.0) THEN
	    MirApDiam=length
	  ELSE
c	    MirApDiam=DSQRT((FLOAT(2*nRing+1)*width)**2+(S2*S2/4d0))
	    MirApDiam=FLOAT(2*nRing+1)*width
	  END IF
	ELSE IF (i.EQ.2) THEN
	  IF (parentIsFF .AND. Aperture.GT.0d0) THEN
	    S1 = Aperture
	  ELSE
	    S1 = 12d0
	  END IF
	  CALL DACCEPT(MirApDiam,S1,1,
     &	  'Enter mirror aperture diameter:')
	  SegSize=MirApDiam/FLOAT(2*nRing+1)
	  width=SegSize
	  length=SegSize/DCOS(PI/6d0)
	  w2=width/2d0
	  L2=length/2d0
	  S2=length/4d0
	  dx=w2
	  dy=L2+S2
	  twodx=width
	ELSE
	  WRITE(*,*) 'Invalid option...'
	  GO TO 21
	END IF

C  Compute segment coordinates
C  Segment blocks are emitted to a scratch unit first so the source-
C  section header (nSeg, width, gap, SegXgrid, SegCoord) can precede
C  them in the final .presc file.

	OPEN(8, STATUS='SCRATCH')

	lMon_p = L2   ! per-segment Mon aperture = half segment size

	iElt=0
	iPrt=iEltPrt-1
	jSeg=0
	SegCrdToNum(0,0)=1
	NumToRing(0)=36000
	! xseg(3)=standoff
	DO 4 iRing=0,nRing
	  iFirst=1
	  IF (iRing.EQ.0) THEN
	    iLast=1
	  ELSE
	    iLast=iRing*6
	  END IF
	  DO 3 iSeg=iFirst,iLast
	    iEm1=iElt
	    iElt=iElt+1
	    iPrt=iPrt+1
	    jSeg=jSeg+1
	    IF (iSeg.EQ.iFirst) THEN
C - First segment in ring is rightmost on top leg
	      iLeg=5
		  xseg=SIN60*xs + COS60*ys
	   !  xseg(1)=SIN60
	   !  xseg(2)=COS60
	      pin=dx*iRing*xs + dy*iRing*ys + standoff*zs
	   !  pin(1)=0d0+dx*iRing
	   !  pin(2)=0d0+dy*iRing
	   !  pin(3)=100d0
	      SegCoord(1,iElt)=iRing
	      SegCoord(2,iElt)=2*iRing
	      SegCoord(3,iElt)=iRing
	    ELSE 
	      iLeg=INT((iSeg-2)/iRing)
	      IF ((iLeg.EQ.0).OR.(iLeg.EQ.6)) THEN
C - Top leg
	        pin=pin-twodx*xs
	        xseg=ys
	        SegCoord(1,iElt)=SegCoord(1,iEm1)-2
	        SegCoord(2,iElt)=SegCoord(2,iEm1)-1
	        SegCoord(3,iElt)=SegCoord(3,iEm1)+1
	      ELSE IF (iLeg.EQ.1) THEN
C - Upper left leg
	        xseg=-SIN60*xs+COS60*ys
	        pin=pin-dx*xs-dy*ys
	        SegCoord(1,iElt)=SegCoord(1,iEm1)-1
	        SegCoord(2,iElt)=SegCoord(2,iEm1)-2
	        SegCoord(3,iElt)=SegCoord(3,iEm1)-1
	      ELSE IF (iLeg.EQ.2) THEN
C - Lower left leg
	        xseg=-SIN60*xs-COS60*ys
	        pin=pin+dx*xs-dy*ys
	        SegCoord(1,iElt)=SegCoord(1,iEm1)+1
	        SegCoord(2,iElt)=SegCoord(2,iEm1)-1
	        SegCoord(3,iElt)=SegCoord(3,iEm1)-2
	      ELSE IF (iLeg.EQ.3) THEN
C - Bottom leg
	        xseg=-ys
	        pin=pin+twodx*xs
	        SegCoord(1,iElt)=SegCoord(1,iEm1)+2
	        SegCoord(2,iElt)=SegCoord(2,iEm1)+1
	        SegCoord(3,iElt)=SegCoord(3,iEm1)-1
	      ELSE IF (iLeg.EQ.4) THEN
C - Lower right leg
	        xseg=SIN60*xs-COS60*ys
	        pin=pin+dx*xs+dy*ys
	        SegCoord(1,iElt)=SegCoord(1,iEm1)+1
	        SegCoord(2,iElt)=SegCoord(2,iEm1)+2
	        SegCoord(3,iElt)=SegCoord(3,iEm1)+1
	      ELSE IF (iLeg.EQ.5) THEN
C - Upper right leg
	        xseg=SIN60*xs+COS60*ys
	        pin=pin-dx*xs+dy*ys
	        SegCoord(1,iElt)=SegCoord(1,iEm1)-1
	        SegCoord(2,iElt)=SegCoord(2,iEm1)+1
	        SegCoord(3,iElt)=SegCoord(3,iEm1)+2
	      END IF
	    END IF

	    NumToRing(iElt)=iRing
	    SegCrdToNum(SegCoord(1,iElt),SegCoord(2,iElt))=iElt

C - Find surface normal and local coordinates 
	    CALL SurfCoordFF(rho,pr,L,radhat,tanhat,normhat,
     &	    xhat,yhat,zhat,pv,prot,pin,ihat,f,e,psi,xseg)
	    CALL EQUATE(pSeg(1,iElt),pr,3)
	    CALL ZERO(TElt,36)
	    DO 1 i=1,3
	      j=i+3
	      TElt(i,1)=xhat(i)
	      TElt(i,2)=yhat(i)
	      IF (iDOF.EQ.3) THEN
	        TElt(j,3)=zhat(i)
	      ELSE
	        TElt(i,3)=zhat(i)
	        TElt(j,4)=xhat(i)
	        TElt(j,5)=yhat(i)
	        TElt(j,6)=zhat(i)
	      END IF
	      TbSeg(i,1,iElt)=xhat(i)
	      TbSeg(i,2,iElt)=yhat(i)
	      TbSeg(i,3,iElt)=zhat(i)
  1	    CONTINUE

	    CALL IntToChar(Cinteger,jSeg,i)
	    CALL WriteSegBlock(8,iPrt,Cinteger,f,e,psi,pv,pr,iDOF,TElt,
     &	                       xhat,yhat,zhat)
  3	  CONTINUE
  4	CONTINUE

	nElt=iElt

C  Build source-section header in scratch unit 9 (matches SegDemo3.in
C  layout).  Segment blocks are already buffered in scratch unit 8.
	OPEN(9, STATUS='SCRATCH')
	WRITE(9,'(12x,"GridType=  ",A)') TRIM(GridTypeStr)
	WRITE(9,570) nElt
	WRITE(9,'(12x,"width=  ",A)') TRIM(FmtD(width))
	WRITE(9,'(14x,"gap=  ",A)') TRIM(FmtD(gap))
	WRITE(9,'(9x,"SegXgrid=",3(2x,A))')
     &	  TRIM(FmtD(SegXgrid(1))),
     &	  TRIM(FmtD(SegXgrid(2))),
     &	  TRIM(FmtD(SegXgrid(3)))
	WRITE(9,574) (SegCoord(i,1),i=1,3)
	DO 5 iElt=2,nElt
	  WRITE(9,575) (SegCoord(i,iElt),i=1,3)
  5	CONTINUE
	WRITE(9,'()')

C  Preview: show header + first segment block only (later segments are
C  structurally identical, just with different pose).
	WRITE(*,'(/A/)') '---- Prescription preview ----'
	CALL DumpScratch(9, -1, 0)
	CALL DumpScratch(8, -1, 1)
	IF (nElt .GT. 1) THEN
	  WRITE(*,'(A,I0,A)') '  ... (',nElt-1,
     &	      ' more segment blocks omitted from preview) ...'
	END IF
	WRITE(*,'(/A/)') '---- End of preview ----'

	WRITE(*,'(1X,A)',ADVANCE='NO')
     &	  'Write prescription to file? [Y/n]: '
	READ(*,'(A)') ans
	IF (ans(1:1).EQ.'n' .OR. ans(1:1).EQ.'N') THEN
	  WRITE(*,*) ' Aborted by user; no files written.'
	  CLOSE(2, STATUS='DELETE')
	  CLOSE(3, STATUS='DELETE')
	  CLOSE(8)
	  CLOSE(9)
	  STOP
	END IF

	CALL DumpScratch(9, 2, 0)
	CALL DumpScratch(8, 2, 0)
	CLOSE(8)
	CLOSE(9)

C  Compute edge-sensor measurement matrices

C    First measurement is of master segment absolute piston

	iMeas=1
	CALL ZERO(dzdxi,6)
	dzdxi(6)=1d0
	IF (iDOF.EQ.3) THEN
	  WRITE(3,512)iMeas,1,iDOF,dzdxi(1),dzdxi(2),dzdxi(6)
	ELSE
	  WRITE(3,509)iMeas,1,iDOF,dzdxi
	END IF
	MeasToSeg(1,iMeas)=1
	MeasToSeg(2,iMeas)=1

C    Loop through each segment
	
	DO 20 iElt=2,nElt
	  iSeg=iElt
	  iRing=NumToRing(iElt)
	  DO 19 iAdj=1,6

C    Find each adjacent segment

	    IF (iAdj.EQ.1) THEN
	      i=SegCoord(1,iElt)+1
	      j=SegCoord(2,iElt)+2
	    ELSE IF (iAdj.EQ.2) THEN
	      i=SegCoord(1,iElt)-1
	      j=SegCoord(2,iElt)+1
	    ELSE IF (iAdj.EQ.3) THEN
	      i=SegCoord(1,iElt)-2
	      j=SegCoord(2,iElt)-1
	    ELSE IF (iAdj.EQ.4) THEN
	      i=SegCoord(1,iElt)-1
	      j=SegCoord(2,iElt)-2
	    ELSE IF (iAdj.EQ.5) THEN
	      i=SegCoord(1,iElt)+1
	      j=SegCoord(2,iElt)-1
	    ELSE IF (iAdj.EQ.6) THEN
	      i=SegCoord(1,iElt)+2
	      j=SegCoord(2,iElt)+1
	    END IF
	    IF ((ABS(i).LE.MR2).AND.(ABS(j).LE.MR2)) THEN
	      jSeg=SegCrdToNum(i,j)
	    ELSE
	      jSeg=0
	    END IF


C   If adjacent segment exists and is eligible, compute Hx submatrix

	    IF ((jSeg.GT.0).AND.
     &	    (((iMeasConfig.EQ.1).AND.(NumToRing(jSeg).LE.iRing))
     &	    .OR.(iMeasConfig.EQ.2)))THEN
C  Midpoint of two adjacent segment centers, projected onto the
C  parent's tangent plane through pv (normal zs), then lifted by
C  standoff along zs.  This replaces the legacy world-xy projection
C  plus fixed pin(3)=100, which assumed psi=(0,0,1).
	      dpin(1:3)=5d-1*(pSeg(1:3,iSeg)+pSeg(1:3,jSeg))
	      D1(1:3)=dpin(1:3)-pv(1:3)
	      pin(1:3)=pv(1:3)+D1(1:3)
     &	               -DOT_PRODUCT(D1(1:3),zs(1:3))*zs(1:3)
     &	               +standoff*zs(1:3)

	      CALL SurfCoordFF(rho,pr,L,radhat,tanhat,normhat,
     &	      xhat,yhat,zhat,pv,prot,pin,ihat,f,e,psi,xseg)

	      CALL SUB(rhoi,pr,pSeg(1,iSeg),3)
	      CALL SUB(rhoj,pr,pSeg(1,jSeg),3)
	      CALL MPROD(dzddeli,normhat,TbSeg(1,1,iSeg),1,3,3)

	      CALL MPROD(D1,normhat,TbSeg(1,1,jSeg),1,3,3)
	      CALL NEGATE(dzddelj,D1,3)

	      CALL CROSSMAT(rhoix,rhoi)
	      CALL MPROD(D1,normhat,TbSeg(1,1,iSeg),1,3,3)
	      CALL MPROD(D2,D1,rhoix,1,3,3)
	      CALL NEGATE(dzdthi,D2,3)

	      CALL MPROD(D1,normhat,TbSeg(1,1,jSeg),1,3,3)
	      CALL MPROD(dzdthj,D1,rhoix,1,3,3)

	      CALL EQUATE(dzdxi,dzdthi,3)
	      CALL EQUATE(dzdxi(4),dzddeli,3)
	      CALL EQUATE(dzdxj,dzdthj,3)
	      CALL EQUATE(dzdxj(4),dzddelj,3)

	      iMeas=iMeas+1
	      MeasToSeg(1,iMeas)=iSeg
	      MeasToSeg(2,iMeas)=jSeg

	      i=(iSeg-1)*iDOF+1
	      k=iSeg*iDOF
c	      WRITE(3,509)iMeas,i,k,dzdxi
	      IF (iDOF.EQ.3) THEN
	        WRITE(3,512)iMeas,i,k,dzdxi(1),dzdxi(2),dzdxi(6)
	      ELSE
	        WRITE(3,509)iMeas,i,k,dzdxi
	      END IF
	      j=(jSeg-1)*iDOF+1
	      k=jSeg*iDOF
c	      WRITE(3,509)iMeas,j,k,dzdxj
	      IF (iDOF.EQ.3) THEN
	        WRITE(3,512)iMeas,j,k,dzdxj(1),dzdxj(2),dzdxj(6)
	      ELSE
	        WRITE(3,509)iMeas,j,k,dzdxj
	      END IF

	    END IF
 19	  CONTINUE
 20	CONTINUE

	nMeas=iMeas
	nState=iDOF*nElt
	WRITE(3,508)nMeas,nState
	DO 10 iMeas=1,nMeas
	  WRITE(3,510)iMeas,(MeasToSeg(i,iMeas),i=1,2)
 10	CONTINUE


C  End of program

	CLOSE (2)
	CLOSE (3)
	WRITE(*,*) "Done."
	STOP
	END PROGRAM SegMirMaker

C***********************************************************************

	SUBROUTINE SurfCoord(rho,pr,L,radhat,tanhat,normhat,
     &	xhat,yhat,zhat,pv,prot,pin,ihat,f,e,psi,xseg)
C       Retained for reference; SegMirMaker calls SurfCoordFF instead.
	IMPLICIT NONE
	REAL*8 pv(3),prot(3),pin(3),ihat(3),f,e,psi(3),
     &  pr(3),L,radhat(3),tanhat(3),normhat(3),
     &	xhat(3),yhat(3),zhat(3),xseg(3),rho(3)
	WRITE(*,*) ' SurfCoord: stub (use SurfCoordFF)'
	STOP
	END

C***********************************************************************

	SUBROUTINE SurfCoordFF(rho,pr,L,radhat,tanhat,normhat,
     &	xhat,yhat,zhat,pv,prot,pin,ihat,f,e,psi,xseg)

C	Wraps MACOS FreeFormSrf.  For the legacy conic-only case the
C	parent_mod sentinels (lMon=0, LFF=0, nGridMat=0) make FreeFormSrf
C	reduce to the same conic solve SMPGe used.
C
C	Inputs: pv (vertex), prot (rot-pt, unused), pin (ray origin),
C	        ihat (ray dir), f/e (conic), psi (axis), xseg (hex leg).
C	Outputs: rho (pr-pv), pr (intersection), L (ray length),
C	         radhat/tanhat/normhat (rho-aligned frame),
C	         xhat/yhat/zhat (segment face frame, xseg-aligned).

	USE param_mod,        ONLY: mGridMat
	USE surfsub,          ONLY: FreeFormSrf, LZPFailed
	USE segmir_parent_mod

	IMPLICIT NONE

	REAL*8 pv(3),prot(3),pin(3),ihat(3),f,e,psi(3),
     &  pr(3),L,Nhat(3),radhat(3),tanhat(3),normhat(3),
     &	xhat(3),yhat(3),zhat(3),xseg(3),rho(3)

	REAL*8 Kc,Kr,Nvec(3),Nmag,dNdp(3,3),S1
	REAL*8 GridMatArg(mGridMat,mGridMat)
	LOGICAL LROK
	INTEGER i,j

C  Compose parent surface for FreeFormSrf
	Kc=-e*e
	Kr=-(1d0+e)*f

C  Grid matrix: pass parent's when available, else a zero placeholder
	IF (parentIsFF .AND. nGridMat_p.GT.0) THEN
	  DO j=1,mGridMat
	    DO i=1,mGridMat
	      GridMatArg(i,j)=0d0
	    END DO
	  END DO
	  DO j=1,nGridMat_p
	    DO i=1,nGridMat_p
	      GridMatArg(i,j)=GridMat_p(i,j)
	    END DO
	  END DO
	ELSE
	  DO j=1,mGridMat
	    DO i=1,mGridMat
	      GridMatArg(i,j)=0d0
	    END DO
	  END DO
	END IF

	LZPFailed=.FALSE.
	LROK=.TRUE.
	CALL FreeFormSrf(.FALSE.,.FALSE.,LROK,
     &	  Kc,Kr,MonCoef_p,
     &	  psi,pv,prot,pin,ihat,pr,L,lMon_p,
     &	  nGridMat_p,mGridMat,GridMatArg,GridSrfdx_p,
     &	  pMon_p,xMon_p,yMon_p,zMon_p,
     &	  pData_p,xData_p,yData_p,zData_p,
     &	  lFF_p,pFF_p,xFF_p,yFF_p,zFF_p,FFCoef_p,
     &	  Nvec,Nmag,Nhat,dNdp)
	IF ((.NOT.LROK) .OR. LZPFailed) THEN
	  WRITE(*,*) ' SurfCoordFF: ray missed parent surface.'
	  WRITE(*,*) '   pin = ', pin
	  WRITE(*,*) '   ihat= ', ihat
	  WRITE(*,*) '   pv  = ', pv
	  WRITE(*,*) '   psi = ', psi
	  WRITE(*,*) '   Kc,Kr = ', Kc, Kr
	  WRITE(*,*) '   lMon_p, lFF_p, nGridMat_p = ',
     &	             lMon_p, lFF_p, nGridMat_p
	  WRITE(*,*) '   LROK=',LROK,' LZPFailed=',LZPFailed
	  STOP
	END IF

	rho(1)=pr(1)-pv(1)
	rho(2)=pr(2)-pv(2)
	rho(3)=pr(3)-pv(3)

C  Rho-aligned frame
	S1=rho(1)*rho(1)+rho(2)*rho(2)+rho(3)*rho(3)
	IF (S1.LE.1d-16) THEN
	  tanhat(1)=0d0
	  tanhat(2)=1d0
	  tanhat(3)=0d0
	ELSE
	  CALL CROSSPROD(tanhat,Nhat,rho)
	END IF
	CALL UNITIZE(tanhat)
	CALL EQUATE(normhat,Nhat,3)
	CALL CROSSPROD(radhat,tanhat,Nhat)
	CALL UNITIZE(radhat)

C  Segment face frame (xseg = hex leg direction in base coords)
	S1=xseg(1)*xseg(1)+xseg(2)*xseg(2)+xseg(3)*xseg(3)
	IF (S1.LE.1d-16) THEN
	  yhat(1)=0d0
	  yhat(2)=1d0
	  yhat(3)=0d0
	ELSE
	  CALL CROSSPROD(yhat,Nhat,xseg)
	END IF
	CALL UNITIZE(yhat)
	CALL EQUATE(zhat,Nhat,3)
	CALL CROSSPROD(xhat,yhat,Nhat)
	CALL UNITIZE(xhat)

	RETURN
	END

C	End of SurfCoordFF
C***********************************************************************
C***********************************************************************
	SUBROUTINE SetOutFile

	IMPLICIT NONE
	CHARACTER*32 ans,filnam,outfil,Hxfil
	LOGICAL exist
	INTEGER l

C	Get file name

	DO
	  filnam = ' '
	  WRITE(*,'(1X,A)',ADVANCE='NO') 'Enter output file name: '
	  READ(*,'(A)') filnam
	  IF (filnam.EQ.' ') CYCLE
	  l = LEN(TRIM(filnam))
	  outfil=filnam
	  Hxfil=filnam
	  outfil(l+1:l+6) = '.presc'
	  Hxfil(l+1:l+4)  = 'Hx.m'
	  INQUIRE (FILE=TRIM(outfil),EXIST=exist)
	  IF (exist) THEN
	    WRITE(*,'(1X,A)',ADVANCE='NO')
     &        'Output file '//TRIM(outfil)//
     &        ' exists. Overwrite? [y/N]: '
	    READ(*,'(A)') ans
	    IF ((ans(1:1).EQ.'y').OR.(ans(1:1).EQ.'Y')) THEN
	      OPEN (2,FILE=TRIM(outfil),STATUS='REPLACE')
	      OPEN (3,FILE=TRIM(Hxfil), STATUS='REPLACE')
	      RETURN
	    ELSE
	      CYCLE
	    END IF
	  END IF
	  OPEN (2,FILE=TRIM(outfil),STATUS='NEW')
	  OPEN (3,FILE=TRIM(Hxfil), STATUS='NEW')
	  RETURN
	END DO

	END

C***********************************************************************
C  DumpScratch(iSrc, iDst)
C  Copy a scratch unit's contents line-by-line to iDst.  If iDst <= 0,
C  write to stdout instead.  Used to preview .presc contents on screen
C  and to emit them to the final file after user confirmation.
C***********************************************************************
      SUBROUTINE DumpScratch(iSrc, iDst, maxBlocks)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: iSrc, iDst, maxBlocks
      CHARACTER(LEN=256) :: line
      INTEGER :: ios, blockCount
      REWIND(iSrc)
      blockCount = 0
      DO
        READ(iSrc, '(A)', IOSTAT=ios) line
        IF (ios /= 0) EXIT
        IF (INDEX(ADJUSTL(line),'iElt=') .EQ. 1) THEN
          blockCount = blockCount + 1
        END IF
        IF (maxBlocks .GT. 0 .AND. blockCount .GT. maxBlocks) EXIT
        IF (iDst .LE. 0) THEN
          WRITE(*,  '(A)') TRIM(line)
        ELSE
          WRITE(iDst,'(A)') TRIM(line)
        END IF
      END DO
      END SUBROUTINE

C***********************************************************************
C
      SUBROUTINE ADD(A,B,C,N)
      REAL*8 A(N),B(N),C(N)
      DO 1 I=1,N
    1 A(I)=B(I)+C(I)
      RETURN
      END
C
      SUBROUTINE SUB(A,B,C,N)
      REAL*8 A(N),B(N),C(N)
      DO 1 I=1,N
    1 A(I)=B(I)-C(I)
      RETURN
      END
C
      SUBROUTINE SMPROD(A,B,S,N)
      REAL*8 A(1),B(1),S
      DO 1 I=1,N
    1 A(I)=B(I)*S
      RETURN
      END
C
      SUBROUTINE NEGATE(A,B,N)
      REAL*8 A(1),B(1)
      DO 1 J=1,N
    1 A(J)=-B(J)
      RETURN
      END
C
      SUBROUTINE EQUATE(A,B,N)
      REAL*8 A(1),B(1)
      DO 1 J=1,N
    1 A(J)=B(J)
      RETURN
      END
C
      SUBROUTINE EQTsp(A,B,N)
      REAL*4 A(1),B(1)
      DO 1 J=1,N
    1 A(J)=B(J)
      RETURN
      END
C
C
      SUBROUTINE UNITVEC(YHAT,Y,MAG,N)
      REAL*8 YHAT(1),Y(1),MAG
      INTEGER I,N
      MAG=0D0
      DO I=1,N
	MAG=MAG+Y(I)**2
      END DO
      MAG=DSQRT(MAG)
      DO I=1,N
	YHAT(I)=Y(I)/MAG
      END DO
      RETURN
      END
C
      FUNCTION MAG(X,N)
      INTEGER I,N
      REAL*8 X(N),MAG
      MAG=0D0
      DO I=1,N
        MAG=MAG+X(I)*X(I)
      END DO
      MAG=DSQRT(MAG)
      RETURN
      END
C
      FUNCTION DOT(X,Y)
      REAL*8 X(3),Y(3),DOT
      DOT=X(1)*Y(1)+X(2)*Y(2)+X(3)*Y(3)
      RETURN
      END
C
      SUBROUTINE UNITSP(X)
      REAL*4 X(3),Y
      Y=SQRT(X(1)*X(1)+X(2)*X(2)+X(3)*X(3))
      IF (Y.GT.1D-10) THEN
        X(1)=X(1)/Y
        X(2)=X(2)/Y
        X(3)=X(3)/Y
      ELSE
        X(1)=0D0
        X(2)=0D0
        X(3)=0D0
      END IF
      RETURN
      END
C
C
      SUBROUTINE UNITIZE(X)
      REAL*8 X(3),Y
      Y=SQRT(X(1)*X(1)+X(2)*X(2)+X(3)*X(3))
      IF (Y.GT.1D-10) THEN
        X(1)=X(1)/Y
        X(2)=X(2)/Y
        X(3)=X(3)/Y
      ELSE
        X(1)=0D0
        X(2)=0D0
        X(3)=0D0
      END IF
      RETURN
      END
C
      FUNCTION RSS(X)
      REAL*8 X(3),RSS
      RSS=DSQRT(X(1)*X(1)+X(2)*X(2)+X(3)*X(3))
      RETURN
      END
C
C
      SUBROUTINE MPROD(A,B,C,NA,NB,NC)
      REAL*8 A(1),B(1),C(1)
      NAMAX=NA*NC
      NBMAX=NA*NB
      DO 1 I=1,NA
      J=-NB
      DO 1 NAPTR=I,NAMAX,NA
      J=J+NB
      K=0
      A(NAPTR)=0D0
      DO 1 NBPTR=I,NBMAX,NA
      K=K+1
      NCPTR=J+K
    1 A(NAPTR)=A(NAPTR)+B(NBPTR)*C(NCPTR)
      RETURN
      END
C
C
      SUBROUTINE MPRODB(A,B,C,NA,NB,NC)
      REAL*8 A(1),B(1),C(1),BIK
      NAMAX=NA*NC
      NBMAX=NA*NB
      DO 1 NAPTR=1,NAMAX
    1 A(NAPTR)=0D0
      DO 3 I=1,NA
      K=0
      DO 3 NBPTR=I,NBMAX,NA
      K=K+1
      BIK=B(NBPTR)
      IF (BIK.EQ.0D0) GO TO 3
      NCPTR=K
      DO 2 NAPTR=I,NAMAX,NA
      A(NAPTR)=A(NAPTR)+BIK*C(NCPTR)
    2 NCPTR=NCPTR+NB
    3 CONTINUE
      RETURN
      END
C
C
      SUBROUTINE MPBsp(A,B,C,NA,NB,NC)
      REAL*4 A(1),B(1),C(1),BIK
      NAMAX=NA*NC
      NBMAX=NA*NB
      DO 1 NAPTR=1,NAMAX
    1 A(NAPTR)=0D0
      DO 3 I=1,NA
      K=0
      DO 3 NBPTR=I,NBMAX,NA
      K=K+1
      BIK=B(NBPTR)
      IF (BIK.EQ.0D0) GO TO 3
      NCPTR=K
      DO 2 NAPTR=I,NAMAX,NA
      A(NAPTR)=A(NAPTR)+BIK*C(NCPTR)
    2 NCPTR=NCPTR+NB
    3 CONTINUE
      RETURN
      END
C
C
      SUBROUTINE MPRODC(A,B,C,NA,NB,NC)
      REAL*8 A(1),B(1),C(1),CJK
      NAM1=NA-1
      NAMAX=NA*NC
      NCMAX=NB*NC
      DO 1 NAPTR=1,NAMAX
    1 A(NAPTR)=0D0
      K=-NA
      DO 3 J=1,NB
      K=K+NA
      I=-NAM1
      DO 3 NCPTR=J,NCMAX,NB
      I=I+NA
      CJK=C(NCPTR)
      IF (CJK.EQ.0D0) GO TO 3
      II=I+NAM1
      NBPTR=K
      DO 2 NAPTR=I,II
      NBPTR=NBPTR+1
    2 A(NAPTR)=A(NAPTR)+B(NBPTR)*CJK
    3 CONTINUE
      RETURN
      END
C
C
      SUBROUTINE MPCsp(A,B,C,NA,NB,NC)
      REAL*4 A(1),B(1),C(1),CJK
      NAM1=NA-1
      NAMAX=NA*NC
      NCMAX=NB*NC
      DO 1 NAPTR=1,NAMAX
    1 A(NAPTR)=0D0
      K=-NA
      DO 3 J=1,NB
      K=K+NA
      I=-NAM1
      DO 3 NCPTR=J,NCMAX,NB
      I=I+NA
      CJK=C(NCPTR)
      IF (CJK.EQ.0D0) GO TO 3
      II=I+NAM1
      NBPTR=K
      DO 2 NAPTR=I,II
      NBPTR=NBPTR+1
    2 A(NAPTR)=A(NAPTR)+B(NBPTR)*CJK
    3 CONTINUE
      RETURN
      END
C
C
      SUBROUTINE MPACsp(A,B,C,NA,NB,NC)
      REAL*4 A(1),B(1),C(1),CJK
      NAM1=NA-1
      NAMAX=NA*NC
      NCMAX=NB*NC
c     DO 1 NAPTR=1,NAMAX
c   1 A(NAPTR)=0D0
      K=-NA
      DO 3 J=1,NB
      K=K+NA
      I=-NAM1
      DO 3 NCPTR=J,NCMAX,NB
      I=I+NA
      CJK=C(NCPTR)
      IF (CJK.EQ.0D0) GO TO 3
      II=I+NAM1
      NBPTR=K
      DO 2 NAPTR=I,II
      NBPTR=NBPTR+1
    2 A(NAPTR)=A(NAPTR)+B(NBPTR)*CJK
    3 CONTINUE
      RETURN
      END
C
C
      SUBROUTINE TRPOS(A,B,NA,NB)
      REAL*8 A(NA,NB),B(NB,NA)
      DO 1 I=1,NA
      DO 1 J=1,NB
    1 A(I,J)=B(J,I)
      RETURN
      END
C
      SUBROUTINE ZERO(A,N)
      REAL*8 A(1)
      DO 1 I=1,N
    1 A(I)=0D0
      RETURN
      END
C
       SUBROUTINE MINV2(A1,A)
       REAL*8 A1(2,2),A(2,2),DET
       DET=A(1,1)*A(2,2)-A(2,1)*A(1,2)
       A1(1,1)=A(2,2)/DET
       A1(1,2)=-A(1,2)/DET
       A1(2,1)=-A(2,1)/DET
       A1(2,2)=A(1,1)/DET
       RETURN
       END
C
      SUBROUTINE MINV(AI,A,N)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(N,N),AI(N,N),AX(5,10)
      DO 2 I=1,N
      DO 1 J=1,N
      AX(I,J)=A(I,J)
    1 AX(I,N+J)=0D0
    2 AX(I,N+I)=1D0
      NP=2*N
      DO 7 I=1,N
      ALFA=AX(I,I)
      DO 3 J=1,NP
    3 AX(I,J)=AX(I,J)/ALFA
      DO 6 K=1,N
      IF(K-I)4,6,4
    4 BETA=AX(K,I)
      DO 5 J=1,NP
    5 AX(K,J)=AX(K,J)-BETA*AX(I,J)
    6 CONTINUE
    7 CONTINUE
      DO 8 I=1,N
      DO 8 J=1,N
    8 AI(I,J)=AX(I,J+N)
      RETURN
      END
C
C
	SUBROUTINE OUTER(X,Y,Z)
	INTEGER I,J
	REAL*8 X(3,3),Y(3),Z(3)
	DO I=1,3
	  DO J=1,3
	    X(I,J)=Y(I)*Z(J)
	  END DO
	END DO
	RETURN
	END
C
	SUBROUTINE REFLECT(R,N)
	REAL*8 R(3,3),N(3),MAG
        MAG=SQRT(N(1)*N(1)+N(2)*N(2)+N(3)*N(3))
        N(1)=N(1)/MAG
        N(2)=N(2)/MAG
        N(3)=N(3)/MAG
	R(1,1)=-2D0*N(1)*N(1)+1D0
	R(1,2)=-2D0*N(1)*N(2)
	R(1,3)=-2D0*N(1)*N(3)
	R(2,1)=R(1,2)
	R(2,2)=-2D0*N(2)*N(2)+1D0
	R(2,3)=-2D0*N(2)*N(3)
	R(3,1)=R(1,3)
	R(3,2)=R(2,3)
	R(3,3)=-2D0*N(3)*N(3)+1D0
	RETURN
	END
C
C
      SUBROUTINE DTSP(A,B,NAROW,NACOL,NACAT)
      REAL*4 A(NAROW,NACOL,NACAT)
      REAL*8 B(NAROW,NACOL,NACAT)
      DO 1 K=1,NACAT
      DO 1 J=1,NACOL
      DO 1 I=1,NAROW
    1 A(I,J,K)=B(I,J,K)
      RETURN
      END
C
C
      SUBROUTINE COMPACT(A,B,NAROW,NACOL,NBROW,NBCOL)
      REAL*4 A(NAROW,NACOL)
      REAL*8 B(NBROW,NBCOL)
      DO 1 J=1,NACOL
      DO 1 I=1,NAROW
    1 A(I,J)=B(I,J)
      RETURN
      END
C
	SUBROUTINE PROJECT(P,N)
	REAL*8 P(3,3),N(3),MAG
        MAG=SQRT(N(1)*N(1)+N(2)*N(2)+N(3)*N(3))
        N(1)=N(1)/MAG
        N(2)=N(2)/MAG
        N(3)=N(3)/MAG
	P(1,1)=-N(1)*N(1)+1D0
	P(1,2)=-N(1)*N(2)
	P(1,3)=-N(1)*N(3)
	P(2,1)=P(1,2)
	P(2,2)=-N(2)*N(2)+1D0
	P(2,3)=-N(2)*N(3)
	P(3,1)=P(1,3)
	P(3,2)=P(2,3)
	P(3,3)=-N(3)*N(3)+1D0
	RETURN
	END
C
	SUBROUTINE CROSSMAT(X,Y)
	REAL*8 X(3,3),Y(3)
	X(1,1)=0D0
	X(1,2)=-Y(3)
	X(1,3)= Y(2)
	X(2,1)= Y(3)
	X(2,2)=0D0
	X(2,3)=-Y(1)
	X(3,1)=-Y(2)
	X(3,2)= Y(1)
	X(3,3)=0D0
	RETURN
	END
C
	SUBROUTINE CROSSPROD(X,Y,Z)
	REAL*8 X(3),Y(3),Z(3)
	X(1)=-Y(3)*Z(2)+Y(2)*Z(3)
	X(2)= Y(3)*Z(1)-Y(1)*Z(3)
	X(3)=-Y(2)*Z(1)+Y(1)*Z(2)
	RETURN
	END

C***********************************************************************

	SUBROUTINE DACCEPT(DVAR,DDEF,N,TEXT)
C	DOUBLE-PRECISION N-VECTOR INPUT ROUTINE
C	Prints TEXT + default DDEF, reads one line.
C	Blank line -> DVAR = DDEF; else parse list-directed.
C	Assumed-size arrays let callers pass a scalar when N=1.

	IMPLICIT NONE
	CHARACTER*(*) TEXT
	INTEGER N
	REAL*8 DVAR(*),DDEF(*)

	CHARACTER*256 BUF
	CHARACTER*32  FMT
	INTEGER I,ios

100	CONTINUE
	WRITE(FMT,'(A,I0,A)') "(1X,A,' [',1P,",N,"G15.7,']: ')"
	WRITE(*,FMT,ADVANCE='NO') TEXT,(DDEF(I),I=1,N)
	READ(*,'(A)') BUF
	IF (LEN_TRIM(BUF).EQ.0) THEN
	  DO I=1,N
	    DVAR(I)=DDEF(I)
	  END DO
	ELSE
	  READ(BUF,*,IOSTAT=ios) (DVAR(I),I=1,N)
	  IF (ios.NE.0) THEN
	    WRITE(*,*) '  (parse error, try again)'
	    GO TO 100
	  END IF
	END IF
	RETURN
	END

C***********************************************************************

	SUBROUTINE RACCEPT(RVAR,RDEF,N,TEXT)
C	REAL*4 N-VECTOR INPUT ROUTINE
C	Blank line -> RVAR = RDEF; else parse list-directed.

	IMPLICIT NONE
	CHARACTER*(*) TEXT
	INTEGER N
	REAL*4 RVAR(*),RDEF(*)

	CHARACTER*256 BUF
	CHARACTER*32  FMT
	INTEGER I,ios

100	CONTINUE
	WRITE(FMT,'(A,I0,A)') "(1X,A,' [',1P,",N,"G15.7,']: ')"
	WRITE(*,FMT,ADVANCE='NO') TEXT,(RDEF(I),I=1,N)
	READ(*,'(A)') BUF
	IF (LEN_TRIM(BUF).EQ.0) THEN
	  DO I=1,N
	    RVAR(I)=RDEF(I)
	  END DO
	ELSE
	  READ(BUF,*,IOSTAT=ios) (RVAR(I),I=1,N)
	  IF (ios.NE.0) THEN
	    WRITE(*,*) '  (parse error, try again)'
	    GO TO 100
	  END IF
	END IF
	RETURN
	END

C***********************************************************************

	SUBROUTINE IACCEPT(IVAR,IDEF,N,TEXT)
C	INTEGER N-VECTOR INPUT ROUTINE
C	Blank line -> IVAR = IDEF; else parse list-directed.

	IMPLICIT NONE
	CHARACTER*(*) TEXT
	INTEGER N
	INTEGER IVAR(*),IDEF(*)

	CHARACTER*256 BUF
	CHARACTER*32  FMT
	INTEGER I,ios

100	CONTINUE
	WRITE(FMT,'(A,I0,A)') "(1X,A,' [',",N,"(I0,1X),']: ')"
	WRITE(*,FMT,ADVANCE='NO') TEXT,(IDEF(I),I=1,N)
	READ(*,'(A)') BUF
	IF (LEN_TRIM(BUF).EQ.0) THEN
	  DO I=1,N
	    IVAR(I)=IDEF(I)
	  END DO
	ELSE
	  READ(BUF,*,IOSTAT=ios) (IVAR(I),I=1,N)
	  IF (ios.NE.0) THEN
	    WRITE(*,*) '  (parse error, try again)'
	    GO TO 100
	  END IF
	END IF
	RETURN
	END

C***********************************************************************

	SUBROUTINE CACCEPT(CVAR,CDEF,TEXT)
C	CHARACTER INPUT ROUTINE
C	Blank line -> CVAR = CDEF (if given); else reprompt.

	IMPLICIT NONE
	CHARACTER*(*) TEXT,CDEF
	CHARACTER*32 CVAR

  1	FORMAT(1X,A,' [',A,']: ')
  5	FORMAT(1X,A,': ')

100	CONTINUE
	CVAR=' '
	IF (CDEF.EQ.' ') THEN
	  WRITE(*,5,ADVANCE='NO') TEXT
	  READ(*,'(A)') CVAR
	  IF (CVAR.EQ.' ') GO TO 100
	ELSE
	  WRITE(*,1,ADVANCE='NO') TEXT,TRIM(CDEF)
	  READ(*,'(A)') CVAR
	  IF (CVAR.EQ.' ') CVAR=CDEF
	END IF
	RETURN
	END

C***********************************************************************

	SUBROUTINE PROMPT(CVAR,TEXT)
	IMPLICIT NONE
	CHARACTER*(*) TEXT
	CHARACTER*32 CVAR
  5	FORMAT(1X,A,': ')
100	CONTINUE
	CVAR=' '
	WRITE(*,5,ADVANCE='NO') TEXT
	READ(*,'(A)') CVAR
	IF (CVAR.EQ.' ') GO TO 100
	RETURN
	END

C***********************************************************************
C  IntToChar is provided by macos_f90/utilsub.F via libsmacos.a; do not
C  redefine it locally (causes duplicate-symbol link error).
C***********************************************************************
C  ZeroParent
C
C  Reset parent_mod state to the legacy conic-only defaults so
C  FreeFormSrf reduces to the same conic solve SMPGe used: lMon=0,
C  lFF=0, nGridMat=0, and all coord-frame / coefficient blocks zero.
C***********************************************************************

	SUBROUTINE ZeroParent()
	USE segmir_parent_mod
	IMPLICIT NONE
	INTEGER i

	parentIsFF    = .FALSE.
	parentElt     = 0

	Kc_p  = 0d0
	Kr_p  = 0d0
	f_p   = 0d0
	e_p   = 0d0
	lMon_p = 0d0
	lFF_p  = 0d0
	nGridMat_p  = 0
	GridSrfdx_p = 0d0
	FFZernTypeL_p  = 0
	MonZernTypeL_p = 0
	GridFile_p = ' '

	DO i=1,3
	  psi_p(i)=0d0;  pv_p(i)=0d0
	  pMon_p(i)=0d0; xMon_p(i)=0d0; yMon_p(i)=0d0; zMon_p(i)=0d0
	  pFF_p(i)=0d0;  xFF_p(i)=0d0;  yFF_p(i)=0d0;  zFF_p(i)=0d0
	  pData_p(i)=0d0;xData_p(i)=0d0;yData_p(i)=0d0;zData_p(i)=0d0
	END DO

	DO i=1,mMonCoef
	  MonCoef_p(i)=0d0
	  MonZernCoef_p(i)=0d0
	END DO
	DO i=1,mFFCoef
	  FFCoef_p(i)=0d0
	  FFZernCoef_p(i)=0d0
	END DO

	IF (ALLOCATED(GridMat_p)) DEALLOCATE(GridMat_p)

	RETURN
	END

C***********************************************************************
C  LoadParent(iParent)
C
C  Copy parent element's surface data from elt_mod into parent_mod.
C  Per design choice 2, segment Mon slot stays empty, so the parent
C  must have no Mon component; abort with a clear message if it does.
C***********************************************************************

	SUBROUTINE LoadParent(iParent)
	USE elt_mod, ONLY: KcElt, KrElt, psiElt, VptElt, fElt, eElt,
     &	  lMon, pMon, xMon, yMon, zMon, MonCoef,
     &	  lFF, pFF, xFF, yFF, zFF, FFCoef,
     &	  FFZernTypeL, FFZernCoef, MonZernTypeL, MonZernCoef,
     &	  nGridMat, iEltToGridSrf, GridMat, GridSrfdx,
     &	  pData, xData, yData, zData, mElt, mFFCoef, mMonCoef,
     &	  ZernType_ANSI, ZernType_BornWolf, ZernType_Fringe,
     &	  ZernType_NormANSI, ZernType_NormBornWolf,
     &	  ZernType_NormFringe, ZernType_NormHex,
     &	  ZernType_Noll, ZernType_NormNoll,
     &	  ZernType_NormAnnularNoll, ZernAnnuRatio
	USE param_mod,   ONLY: mGridMat
	USE cfiles_mod,  ONLY: GridFile
	USE surfsub,     ONLY: ZerntoMon1, ZerntoMon2, ZerntoMon3,
     &	  ZerntoMon4, ZerntoMon6, ZerntoMon7
	USE segmir_parent_mod
	IMPLICIT NONE
	INTEGER, INTENT(IN) :: iParent
	INTEGER i,j,jG

	IF (iParent.LT.1 .OR. iParent.GT.mElt) THEN
	  WRITE(*,*) ' LoadParent: parent element ',iParent,
     &	             ' out of range.'
	  STOP
	END IF


	Kc_p = KcElt(iParent)
	Kr_p = KrElt(iParent)
C  fElt/eElt are not populated by the OLD parser -- derive from Kc,Kr
C  (same convention used by SMPGe's dialog defaults).
	IF (Kc_p.LT.0d0) THEN
	  e_p = SQRT(-Kc_p)
	ELSE
	  e_p = 0d0
	END IF
	IF ((1d0+e_p).NE.0d0) THEN
	  f_p = -Kr_p/(1d0+e_p)
	ELSE
	  f_p = 0d0
	END IF
	DO i=1,3
	  psi_p(i) = psiElt(i,iParent)
	  pv_p(i)  = VptElt(i,iParent)
	END DO

	lMon_p = lMon(iParent)
	DO i=1,3
	  pMon_p(i) = pMon(i,iParent)
	  xMon_p(i) = xMon(i,iParent)
	  yMon_p(i) = yMon(i,iParent)
	  zMon_p(i) = zMon(i,iParent)
	END DO
	DO i=1,mMonCoef
	  MonCoef_p(i)    = MonCoef(i,iParent)
	  MonZernCoef_p(i)= MonZernCoef(i,iParent)
	END DO

	lFF_p = lFF(iParent)
	DO i=1,3
	  pFF_p(i) = pFF(i,iParent)
	  xFF_p(i) = xFF(i,iParent)
	  yFF_p(i) = yFF(i,iParent)
	  zFF_p(i) = zFF(i,iParent)
	END DO
	DO i=1,mFFCoef
	  FFCoef_p(i)    = FFCoef(i,iParent)
	  FFZernCoef_p(i)= FFZernCoef(i,iParent)
	END DO
	FFZernTypeL_p  = FFZernTypeL(iParent)
	MonZernTypeL_p = MonZernTypeL(iParent)

	nGridMat_p  = nGridMat(iParent)
	GridSrfdx_p = GridSrfdx(iParent)
	GridFile_p  = GridFile(iParent)
	DO i=1,3
	  pData_p(i) = pData(i,iParent)
	  xData_p(i) = xData(i,iParent)
	  yData_p(i) = yData(i,iParent)
	  zData_p(i) = zData(i,iParent)
	END DO

	IF (ALLOCATED(GridMat_p)) DEALLOCATE(GridMat_p)
	IF (nGridMat_p.GT.0) THEN
	  jG = iEltToGridSrf(iParent)
	  IF (jG.LE.0) THEN
	    WRITE(*,*) ' LoadParent: nGridMat>0 but iEltToGridSrf=0'
	    STOP
	  END IF
	  ALLOCATE(GridMat_p(nGridMat_p,nGridMat_p))
	  DO j=1,nGridMat_p
	    DO i=1,nGridMat_p
	      GridMat_p(i,j) = GridMat(i,j,jG)
	    END DO
	  END DO
	END IF

C  Populate FFCoef_p / MonCoef_p (monomial form) from the parent's
C  FFZernCoef / MonZernCoef (Zernike form).  In MACOS this conversion
C  normally happens in CTRACE at trace time (tracesub.F:1316-1395), so
C  if SegMirMaker reads FFCoef/MonCoef directly out of elt_mod after
C  OLD (with no intervening trace), the arrays are still zero and
C  SurfCoordFF would compute the conic-only intersection -- placing
C  RptElt off the actual FreeForm surface by the FF sag (~mm).  Mirror
C  the trace-time dispatch here so RptElt lands on the FreeForm
C  surface that MACOS will see at trace time.

C  -- FF Zernikes --
	IF (FFZernTypeL_p.EQ.ZernType_ANSI .OR.
     &      FFZernTypeL_p.EQ.ZernType_NormANSI) THEN
	  CALL ZerntoMon1(FFZernCoef_p, FFCoef_p, iParent)
	ELSE IF (FFZernTypeL_p.EQ.ZernType_BornWolf .OR.
     &           FFZernTypeL_p.EQ.ZernType_NormBornWolf) THEN
	  CALL ZerntoMon2(FFZernCoef_p, FFCoef_p, iParent)
	ELSE IF (FFZernTypeL_p.EQ.ZernType_Fringe .OR.
     &           FFZernTypeL_p.EQ.ZernType_NormFringe) THEN
	  CALL ZerntoMon3(FFZernCoef_p, FFCoef_p, iParent)
	ELSE IF (FFZernTypeL_p.EQ.ZernType_NormHex) THEN
	  CALL ZerntoMon4(FFZernCoef_p, FFCoef_p)
	ELSE IF (FFZernTypeL_p.EQ.ZernType_NormNoll .OR.
     &           FFZernTypeL_p.EQ.ZernType_Noll) THEN
	  CALL ZerntoMon6(FFZernCoef_p, FFCoef_p, iParent)
	ELSE IF (FFZernTypeL_p.EQ.ZernType_NormAnnularNoll) THEN
	  CALL ZerntoMon7(FFZernCoef_p, FFCoef_p,
     &                    ZernAnnuRatio(iParent))
	END IF

C  -- Mon Zernikes --
	IF (MonZernTypeL_p.EQ.ZernType_ANSI .OR.
     &      MonZernTypeL_p.EQ.ZernType_NormANSI) THEN
	  CALL ZerntoMon1(MonZernCoef_p, MonCoef_p, iParent)
	ELSE IF (MonZernTypeL_p.EQ.ZernType_BornWolf .OR.
     &           MonZernTypeL_p.EQ.ZernType_NormBornWolf) THEN
	  CALL ZerntoMon2(MonZernCoef_p, MonCoef_p, iParent)
	ELSE IF (MonZernTypeL_p.EQ.ZernType_Fringe .OR.
     &           MonZernTypeL_p.EQ.ZernType_NormFringe) THEN
	  CALL ZerntoMon3(MonZernCoef_p, MonCoef_p, iParent)
	ELSE IF (MonZernTypeL_p.EQ.ZernType_NormHex) THEN
	  CALL ZerntoMon4(MonZernCoef_p, MonCoef_p)
	ELSE IF (MonZernTypeL_p.EQ.ZernType_NormNoll .OR.
     &           MonZernTypeL_p.EQ.ZernType_Noll) THEN
	  CALL ZerntoMon6(MonZernCoef_p, MonCoef_p, iParent)
	ELSE IF (MonZernTypeL_p.EQ.ZernType_NormAnnularNoll) THEN
	  CALL ZerntoMon7(MonZernCoef_p, MonCoef_p,
     &                    ZernAnnuRatio(iParent))
	END IF

	parentElt  = iParent
	parentIsFF = .TRUE.

	RETURN
	END

C***********************************************************************
C  WriteSegBlock
C
C  Emit one element block.  parentIsFF=.FALSE. preserves the original
C  SMPGe output byte-for-byte (EltType=5 + conic).  parentIsFF=.TRUE.
C  emits a FreeForm element (Element=Segment + Surface=FreeForm) with
C  parent FF / grid data replicated in each segment's FF / grid slots
C  and empty Mon slot.
C***********************************************************************

	SUBROUTINE WriteSegBlock(iUnit, iPrt, Cinteger,
     &	                         f, e, psi, pv, pr, iDOF, TElt,
     &	                         xhat, yhat, zhat)
	USE elt_mod, ONLY: mFFCoef, mMonCoef, ZernTypeNameL
	USE segmir_parent_mod
	IMPLICIT NONE
	INTEGER, INTENT(IN) :: iUnit, iPrt, iDOF
	CHARACTER(*), INTENT(IN) :: Cinteger
	REAL*8, INTENT(IN) :: f, e, psi(3), pv(3), pr(3), TElt(6,6)
	REAL*8, INTENT(IN) :: xhat(3), yhat(3), zhat(3)

	INTEGER i, k, nFFnz, nMonnz
	INTEGER modesFF(mFFCoef), modesMon(mMonCoef)
	REAL*8  coefsFF(mFFCoef), coefsMon(mMonCoef)
	CHARACTER*32 zernTypeFF, zernTypeMon
	CHARACTER(LEN=300) :: line

 601	FORMAT(' FFZernType=  ',A)
 602	FORMAT('nFFZernCoef= ',i4)
 603	FORMAT('FFZernModes= ',6(i4,1x))
 607	FORMAT('MonZernType=  ',A)
 608	FORMAT('nMonZernCoef=    1')
 611	FORMAT(' nGridMat= ',i4)
 612	FORMAT(' GridFile=  ',A)
 621	FORMAT('nMonZernCoef= ',i4)
 622	FORMAT('MonZernModes= ',6(i4,1x))

	IF (.NOT.parentIsFF) THEN
C	  Legacy conic segment output
	  WRITE(iUnit,'(A,I4)')    '    iElt=', iPrt
	  WRITE(iUnit,'(A,A)')     ' EltName=  Seg', TRIM(Cinteger)
	  WRITE(iUnit,'(A)')       ' EltType=  5'
	  WRITE(iUnit,'(A,2x,A)')  '    fElt=', TRIM(FmtD(f))
	  WRITE(iUnit,'(A,2x,A)')  '    eElt=', TRIM(FmtD(e))
	  WRITE(iUnit,'(A,3(2x,A))') '  psiElt=',
     &	    TRIM(FmtD(psi(1))),TRIM(FmtD(psi(2))),TRIM(FmtD(psi(3)))
	  WRITE(iUnit,'(A,3(2x,A))') '  VptElt=',
     &	    TRIM(FmtD(pv(1))),TRIM(FmtD(pv(2))),TRIM(FmtD(pv(3)))
	  WRITE(iUnit,'(A,3(2x,A))') '  RptElt=',
     &	    TRIM(FmtD(pr(1))),TRIM(FmtD(pr(2))),TRIM(FmtD(pr(3)))
	  WRITE(iUnit,'(A)')       '  IndRef=  1d0'
	  WRITE(iUnit,'(A,2x,A)')  '    zElt=', TRIM(FmtD(0d0))
	  WRITE(iUnit,'(A)')       ' PropType= 1'
	  WRITE(iUnit,'(A,I2)')    '  nECoord=', iDOF
	  WRITE(iUnit,'(A,6(2x,A))') '    TElt=',
     &	    (TRIM(FmtD(TElt(1,k))),k=1,6)
	  DO i=2,6
	    WRITE(iUnit,'(10x,6(2x,A))') (TRIM(FmtD(TElt(i,k))),k=1,6)
	  END DO
	  WRITE(iUnit,'()')
	  RETURN
	END IF

C  FreeForm segment output -----------------------------------------
	IF (FFZernTypeL_p.GE.1 .AND.
     &	    FFZernTypeL_p.LE.SIZE(ZernTypeNameL)) THEN
	  zernTypeFF = ZernTypeNameL(FFZernTypeL_p)
	ELSE
	  zernTypeFF = 'ANSI'
	END IF
	IF (MonZernTypeL_p.GE.1 .AND.
     &	    MonZernTypeL_p.LE.SIZE(ZernTypeNameL)) THEN
	  zernTypeMon = ZernTypeNameL(MonZernTypeL_p)
	ELSE
	  zernTypeMon = 'ANSI'
	END IF

	nFFnz = 0
	DO i=1,mFFCoef
	  IF (FFZernCoef_p(i).NE.0d0) THEN
	    nFFnz = nFFnz + 1
	    modesFF(nFFnz) = i
	    coefsFF(nFFnz) = FFZernCoef_p(i)
	  END IF
	END DO

	WRITE(iUnit,'(A,I4)')    '    iElt=', iPrt
	WRITE(iUnit,'(A,A)')     ' EltName=  Seg', TRIM(Cinteger)
	WRITE(iUnit,'(A)')       ' Element=  Segment'
	WRITE(iUnit,'(A)')       ' Surface=  FreeForm'
	WRITE(iUnit,'(A,2x,A)')  '    fElt=', TRIM(FmtD(f))
	WRITE(iUnit,'(A,2x,A)')  '    eElt=', TRIM(FmtD(e))
	WRITE(iUnit,'(A,2x,A)')  '   KrElt=', TRIM(FmtD(Kr_p))
	WRITE(iUnit,'(A,2x,A)')  '   KcElt=', TRIM(FmtD(Kc_p))
	WRITE(iUnit,'(A,3(2x,A))') '  psiElt=',
     &	  TRIM(FmtD(psi(1))),TRIM(FmtD(psi(2))),TRIM(FmtD(psi(3)))
	WRITE(iUnit,'(A,3(2x,A))') '  VptElt=',
     &	  TRIM(FmtD(pv(1))),TRIM(FmtD(pv(2))),TRIM(FmtD(pv(3)))
	WRITE(iUnit,'(A,3(2x,A))') '  RptElt=',
     &	  TRIM(FmtD(pr(1))),TRIM(FmtD(pr(2))),TRIM(FmtD(pr(3)))

	WRITE(iUnit,601)TRIM(zernTypeFF)
	IF (nFFnz.EQ.0) THEN
	  WRITE(iUnit,'(A)')'nFFZernCoef=    1'
	  WRITE(iUnit,'(A)')' FFZernCoef=  0d0'
	ELSE
	  WRITE(iUnit,602)nFFnz
	  WRITE(iUnit,603)(modesFF(i),i=1,nFFnz)
	  line = ''
	  DO i=1,MIN(nFFnz,6)
	    line = TRIM(line)//'  '//TRIM(FmtD(coefsFF(i)))
	  END DO
	  WRITE(iUnit,'(A,A)') ' FFZernCoef=', TRIM(line)
	  DO i=7,nFFnz,6
	    line = ''
	    DO k=i,MIN(i+5,nFFnz)
	      line = TRIM(line)//'  '//TRIM(FmtD(coefsFF(k)))
	    END DO
	    WRITE(iUnit,'(12x,A)') TRIM(line)
	  END DO
	END IF
	WRITE(iUnit,'(A,2x,A)') '     lFF=', TRIM(FmtD(lFF_p))

C  Mon: replicate parent's Mon (if any) into every segment's Mon slot.
C  Each segment's Mon coordinate frame is its own (pMon=RptElt,
C  {xMon,yMon,zMon}=segment face triad), emitted below via format 619.
	WRITE(iUnit,607)TRIM(zernTypeMon)
	nMonnz = 0
	DO i=1,mMonCoef
	  IF (MonZernCoef_p(i).NE.0d0) THEN
	    nMonnz = nMonnz + 1
	    modesMon(nMonnz) = i
	    coefsMon(nMonnz) = MonZernCoef_p(i)
	  END IF
	END DO
	IF (nMonnz.EQ.0) THEN
	  WRITE(iUnit,608)
	  WRITE(iUnit,622) 1
	  WRITE(iUnit,'(A)')' MonZernCoef=  0d0'
	ELSE
	  WRITE(iUnit,621)nMonnz
	  WRITE(iUnit,622)(modesMon(i),i=1,nMonnz)
	  line = ''
	  DO i=1,MIN(nMonnz,6)
	    line = TRIM(line)//'  '//TRIM(FmtD(coefsMon(i)))
	  END DO
	  WRITE(iUnit,'(A,A)') ' MonZernCoef=', TRIM(line)
	  DO i=7,nMonnz,6
	    line = ''
	    DO k=i,MIN(i+5,nMonnz)
	      line = TRIM(line)//'  '//TRIM(FmtD(coefsMon(k)))
	    END DO
	    WRITE(iUnit,'(13x,A)') TRIM(line)
	  END DO
	END IF
	WRITE(iUnit,'(A,2x,A)') '    lMon=', TRIM(FmtD(lMon_p))

	IF (nGridMat_p.GT.0) THEN
	  WRITE(iUnit,611)nGridMat_p
	  WRITE(iUnit,612)TRIM(GridFile_p)
	  WRITE(iUnit,'(A,2x,A)') ' GridSrfdx=', TRIM(FmtD(GridSrfdx_p))
	END IF

	WRITE(iUnit,'(A,3(2x,A))') '     pFF=',
     &	  TRIM(FmtD(pFF_p(1))),TRIM(FmtD(pFF_p(2))),TRIM(FmtD(pFF_p(3)))
	WRITE(iUnit,'(A,3(2x,A))') '     xFF=',
     &	  TRIM(FmtD(xFF_p(1))),TRIM(FmtD(xFF_p(2))),TRIM(FmtD(xFF_p(3)))
	WRITE(iUnit,'(A,3(2x,A))') '     yFF=',
     &	  TRIM(FmtD(yFF_p(1))),TRIM(FmtD(yFF_p(2))),TRIM(FmtD(yFF_p(3)))
	WRITE(iUnit,'(A,3(2x,A))') '     zFF=',
     &	  TRIM(FmtD(zFF_p(1))),TRIM(FmtD(zFF_p(2))),TRIM(FmtD(zFF_p(3)))
	WRITE(iUnit,'(A,3(2x,A))') '    pMon=',
     &	  TRIM(FmtD(pr(1))),TRIM(FmtD(pr(2))),TRIM(FmtD(pr(3)))
	WRITE(iUnit,'(A,3(2x,A))') '    xMon=',
     &	  TRIM(FmtD(xhat(1))),TRIM(FmtD(xhat(2))),TRIM(FmtD(xhat(3)))
	WRITE(iUnit,'(A,3(2x,A))') '    yMon=',
     &	  TRIM(FmtD(yhat(1))),TRIM(FmtD(yhat(2))),TRIM(FmtD(yhat(3)))
	WRITE(iUnit,'(A,3(2x,A))') '    zMon=',
     &	  TRIM(FmtD(zhat(1))),TRIM(FmtD(zhat(2))),TRIM(FmtD(zhat(3)))
	IF (nGridMat_p.GT.0) THEN
	  WRITE(iUnit,'(A,3(2x,A))') '   pData=',
     &	    TRIM(FmtD(pData_p(1))),TRIM(FmtD(pData_p(2))),
     &	    TRIM(FmtD(pData_p(3)))
	  WRITE(iUnit,'(A,3(2x,A))') '   xData=',
     &	    TRIM(FmtD(xData_p(1))),TRIM(FmtD(xData_p(2))),
     &	    TRIM(FmtD(xData_p(3)))
	  WRITE(iUnit,'(A,3(2x,A))') '   yData=',
     &	    TRIM(FmtD(yData_p(1))),TRIM(FmtD(yData_p(2))),
     &	    TRIM(FmtD(yData_p(3)))
	  WRITE(iUnit,'(A,3(2x,A))') '   zData=',
     &	    TRIM(FmtD(zData_p(1))),TRIM(FmtD(zData_p(2))),
     &	    TRIM(FmtD(zData_p(3)))
	END IF

	WRITE(iUnit,'(A)')       '  IndRef=  1d0'
	WRITE(iUnit,'(A)')       '    zElt=  0d0'
	WRITE(iUnit,'(A)')       ' PropType= 1'
	WRITE(iUnit,'(A,I2)')    '  nECoord=', iDOF
	WRITE(iUnit,'(A,6(2x,A))') '    TElt=',
     &	  (TRIM(FmtD(TElt(1,k))),k=1,6)
	DO i=2,6
	  WRITE(iUnit,'(10x,6(2x,A))') (TRIM(FmtD(TElt(i,k))),k=1,6)
	END DO
	WRITE(iUnit,'()')

	RETURN
	END

C***********************************************************************
