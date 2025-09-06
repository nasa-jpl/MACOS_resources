  module api
    use Kinds
    use math_mod
    use param_mod
    use src_mod
    use elt_mod
    use macos_mod
    use smacos_mod

    implicit none

    logical, parameter :: PASS = 1    ! logic Python / Fortran
    logical, parameter :: FAIL = 0

    real(8), dimension(:,:), allocatable, save :: PixArray, OPDMat, RaySpot, SPOT
    real(8), dimension(:),   allocatable, save :: OPD, PIX, USER

    logical, save      :: firstEntry=.true.
    logical, save      :: rxLoaded=.false.

    ! These are the SMACOS call-line variables:
    ! TODO: change len=256 to len=max_cmd_len
    character(len=256)  :: command
    integer,parameter   :: MARG=9
    character(len=256)  :: CARG(MARG)
    real(8)             :: DARG(MARG)
    integer             :: IARG(MARG)
    logical             :: LARG
    real(8)             :: RARG(MARG),RMSWFE

    ! Pt. Source
    real(8), parameter  :: zSourceMax = 1d10

    contains

      integer function currrent_macos_model_size()
        implicit none
        if (firstEntry) then
          currrent_macos_model_size = -1
        else
          currrent_macos_model_size = macos_model_size
        end if
      end function currrent_macos_model_size


    !=============================================================================================
    !
    ! Model & Optical System Setup
    !
    !---------------------------------------------------------------------------------------------
    !      [SMACOS]: modified_rx
    ! --------------------------------------------------------------------------------------------
    !      [Source]: set/get_src_info
    !      [Source]: set/get_src_sampling
    !      [OptSys]: stop_info_get
    !      [OptSys]: stop_info_set
    !      [OptSys]: xp_fnd
    !      [ ] prb_elt
    !      [ ] prb_elt_grp
    !=============================================================================================

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Submit a Rx modified cmd. to SMACOS to reset ray-trace dependent parameters, which
      !            is recommended after a Rx modification, i.e., perturbElt, setVptElt, ...
      ! Call     : CALL modified_rx(OK,XP)
      ! Input    : ---
      ! Output   : ok   [1x1,B]: = (True,1) if successful; (False,0) otherwise
      ! Require  : Rx loaded
      !---------------------------------------------------------------------------------------------
      subroutine modified_rx(ok)

        implicit none
        logical, intent(out):: ok
        ! ------------------------------------------------------
        ok = FAIL
        if (.not. SystemCheck()) return

        command = 'MODIFY'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ok = PASS

      end subroutine modified_rx


      !---------------------------------------------------------------------------------------------
      ! Apply 6-DoF rigid body perturbation to selected elements defined by iElt
      !---------------------------------------------------------------------------------------------
      !   ifGlobal [Nx1]: (1=True ):Global Coordinate Frame
      !                   (0=False):Local Element Coordinate Frame  (must be defined in Rx)
      !   prb      [6xN]: Rigid Body Perturbation Matrix
      !                   = [[Rx,Ry,Rz,Tx,Ty,Tz]_1;...;[Rx,Ry,Rz,Tx,Ty,Tz]_N]
      !                   Rotation    Vector: R = [Rx,Ry,Rz]_i
      !                   Translation Vector: T = [Tx,Ty,Tz]_i
      !---------------------------------------------------------------------------------------------
      subroutine prb_elt(OK,iElt,prb,ifGlobal,n)

        implicit none
        logical,                 intent(out):: OK        ! (True) if successful; (False) otherwise
        integer, dimension(n),   intent(in) :: iElt      ! ID (Range: -nElt < iElt[i,j] <= nElt)
        real(8), dimension(6,n), intent(in) :: prb
        integer, dimension(n),   intent(in) :: ifGlobal

        integer,                 intent(in) :: n         ! Number of different elements to be updated
        !f2py  integer intent(hide), depend(iElt) :: n=len(iElt)

        integer :: i
        ! ------------------------------------------------------
        OK = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(i,j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check: if local coordinate frame selected -- is it defined?
        if (.not. all(((ifGlobal==0) .and. (nECoord(iElt)>0)) .or. (ifGlobal==1))) return

        ! loop over different perturbation settings
        do i=1,n

          command = 'PERTURB'
          IARG(1) = iElt(i)
          if (ifGlobal(i)>0) then  ! define coordinate frame: (0):Element, (1):Global
            CARG(1) = 'GLOBAL'
          else
            CARG(1) = 'ELEMENT'
          end if
          DARG(1:6) = prb(:,i)
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        end do

        OK = PASS

      end subroutine prb_elt


      !---------------------------------------------------------------------------------------------
      ! Apply 6-DoF rigid body perturbation to selected elements identified by "EltGrp" keyword
      ! at a given element.
      !---------------------------------------------------------------------------------------------
      !   ifGlobal [Nx1]: (1=True ):Global Coordinate Frame
      !                   (0=False):Local Element Coordinate Frame  (must be defined in Rx)
      !   prb      [6xN]: Rigid Body Perturbation Matrix
      !                   = [[Rx,Ry,Rz,Tx,Ty,Tz]_1;...;[Rx,Ry,Rz,Tx,Ty,Tz]_N]
      !                   Rotation    Vector: R = [Rx,Ry,Rz]_i
      !                   Translation Vector: T = [Tx,Ty,Tz]_i
      !---------------------------------------------------------------------------------------------
      subroutine prb_elt_grp(OK,iElt,prb,ifGlobal,n)

        implicit none
        logical,                 intent(out):: OK        ! (True) if successful; (False) otherwise
        integer, dimension(n),   intent(in) :: iElt      ! ID (Range: -nElt < iElt[i,j] <= nElt)
        real(8), dimension(6,n), intent(in) :: prb
        logical, dimension(n),   intent(in) :: ifGlobal

        integer,                 intent(in) :: n         ! Number of different elements to be updated
        !f2py  integer intent(hide), depend(iElt) :: n=len(iElt)

        integer :: i
        ! ------------------------------------------------------
        OK = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(i,j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check: if local coordinate frame selected -- is it defined?
        if (.not. all(((ifGlobal==0) .and. (nECoord(iElt)>0)) .or. (ifGlobal==1))) return

        ! check: if Element Group(s) are defined at all defined element(s)
        if (any(EltGrp(0, iElt)==0)) return

        ! loop over different perturbation settings
        do i=1,n

          ! loop over identical perturbation settings
          command   = 'GPERTURB'
          IARG(1)   = iElt(i)
          if (ifGlobal(i)>0) then  ! define coordinate frame: (0):Element, (1):Global
            CARG(1) = 'GLOBAL'
          else
            CARG(1) = 'ELEMENT'
          end if
          DARG(1:6) = prb(:,i)
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        end do

        OK = PASS

      end subroutine prb_elt_grp



    !=============================================================================================
    !
    ! Source
    !
    !=============================================================================================

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Set Source Coordinate Frame with Chf. Ray Direction as z-Axis.
      !
      ! Call     : CALL set_src_csys(ok,xDir,yDir,zDir,Axis,xAxis,Rz,filter)
      ! Input    : Axis   [3x1,D   ]: = [L,M,N] => x- or y-Axis expressed in GCF (1 = L^2+M^2+N^2 )
      !            xAxis  [1x1,I(1)]: if set (/=0), Axis == x-Axis; otherwise, y-Axis
      !            Rz     [1x1,D   ]: [rad] Rot. mag. for post. rot. about zDir = zGrid = ChfRayDir
      !            filter [1x1,I(1)]: if set (/=0), set to 0 if -eps < axis dir. cosine values  < eps
      ! Output   : ok     [1x1,B   ]: = (True) if successful; (False) otherwise
      !            xDir   [3x1,D]: = [Lx,Ly,Lz] => Src. Coord. Frame:  x-axis expressed in GCF
      !            yDir   [3x1,D]: = [Mx,My,Mz] => Src. Coord. Frame:  y-axis expressed in GCF
      !            zDir   [3x1,D]: = [Nx,Ny,Nz] => Src. Coord. Frame:  z-axis expressed in GCF
      ! Require  : check if pymacos is initialized & Rx loaded
      ! Note     : -- GCF => Global Coordinate Frame
      !            -- zDir <= ChfRayDir
      !            -- will orthonormalize:
      !                 if xAxis yDir <= cross(zDir,xDir)   else   xDir <= cross(yDir,zDir)
      !                          xDir <= cross(yDir,zDir)          yDir <= cross(zDir,xDir)
      !            -- rotation will be applied afterwards about zGrid = ChfRayDir, i.e.,
      !                          xDir <= Rot(Rz)*xDir   and
      !                          yDir <= Rot(Rz)*yDir
      !---------------------------------------------------------------------------------------------
      subroutine set_src_csys(ok,xDir,yDir,zDir,Axis,xAxis,Rz,filter)
        use Constants, only: eps
        use   src_mod, only: xGrid, yGrid, zGrid, ChfRayDir

        implicit none
        logical, intent(out):: ok
        real(8), intent(out):: xDir(3),yDir(3),zDir(3)
        real(8), intent(in) :: Axis(3)
        logical, intent(in) :: xAxis
        real(8), intent(in) :: Rz
        logical, intent(in) :: filter

        real(8)  :: Q(3,3), dQ(3,3)
        logical  :: LQ
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status chk:
        if (.not. SystemCheck()) return

        ! check for null-vector
        if (norm2(Axis)<=eps) return

        ! define Src. Coord. Frame
        zDir = ChfRayDir

        ! setup Post-Rotation about zDir = zGrid = ChfRayDir
        LQ = abs(Rz)>eps
        if (LQ) CALL Qform(Q,dQ,zDir*Rz)   ! rot. matrix

        ! set Source Coordinate Frame:
        if (xAxis==PASS) then
          !
          ! xDir provided
          !
          xDir = Axis
          CALL DUNITIZE(xDir)

          ! yGrid <= cross(zGrid,xGrid)
          CALL DXPROD(yDir,zDir,xDir)
          if (norm2(yDir)<=eps) return

          ! apply post rot.
          if (LQ) yDir = matmul(Q,yDir)

          ! xGrid <= cross(yGrid,zGrid)
          if (filter/=0) where (abs(yDir)<eps) yDir = 0e0_pr
          CALL DUNITIZE(yDir)
          CALL DXPROD(xDir,yDir,zDir)

        else
          !
          ! yDir provided
          !
          yDir = Axis
          CALL DUNITIZE(yDir)

          ! xGrid <= cross(yGrid,zGrid)
          CALL DXPROD(xDir,yDir,zDir)
          if (norm2(xDir)<=eps) return

          ! apply post rot.
          if (LQ) xDir = matmul(Q,xDir)

          ! yGrid <= cross(zGrid,xGrid)
          if (filter/=0) where (abs(xDir)<eps) xDir = 0e0_pr
          CALL DUNITIZE(xDir)
          CALL DXPROD(yDir,zDir,xDir)

        end if

        ! update actual Src. Grid
        xGrid = xDir
        yGrid = yDir
        zGrid = zDir

        ! return
        ok = PASS

      end subroutine set_src_csys

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Get current setting of Source Coordinate Frame (xGrid,yGrid,zGrid)
      !
      ! Call     : CALL get_src_csys(ok,xDir,yDir,zDir)
      ! Input    : ---
      ! Output   : ok     [1x1,B]: = (True) if successful; (False) otherwise
      !            xDir   [3x1,D]: = [Lx,Ly,Lz] => x-axis expressed in GCF
      !            yDir   [3x1,D]: = [Mx,My,Mz] => y-axis expressed in GCF
      !            zDir   [3x1,D]: = [Nx,Ny,Nz] => z-axis expressed in GCF
      ! Require  : check if pymacos is initialized & Rx loaded
      ! Note     : -- GCF => Global Coordinate Frame
      !---------------------------------------------------------------------------------------------
      subroutine get_src_csys(ok,xDir,yDir,zDir)
        use src_mod, only: xGrid,yGrid,zGrid

        implicit none
        logical,               intent(out):: ok
        real(8), dimension(3), intent(out):: xDir, yDir, zDir
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status chk:
        if (.not. SystemCheck()) return

        ! get Src. Coord. Frame Info
        xDir = xGrid
        yDir = yGrid
        zDir = zGrid

        ! return
        ok = PASS

      end subroutine get_src_csys

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Set Sampling of Source, i.e., nGridPts, where nGridPts <= ModelSize   (will be limited)
      ! Call     : CALL set_src_sampling(OK,nGridPts)
      ! Input    : N         [1x1,I]: Source Sampling -> N = nGridPts (Sampling => nGridPts x nGridPts)
      ! Output   : OK        [1x1,B]: = (True) if successful; (False) otherwise
      ! Require  : check if Rx is loaded
      ! Note     : - 3 <= nGridPts <= mpts ---> limit nGridPts to (3,mpts)
      !---------------------------------------------------------------------------------------------
      subroutine set_src_sampling(OK,N)
        use smacos_vars_mod, only: npts

        implicit none
        logical, intent(out):: OK
        integer, intent(in) :: N
        !f2py  intent(hide)        :: OK
        ! ------------------------------------------------------
        ! initialisation
        OK = FAIL

        ! Rx loaded
        if (.not. SystemCheck()) return

        ! Change # of Rays to be traced (wavefront sampling)
        if (N>mpts) then
          nGridPts = mpts
        elseif (N<3) then
          nGridPts = 3
        else
          nGridPts = N
        end if
        npts = nGridPts-1

        command = 'MODIFY'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ! return
        OK = PASS

      end subroutine set_src_sampling

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Return Source Sampling setting, i.e., nGridPts.
      ! Call     : CALL get_src_sampling(OK,nGridPts)
      ! Input    : ---
      ! Output   : OK    [1x1,B]: = (True) if successful; (False) otherwise
      !            N     [1x1,I]: Source Sampling -> N = nGridPts (Sampling = N x N)
      ! Require  : check if Rx is loaded
      ! Note     :
      !---------------------------------------------------------------------------------------------
      subroutine get_src_sampling(OK,N)

        implicit none
        logical, intent(out):: OK
        integer, intent(out):: N
        !f2py  intent(hide)        :: OK, N

        ! ------------------------------------------------------
        ! initialisation
        OK = FAIL
        N  = 0

        ! Rx loaded
        if (.not. SystemCheck()) return

        ! Change # of Rays to be traced (wavefront sampling)
        N = nGridPts

        ! return
        OK = PASS

      end subroutine get_src_sampling


      !-------------------------------------------------------------------------------------------------------------
      ! Purpose  : Get BaseUnit and WaveUnit
      !------------------------------------------------------------------------------------------------------------
      subroutine sys_units(OK, BaseUnitID, WaveUnitID)
        implicit none
        logical, intent(out):: OK
        integer, intent(out):: BaseUnitID    ! (1) 'm', (2) 'cm', (3) 'mm', (4) 'in', (0) 'none' = not set
        integer, intent(out):: WaveUnitID    ! (1) 'm', (2) 'cm', (3) 'mm', (4) 'um', (5) 'nm', (6) 'A', (7) 'in', (9) 'none'= not set
        !f2py  intent(hide):: OK,BaseUnitID,WaveUnitID

        integer, parameter :: MaxBaseUnit = 10, &
                              MaxWaveUnit = 16
        character(len=*), dimension(MaxBaseUnit), parameter:: BaseUnits_ = (/'m', 'cm', 'mm', 'in', 'none', 'M', 'CM', 'MM', 'IN', 'NONE'/)
        character(len=*), dimension(MaxWaveUnit), parameter:: WaveUnits_ = (/'m', 'cm', 'mm', 'um', 'nm', 'A', 'in', 'none','M', 'CM', 'MM', 'UM', 'NM', 'A', 'IN', 'NONE'/)
        ! ------------------------------------------------------
        ! initialisation
        OK         = FAIL
        BaseUnitID = -1
        WaveUnitID = -1

        if (.not. SystemCheck()) return

        do BaseUnitID=1,MaxBaseUnit
          if (trim(BaseUnits) == trim(BaseUnits_(BaseUnitID))) exit
        end do
        BaseUnitID = modulo(BaseUnitID, MaxBaseUnit/2)

        do WaveUnitID=1,MaxWaveUnit
          if (trim(WaveUnits) == trim(WaveUnits_(WaveUnitID))) exit
        end do
        WaveUnitID = modulo(WaveUnitID, MaxWaveUnit/2)

        OK = PASS

      end subroutine sys_units


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Retrieve active source information: shape, position & WL
      !
      !               zSrc: Distance from Src. Position to Spherical wavefront position (=zSource)
      !                     0 < |zSrc| <= 1d10: Pt. Src.: if zSrc<0 -> converging wave (to   Pt. Src.)
      !                                                   if zSrc>0 -> diverging  wave (from Pt. Src.)
      !                         |zSrc|  > 1d10: Col.Src.
      !             SrcPos: if Pt. Src.: SrcPos <= ChfRayPos + zSource*SrcDir
      !                     if Col.Src.: SrcPos <= ChfRayPos
      !             SrcDir: Centre Beam Direction  (= ChfRayDir)
      !            IsPtSrc: (true) if |zSource| <= 1d10; otherwise, (false)
      !                 WL: Source Wavelength in WaveUnits
      !           Aperture: if Pt. Src. => N.A. of beam
      !                     if Col.Src. => Beam Diameter in BaseUnits
      !           Obscratn: if Pt. Src. => N.A. of beam
      !                     if Col.Src. => Beam Diameter in BaseUnits
      !         BaseUnitID: (0) m, (1) cm, (2) mm, (3) in, (4) none  (-1) Error
      !
      ! Call     : CALL src_info(OK,zSource,SrcPos,SrcDir,IsPtSrcWL,SrcApe,SrcObs,BaseUnit)
      ! Input    : ---
      ! Output   : OK            [1x1,B]: = (True) if successful; (False) otherwise
      !            zSrc          [1x1,D]: Distance from Src. Position to Spherical wavefront position
      !            SrcPos        [1x3,D]: Src. Position  if Col.Src, SrcPos = ChfRayPos
      !            SrcDir        [1x3,D]: Centre Beam Direction (= ChfRayDir)
      !            IsPtSrc       [1x1,B]: (true=1) if |zSource| <= 1d10; otherwise, returns (false=0)
      !            WL            [1x1,D]: Wavelength in WaveUnits
      !            SrcApe        [1x1,D]: Ape. Beam N.A. for Pt.Src.;otherwise, Apt. == Beam Diameter
      !            SrcObs        [1x1,D]: Obs. Beam N.A. for Pt.Src.;otherwise, Obs. == Beam Diameter
      ! Require  : Rx loaded
      !---------------------------------------------------------------------------------------------
      subroutine src_info(OK,zSrc,SrcPos,SrcDir,IsPtSrc,WL,SrcApe,SrcObs,BaseUnitID,WaveUnitID)

        implicit none
        logical,               intent(out):: OK
        real(8),               intent(out):: zSrc
        real(8), dimension(3), intent(out):: SrcPos,SrcDir
        logical,               intent(out):: IsPtSrc
        real(8),               intent(out):: WL,SrcApe,SrcObs
        integer,               intent(out):: BaseUnitID         ! (1) 'm', (2) 'cm', (3) 'mm', (4) 'in', (0) 'none' = not set
        integer,               intent(out):: WaveUnitID         ! (1) 'm', (2) 'cm', (3) 'mm', (4) 'um', (5) 'nm', (6) 'A', (7) 'in', (9) 'none'= not set
        !f2py   intent(hide):: OK,zSrc,SrcPos,SrcDir,IsPtSrc,WL,SrcApe,SrcObs,BaseUnitID,WaveUnitID
        ! ------------------------------------------------------

        OK         = FAIL
        zSrc       = 0e0_pr
        SrcPos     = 0e0_pr
        SrcDir     = 0e0_pr
        IsPtSrc    = .false.
        WL         = 0e0_pr
        SrcApe     = 0e0_pr
        SrcObs     = 0e0_pr
        BaseUnitID = -1
        WaveUnitID = -1

        if (.not. SystemCheck()) return

        ! get partial Src. Info.
        WL      = Wavelen
        zSrc    = zSource
        call src_finite(OK, IsPtSrc) ! abs(zSource) <= zSourceMax
        SrcApe  = Aperture
        SrcObs  = Obscratn
        SrcDir  = ChfRayDir
        SrcPos  = ChfRayPos

        if (IsPtSrc==PASS) SrcPos = SrcPos + zSource*ChfRayDir

        CALL sys_units(OK, BaseUnitID, WaveUnitID)

        ! return
        OK = PASS

      end subroutine src_info


      !-------------------------------------------------------------------------------------------------------------
      ! Retrieve active source Field-of-View (FoV) Position & Direction
      !------------------------------------------------------------------------------------------------------------
      subroutine get_src_fov(OK, zSrc, SrcPos, SrcDir, IsPtSrc)
        use sourcsub_mod, only: SourcePos
        use dopt_mod

        implicit none
        logical,               intent(out):: OK         ! PASS = if successful; (FAIL) otherwise
        logical,               intent(out):: IsPtSrc    ! PASS if |zSource| <= 1d10; otherwise, FAIL
        real(8),               intent(out):: zSrc       ! Distance from wavefront position to Src. Pos. (= zSource)
        real(8), dimension(3), intent(out):: SrcPos     ! if Point Src: SrcPos = ChfRayPos + zSource*ChfRayDir
                                                        !     Inf. Src: SrcPos = ChfRayPos
        real(8), dimension(3), intent(out):: SrcDir     ! Centre Beam Direction (= ChfRayDir) -> will be normalized
        ! ------------------------------------------------------
        OK      = FAIL
        zSrc    = 0e0_pr
        SrcPos  = 0e0_pr
        SrcDir  = 0e0_pr
        IsPtSrc = FAIL

        if (.not. SystemCheck()) return

        ! Get FoV.
        zSrc    = zSource
        call src_finite(OK, IsPtSrc) ! abs(zSource) <= zSourceMax
        SrcDir  = ChfRayDir
        SrcPos  = ChfRayPos
        if (IsPtSrc) SrcPos = SourcePos()  ! SrcPos + zSource*ChfRayDir

        ! Return
        OK = PASS

        ! needed because python interprets Fortran(True) = -1 and Fortran(False) = 0
        if (IsPtSrc) then
        	 IsPtSrc = PASS
        else
        	 IsPtSrc = FAIL
        end if
      end subroutine get_src_fov


      !---------------------------------------------------------------------------------------------
      ! Set / Get Source Wavelength (WVL) in units of Rx defined WaveUnits
      !---------------------------------------------------------------------------------------------
      subroutine src_wvl(OK, WVL, setter)
        implicit none
        integer, intent(out)   :: OK       ! (PASS) if successful; (FAIL) otherwise
        real(8), intent(inout) :: WVL      ! Wavelength in WaveUnits (WVL > 0)
        integer, intent(in)    :: setter   ! (PASS) to set & (FAIL) to get
        ! ------------------------------------------------------
        OK = FAIL
        if (setter==FAIL) WVL = 0d0

        if (.not. SystemCheck()) return

        if (setter==PASS) then

          if (isnan(WVL)) return
          if ((WVL<=0e0_pr).or. .not.(abs(WVL)<=huge(WVL))) return    ! valid input - hmm, check range?

          Wavelen = WVL         ! in WaveUnits
          WaveBU  = CWB*WVL     ! in internal units <= CWB = CWM/CBM (CWM = WaveUnits, CBM = BaseUnits)

        else
          WVL = Wavelen
        end if

        OK = PASS
      end subroutine src_wvl


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Define active source information: shape, position & WL
      !
      !               zSrc: Distance from Src. Position to Spherical wavefront position (=zSource)
      !                     0 < |zSrc| <= 1d10: Pt. Src.: if zSrc<0 -> converging wave (to   Pt. Src.)
      !                                                   if zSrc>0 -> diverging  wave (from Pt. Src.)
      !                         |zSrc|  > 1d10: Col.Src.
      !             SrcPos: if Pt. Src.: SrcPos <= ChfRayPos + zSource*SrcDir (ChfRayDir if SrcDir not defined)
      !                     if Col.Src.: SrcPos <= ChfRayPos
      !             SrcDir: Centre Beam Direction  (= ChfRayDir)
      !            IsPtSrc: (true) if |zSource| <= 1d10; otherwise, returns (false)
      !                 WL: Source Wavelength in WaveUnits
      !           Aperture: if Pt. Src. => N.A. of beam
      !                     if Col.Src. => Beam Diameter in BaseUnits
      !           Obscratn: if Pt. Src. => N.A. of beam
      !                     if Col.Src. => Beam Diameter in BaseUnits
      !
      ! Call     : CALL set_src_info(OK,zSrc,SrcPos,SrcDir,WL,SrcApe,SrcObs)
      ! Input    : zSrc          [1x1,D]: Distance from Src. Position to Spherical wavefront position (=zSource)
      !            SrcPos        [1x3,D]: Src. Position  if Col.Src, SrcPos = ChfRayPos
      !            SrcDir        [1x3,D]: Centre Beam Direction (= ChfRayDir) -> will be normalized
      !            WL            [1x1,D]: Wavelength in WaveUnits
      !            SrcApe        [1x1,D]: Ape. Beam N.A. for Pt.Src.;otherwise, Apt. == Beam Diameter
      !            SrcObs        [1x1,D]: Obs. Beam N.A. for Pt.Src.;otherwise, Obs. == Beam Diameter
      ! Output   : OK            [1x1,B]: = (True) if successful; (False) otherwise
      ! Require  : check if Rx is loaded
      ! Note     : - if abs(zSource)>1d10 ==> Col. Src.; otherwise, Pt.Src.
      !            - for NAN values, parameters will not be updated
      !            - if SrcPos<>NAN then require zSrc<>NAN
      !            - if defined: require WL<>0, SrcApe>0, SrcObs<SrcApe, SrcObs>=0
      !---------------------------------------------------------------------------------------------
      subroutine set_src_info(OK,zSrc,SrcPos,SrcDir,WL,SrcApe,SrcObs)
        use Constants, only: eps

        implicit none
        logical,               intent(out):: OK
        real(8),               intent(in) :: zSrc
        real(8), dimension(3), intent(in) :: SrcPos,SrcDir
        real(8),               intent(in) :: WL,SrcApe,SrcObs
        !f2py  intent(hide):: OK

        integer :: tmp
        ! ------------------------------------------------------
        ! initialisation
        OK = FAIL

        ! Rx Loaded?
        if (.not. SystemCheck()) return

        ! updated Src. parameters
        if (.not. isnan(WL)) then
          if (WL<=eps) return
          Wavelen = WL
        end if

        if (count(isnan((/SrcApe,SrcObs/)))==0) then
          if ((abs(SrcApe)<=abs(SrcObs)).or.(SrcApe<=eps).or.(SrcObs<0e0_pr)) return
          Aperture = SrcApe
          Obscratn = SrcObs
        else
          if (.not. isnan(SrcApe)) then
            if ((SrcApe<=eps).or.(SrcApe<=Obscratn)) return
            Aperture = SrcApe
          end if

          if (.not. isnan(SrcObs)) then
            if ((abs(SrcObs)>=Aperture).or.(SrcObs<0e0_pr)) return
            Obscratn = SrcObs
          end if
        end if

        if (.not. any(isnan(SrcDir))) then
          if (.not. any(abs(SrcDir)>eps)) return
          ChfRayDir = SrcDir/sqrt(sum(SrcDir**2))
        end if

        tmp = count((/isnan(zSrc),(count(isnan(SrcPos))/=0)/)) ! maybe better to write out
        if (tmp==1) return
        if (tmp==0) then
          zSource   = zSrc
          ChfRayPos = SrcPos
          if (abs(zSource) <= zSourceMax) &
            ChfRayPos = ChfRayPos - zSource*ChfRayDir
        end if

        ! return
        OK = PASS

      end subroutine set_src_info


      !---------------------------------------------------------------------------------------------
      ! Purpose: Define active source Field-of-View (FoV) Position
      ! Call   : CALL set_src_fov(OK,zSrc,SrcPos,SrcDir)
      ! Input  : zSrc   [1x1,D]: Distance from wavefront position to Src. Pos. (= zSource)
      !          SrcPos [1x3,D]: Src. Position:  if Col.Src, SrcPos = ChfRayPos
      !          SrcDir [1x3,D]: Centre Beam Direction (= ChfRayDir) -> will be normalized
      ! Output : OK     [1x1,B]: = (True,1) if successful; (False,0) otherwise
      ! Require  : check if Rx is loaded
      ! Note     : -zSrc    0 < |zSrc| <= 1d10: Pt. Src.: if zSrc<0 -> converging wave (to   Pt. Src.)
      !                                                   if zSrc>0 -> diverging  wave (from Pt. Src.)
      !                         |zSrc|  > 1d10: Col.Src.
      !            - SrcPos: if Pt. Src.: SrcPos = ChfRayPos + zSource*SrcDir (ChfRayDir if SrcDir not defined)
      !                      if Col.Src.: SrcPos = ChfRayPos
      !            - ChfRayDir <= SrcDir
      !            - ChfRayPos <= SrcPos - zSource*SrcDir
      !---------------------------------------------------------------------------------------------
      subroutine set_src_fov(OK,zSrc,SrcPos,SrcDir)
        use Constants, only: EPS

        implicit none
        logical,               intent(out):: OK
        real(8),               intent(in) :: zSrc
        real(8), dimension(3), intent(in) :: SrcPos,SrcDir

        real(8) :: tmp(7)
        ! ------------------------------------------------------
        ! initialisation
        OK  = FAIL
        tmp = (/zSrc,SrcPos,SrcDir/)

        ! Rx Loaded?
        if (.not. SystemCheck()) return

        ! input check
        if (any(isnan(tmp))) return                     ! check NaN
        if (any(.not.(abs(tmp)<=huge(zSrc)))) return    ! check inf
        if (.not. any(SrcDir/=0e0_pr)) return           ! check validity
        if (abs(zSrc) <= EPS) return                    ! zSrc cannot be 0

        ! set values
        zSource   = zSrc
        ChfRayDir = SrcDir/sqrt(sum(SrcDir**2))
        ChfRayPos = SrcPos
        if (abs(zSource) <= zSourceMax) ChfRayPos = ChfRayPos - zSource*ChfRayDir

        ! return
        OK = PASS

      end subroutine set_src_fov


      !---------------------------------------------------------------------------------------------
      ! Set/Get Source Size, i.e., Beam Extend (Aperture) & Beam Obscuration
      !
      !  SrcApe  [1x1]: if |zSource|<=1e10 (Pt. Src.) ==> Ape. Beam N.A.
      !                    |zSource| >1e10 (Col.Src.) ==> Apt. Beam Diameter (BaseUnits)
      !
      !  SrcObs  [1x1]: if |zSource|<=1e10 (Pt. Src.) ==> Obs. Beam N.A.
      !                    |zSource| >1e10 (Col.Src.) ==> Obs. Beam Diameter (BaseUnits)
      !
      !  Ignore value (when set) if
      !    SrcApe <= 0
      !    SrcObs <  0
      !---------------------------------------------------------------------------------------------
      subroutine src_size(OK, SrcApe, SrcObs, Setter)
        implicit none
        logical, intent(out)  :: OK      ! (PASS) if successful; (FAIL) otherwise
        real(8), intent(inout):: SrcApe  ! Ape. Beam N.A. for Pt.Src.;otherwise, Apt. Beam Diameter
        real(8), intent(inout):: SrcObs  ! Obs. Beam N.A. for Pt.Src.;otherwise, Obs. Beam Diameter
        integer, intent(in)   :: Setter  ! (PASS) to set & (FAIL) to get

        logical :: set_ape, set_obs
        ! ------------------------------------------------------
        OK = FAIL
        if (Setter==FAIL) then
          SrcApe = 0e0_pr
          SrcObs = 0e0_pr
        end if
        if (.not. SystemCheck()) return

        if (Setter==PASS) then

          if (isnan(SrcApe).or.isnan(SrcObs)) return                                           ! NaN   check (non standard)
          if (.not.(abs(SrcApe)<=huge(SrcApe)) .or. .not.(abs(SrcObs)<=huge(SrcObs))) return   ! inf   check

          set_ape = (SrcApe >  0e0_pr)
          set_obs = (SrcObs >= 0e0_pr)

          if (set_ape .and. set_obs) then
            if (SrcApe<=SrcObs) return   ! range check

            Aperture = SrcApe
            Obscratn = SrcObs

          else if (set_ape) then
            if (SrcApe<=Obscratn) return   ! range check

            Aperture = SrcApe

          else if (set_obs) then
            if (Aperture<=SrcObs) return   ! range check

            Obscratn = SrcObs

          else
            return   ! no values to be updated
          end if

        else

          SrcApe  = Aperture
          SrcObs  = Obscratn

        end if

        OK = PASS
      end subroutine src_size


      !------------------------------------------------------------------------------------------------------------
      ! Purpose  : Retrieve active source size, i.e., Beam Extend (Aperture) & Beam Obscuration
      !
      !     Call     : CALL get_src_size(OK,SrcApe,SrcObs)
      !     Output   : OK      [1x1,B]: = (True,1) if successful; (False,0) otherwise
      !                SrcApe  [1x1,D]: if |zSource|<=1e10 (Pt. Src.) ==> Ape. Beam N.A.
      !                                  |zSource| >1e10 (Col.Src.) ==> Apt. Beam Diameter (BaseUnits)
      !              SrcObs  [1x1,D]: if |zSource|<=1e10 (Pt. Src.) ==> Obs. Beam N.A.
      !                                  |zSource| >1e10 (Col.Src.) ==> Obs. Beam Diameter (BaseUnits)
      !     Require  : check if Rx is loaded

      !------------------------------------------------------------------------------------------------------------
      subroutine get_src_size(OK, SrcApe, SrcObs)
        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: SrcApe,SrcObs
        !f2py  intent(hide):: OK, SrcApe, SrcObs

        ! ------------------------------------------------------
        OK     = FAIL
        SrcApe = 0e0_pr
        SrcObs = 0e0_pr

        if (.not. SystemCheck()) return

        SrcApe  = Aperture
        SrcObs  = Obscratn
        OK      = PASS

      end subroutine get_src_size


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Define active source size, i.e., Aperture & Obscuration
      !
      !              Aperture: if Pt. Src. => N.A. of beam
      !                        if Col.Src. => Beam Diameter in BaseUnits
      !              Obscratn: if Pt. Src. => N.A. of beam
      !                        if Col.Src. => Beam Diameter in BaseUnits
      !
      ! Call     : CALL set_src_size(OK,SrcApe,SrcObs)
      ! Input    : SrcApe  [1x1,D]: Ape. Beam N.A. for Pt.Src.;otherwise, Apt. Beam Diameter (BaseUnits)
      !            SrcObs  [1x1,D]: Obs. Beam N.A. for Pt.Src.;otherwise, Obs. Beam Diameter (BaseUnits)
      ! Output   : OK      [1x1,B]: = (True,1) if successful; (False,0) otherwise
      ! Require  : check if Rx is loaded
      ! Note     : -- require SrcApe>0, SrcApe>SrcObs, SrcObs>=0
      !---------------------------------------------------------------------------------------------
      subroutine set_src_size(OK,SrcApe,SrcObs)
        !use,intrinsic :: ieee_arithmetic  -- not working with gcc to use with ieee_is_nan(), ieee_is_finite(x)

        implicit none
        logical, intent(out):: OK
        real(8), intent(in) :: SrcApe,SrcObs
        !f2py   intent(hide):: OK
        ! ------------------------------------------------------
        ! initialisation
        OK = FAIL

        ! Rx Loaded?
        if (.not. SystemCheck()) return

        ! input check
        if (isnan(SrcApe).or.isnan(SrcObs)) return                                           ! NaN   check (non standard)
        if (.not.(abs(SrcApe)<=huge(SrcApe)) .or. .not.(abs(SrcObs)<=huge(SrcObs))) return   ! inf   check
        if ((SrcApe<=0e0_pr).or.(SrcApe<=SrcObs).or.(SrcObs<0e0_pr)) return                  ! range check

        ! set values
        Aperture = SrcApe
        Obscratn = SrcObs

        ! return
        OK = PASS

      end subroutine set_src_size


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Return if Source is Finite
      ! Call     : CALL src_finite(OK, IsPtSrc)
      ! Output   : OK       [1x1,B]: = (True,1) if successful; (False,0) otherwise
      !            IsPtSrc) [1x1,B]: = (True,1) if zSource <= zSourceMax
      ! Require  : Rx loaded
      !---------------------------------------------------------------------------------------------
      subroutine src_finite(OK, IsPtSrc)
        use sourcsub_mod, only: isPointSource

        implicit none
        logical, intent(out):: OK
        logical, intent(out):: IsPtSrc
        !f2py  intent(hide):: OK, IsPtSrc
        ! ------------------------------------------------------

        if (SystemCheck()) then

          OK = PASS
          if (isPointSource()) then
            IsPtSrc = PASS
          else
            IsPtSrc = FAIL
          end if

        else

          OK      = FAIL  ! .False.
          IsPtSrc = FAIL  ! .False.

        end if

      end subroutine src_finite


    !=============================================================================================
    !
    ! Element Properties
    !
    !---------------------------------------------------------------------------------------------
    ! [ ] Group
    !      [x] elt_grp_max_all    Get max. Number of Srfs. defined in Element-Grp. over all Elements.
    !      [x] elt_grp_max        Get max. Number of Srfs defined in Element-Grp. at def. Element(s).
    !      [x] elt_grp_any        Get if at least one Element-Grp. is defined in Rx
    !      [x] elt_grp_fnd        Determine if a Element-Grps. are defined at specified Element(s).
    !      [x] elt_grp_del        Remove Element-Grp. Settings at specified Element(s)
    !      [x] elt_grp_del_all    Wipes all Element-Grp. Settings from Rx
    !      [x] elt_grp_set        Set Element Grp. Definitions
    !      [x] elt_grp_get        Get Element Grp. Definitions
    !=============================================================================================

      !-------------------------------------------------------------------------------------------
      ! Determine max. Number of Srfs. defined in Element-Grp. over all Elements.
      ! => no system check
      ! => intention is to support data exchange
      !-------------------------------------------------------------------------------------------
      subroutine elt_grp_max_all(maxGrpSize)

        implicit none
        integer, intent(out):: maxGrpSize ! max. defined Grp. Size
        ! ------------------------------------------------------

        maxGrpSize = maxval(EltGrp(0, :nElt))

      end subroutine elt_grp_max_all


      !-------------------------------------------------------------------------------------------
      ! Determine max. Number of Srfs defined in Element-Grp. at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_grp_max(ok,maxGrpSize,iElt,N)

        implicit none
        logical,               intent(out):: ok         ! success (1) or Fail (0)
        integer,               intent(out):: maxGrpSize ! max. defined Grp. Size
        integer, dimension(N), intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)

        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok         = FAIL
        maxGrpSize = 0

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! find Grp. Members at Element(s) iElt
        maxGrpSize = maxval(EltGrp(0, iElt(:)))

        ok = PASS

      end subroutine elt_grp_max


      !-------------------------------------------------------------------------------------------
      ! Determine if at least one Element-Grp. is defined in Rx
      !-------------------------------------------------------------------------------------------
      subroutine elt_grp_any(any_elt_grp)

        implicit none
        logical, intent(out):: any_elt_grp
        ! ------------------------------------------------------
        any_elt_grp = FAIL

        ! SMACOS and Rx status & nElt > 0
        if (.not. (SystemCheck() .and. nElt>0)) return

        ! check all surfaces if Elt. Grp. is defined
        if (maxval(EltGrp(0, :nElt)) > 0) any_elt_grp = PASS

      end subroutine elt_grp_any


      !-------------------------------------------------------------------------------------------
      ! Determine if a Element-Grps. are defined at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_grp_fnd(ok,nGrp,iElt,N)

        implicit none
        logical,                     intent(out):: ok    ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: nGrp  ! 1 (Grp defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt  ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N
        !f2py  integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok      = FAIL
        nGrp(:) = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! find Grp. Members at Element(s) iElt
        where (EltGrp(0, iElt(:)) > 0) nGrp(iElt(:)) = PASS
        ok = PASS

      end subroutine elt_grp_fnd


      !-------------------------------------------------------------------------------------------
      ! Delete Element-Grp. at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_grp_del(ok,iElt,N)

        implicit none
        logical,                     intent(out):: ok     ! (True,1) if successful; (False,0) otherwise
        integer, dimension(N),       intent(in) :: iElt   ! Elt. ID ( -nElt < iElt[i] <= nElt )

        integer,                     intent(in) :: N      ! # of Elements in iElt
        !f2py  integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! reset Grp. Members at Element(s) iElt
        EltGrp(:,iElt) = 0
        ok = PASS

      end subroutine elt_grp_del


      !-------------------------------------------------------------------------------------------
      ! Remove Element-Grp. Perturbation Settings on ALL Elements.
      !-------------------------------------------------------------------------------------------
      subroutine elt_grp_del_all(ok)

        implicit none
        logical, intent(out):: ok  ! (True,1) if successful; (False,0) otherwise
        ! ------------------------------------------------------
        ok = FAIL

        ! SMACOS and Rx status
        if (.not. SystemCheck()) return

        ! remove Grp. Members on ALL Elements
        EltGrp = 0
        ok = PASS

      end subroutine elt_grp_del_all


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Define Element-Grp. Perturbation Members at specified Element(s).
      !---------------------------------------------------------------------------------------------
      subroutine elt_grp_set(ok,iElt,jEltGrp,nEltGrp)

        implicit none
        logical,                     intent(out):: ok       ! (True,1) if successful; (False,0) otherwise
        integer,                     intent(in) :: iElt     ! Elt. ID  (1 <= iElt[i]    <= nElt )
        integer, dimension(nEltGrp), intent(in) :: jEltGrp  ! Grp. Members where (0 <= jEltGrp[i] <= nElt )

        integer,                     intent(in) :: nEltGrp  ! # of Grp. Members (= length(jEltGrp) <= mElt)
        !f2py  integer intent(hide), depend(jEltGrp):: nEltGrp=len(jEltGrp)
        ! ------------------------------------------------------
        ok = FAIL

        ! input check:
        if ((nEltGrp > mElt .or. nEltGrp < 1) .or. &   ! Range: 1 <= nEltGrp <= mElt    (defined in elt_mod.F)
            (.not. EltRangeChk(jEltGrp,0))    .or. &   ! Range: 0 <= jEltGrp(j) <= nElt
            (iElt<0) .or. (iElt>nElt)         .or. &   ! Range: 1 <= iElt       <= nElt
            (.not. SystemCheck()))                 &   ! SMACOS and Rx status
           return

        ! set Grp. Members at Element iElt
        EltGrp(        :,iElt) = 0           ! erase previous settings
        EltGrp(        0,iElt) = nEltGrp
        EltGrp(1:nEltGrp,iElt) = jEltGrp

        ok = PASS

      end subroutine elt_grp_set


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Retrieve Element-Grp. Perturbation Settings at specified Element(s).
      !---------------------------------------------------------------------------------------------
      subroutine elt_grp_get(ok,jEltGrp,nEltGrp,iElt,N,mElt_)

        implicit none
        logical,                     intent(out):: ok       ! (True,1) if successful; (False,0) otherwise
        integer, dimension(mElt_,N), intent(out):: jEltGrp  ! Grp. Members where K = mElt (value set to -1 for not used)
        integer, dimension(N),       intent(out):: nEltGrp  ! # of Grp. Members defined in jEltGrp
        integer, dimension(N),       intent(in) :: iElt     ! Elt. ID  (1 <= iElt[i]    <= nElt )

        integer,                     intent(in) :: N        ! # of Elements where to retrieve Grp. IDs
        integer,                     intent(in) :: mElt_    ! Max. # of Elements permitted (for Python = mElt (SMACOS var))
        !f2py  integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok           = FAIL
        jEltGrp(:,:) = -1       ! identifier for value not in use
        nEltGrp(:)   =  0

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! get Grp. Members at Element(s) iElt
        nEltGrp(:) = EltGrp(0,iElt)
        do concurrent (integer::K=1:N, nEltGrp(K)/=0)
          jEltGrp(1:nEltGrp(K),K) = EltGrp(1:nEltGrp(K),iElt(K))
        end do

        ok = PASS

      end subroutine elt_grp_get



    !=============================================================================================
    !
    ! Element Surface Properties
    !
    !---------------------------------------------------------------------------------------------
    ! [ ] Pose
    !     [x] elt_vpt     : set/get Element Vertex   Point
    !     [x] elt_rpt     : set/get Element Rotation Point
    !     [x] elt_psi     : set/get Element Surface Normal
    !
    ! [ ] Base Srf. Shape
    !     [x] elt_kc      : set/get Element Conic Constant
    !     [x] elt_kr      : set/get Element Base Radius
    !
    ! [ ] Material
    !     [ ] IndRef_     : set/get Refractive Index
    !     [ ] Glass_      : set/get Material Specification and read data from Glass Tbl.
    !     [ ] GlassModel_ : set/get Material Specification based on Glass Properties
    !
    ! [ ] Local CSYS (TElt)
    !     [ ] elt_csys_set
    !     [ ] elt_csys_get
    !     [ ] elt_csys_rm
    !
    ! [Srf. Shape] elt_srf_csys_set   : set Srf. Coordinate Frame
    ! [Srf. Shape] getEltGridInfo    : get Grid Srf. Settings
    ! [Srf. Shape] setEltGrid        : set element surface grid data
    !  [Pos/Shape] xp_set             : set XP parameters (Kr, Psi, Vpt, Rpt & zElt)
    !=============================================================================================

      ! ------------------------------------------------------------------------
      ! set/get Vertex Positions of defined Elements
      ! ------------------------------------------------------------------------
      subroutine elt_vpt(ok, iElt, Vpt, setter, n)

        implicit none
        logical,                  intent(out)  :: ok        ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n),   intent(in)   :: iElt      ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(3,n), intent(inout):: Vpt       ! Vertex Position [3 x N]
        logical,                  intent(in)   :: setter    ! =PASS for set; =FAIL for get

        integer,                  intent(in)   :: n         ! # of Elements
        !f2py integer intent(hide), depend(iElt,Vpt), check(len(iElt)==shape(Vpt,1), shape(Vpt,0)==3) :: n=len(iElt)
        ! ------------------------------------------------------
        ok  = FAIL
        if (setter == FAIL) Vpt = 0e0_pr

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        if (setter == PASS) then
          VptElt(:,iElt) = Vpt(:,:)
        else
          Vpt(:,:) = VptElt(:,iElt)
        end if

        ok = PASS

      end subroutine elt_vpt


      ! ------------------------------------------------------------------------
      ! set/get Vertex Surface Normals of defined Elements
      ! ------------------------------------------------------------------------
      subroutine elt_psi(ok, iElt, Psi, setter, n)
        use Constants, only: EPS

        implicit none
        logical,                  intent(out)  :: ok        ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n),   intent(in)   :: iElt      ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(3,n), intent(inout):: Psi       ! Surface Normal @ VptElt
        logical,                  intent(in)   :: setter    ! =PASS for set; =FAIL for get

        integer,                  intent(in)   :: n         ! # of Elements
        !f2py integer intent(hide), depend(iElt,Psi), check(len(iElt)==shape(Psi,1), shape(Psi,0)==3) :: n=len(iElt)

        real(8), dimension(n) :: norm_val
        ! ------------------------------------------------------
        ok  = FAIL
        if (setter == FAIL) Psi = 0e0_pr

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        if (setter == PASS) then

          ! ensure Psi magnitudes are non-zero
          norm_val = norm2(Psi, dim=1)
          if (any(norm_val<=EPS)) return

          ! Set PsiElt values
          do concurrent (integer::j=1:n)
            PsiElt(:,iElt(j)) = Psi(:,j)/norm_val(j)
          end do

        else
          Psi(:,:) = PsiElt(:,iElt)
        end if

        ok = PASS
      end subroutine elt_psi


      ! ------------------------------------------------------------------------
      ! set/get Rotation Positions of defined Elements
      ! ------------------------------------------------------------------------
      subroutine elt_rpt(ok, iElt, Rpt, setter, n)

        implicit none
        logical,                  intent(out)  :: ok        ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n),   intent(in)   :: iElt      ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(3,n), intent(inout):: Rpt       ! Rotation Position [3 x N]
        logical,                  intent(in)   :: setter    ! =PASS for set; =FAIL for get

        integer,                  intent(in)   :: n         ! # of Elements
        !f2py integer intent(hide), depend(iElt,Rpt), check(len(iElt)==shape(Rpt,1), shape(Rpt,0)==3) :: n=len(iElt)
        ! ------------------------------------------------------
        ok  = FAIL
        if (setter == FAIL) Rpt = 0e0_pr

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        if (setter == PASS) then
          RptElt(:,iElt) = Rpt(:,:)
        else
          Rpt(:,:) = RptElt(:,iElt)
        end if

        ok = PASS

      end subroutine elt_rpt


      !---------------------------------------------------------------------------------------------
      ! Set/Get element Conic Constant for element(s) defined in vector iElt
      !---------------------------------------------------------------------------------------------
      subroutine elt_kc(ok, iElt, Kc, setter, n)

        implicit none
        logical,                intent(out)  :: ok       ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n), intent(in)   :: iElt     ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(n), intent(inout):: Kc       ! Base Radii of elements
        logical,                intent(in)   :: setter   ! if PASS, set; otherwise, return values

        integer,                intent(in)   :: n        ! # of Elements
        !f2py  integer, intent(hide), depend(iElt,Kc), check(len(iElt)==len(Kc)) :: n=len(iElt)
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) Kc(:) = 0d0

        ! SMACOS and Rx status & range chk: 0 < iElt(i,j) <= nElt
        if (.not. StatusChk1(iElt)) return

        if (setter==PASS) then
          KcElt(iElt(:)) = Kc(:)
        else
          Kc(:) = KcElt(iElt(:))
        end if

        ok = PASS
      end subroutine elt_kc


      !---------------------------------------------------------------------------------------------
      ! Set/Get element Base Radius/Radii for element(s) defined in vector iElt
      !---------------------------------------------------------------------------------------------
      subroutine elt_kr(ok, iElt, Kr, setter, n)

        implicit none
        logical,                intent(out)  :: ok       ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n), intent(in)   :: iElt     ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(n), intent(inout):: Kr       ! Base Radii of elements
        logical,                intent(in)   :: setter   ! if PASS, set; otherwise, return values

        integer,                intent(in)   :: n        ! # of Elements
        !f2py  integer, intent(hide), depend(iElt,Kr), check(len(iElt)==len(Kr)) :: n=len(iElt)
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) Kr(:) = 0d0

        ! SMACOS and Rx status & range chk: 0 < iElt(i,j) <= nElt
        if (.not. StatusChk1(iElt)) return

        if (setter==PASS) then
          KrElt(iElt(:)) = Kr(:)
        else
          Kr(:) = KrElt(iElt(:))
        end if

        ok = PASS
      end subroutine elt_kr


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Set local element coordinate frame (TElt). If not active, it will activate it.
      !
      ! Call     : CALL elt_csys_set(ok,iElt,xDir,yDir,zDir,Upd,M)
      ! Input    : iElt   [Mx1,I]: Elt.ID   (Range: -nElt < iElt[j] <= nElt )
      !                            Identical elements to be defined simultaneously,
      !                            e.g., iElt = [1;-5]  => Element [1;nElt-5] have same TElt
      !            xDir   [3x1,D]: = [Lx,Ly,Lz] => x-axis expressed in GCF
      !            yDir   [3x1,D]: = [Mx,My,Mz] => y-axis expressed in GCF
      !            zDir   [3x1,D]: = [Nx,Ny,Nz] => z-axis expressed in GCF
      !            Upd    [1x1,I]: if set (=1), update TElt with element perturbations
      !            M      [1x1,I]: Number of equal elements to be updated
      ! Output   : ok     [1x1,B]: = (True) if successful; (False) otherwise
      ! Require  : check if pymacos is initialized & Rx loaded
      ! Note     : -- M of iElt defines identical elements
      !            -- GCF => Global Coordinate Frame
      !            -- will orthonormalize if not: zDir <= cross(xDir,yDir)
      !                                           yDir <= cross(zDir,xDir)
      !---------------------------------------------------------------------------------------------
      subroutine elt_csys_set(ok,iElt,xDir,yDir,zDir,Upd,m)
        use Constants, only: EPS
        use math_mod,  only: dorthoganalize

        implicit none
        logical,                intent(out):: ok
        integer,  dimension(m), intent(in) :: iElt
        real(8),  dimension(3), intent(in) :: xDir,yDir,zDir
        integer,                intent(in) :: Upd
        integer,                intent(in) :: m
        !f2py  integer intent(hide), depend(iElt), check(len(iElt)>0), check(len(xDir)==3, len(yDir)==3, len(zDir)==3) :: m=len(iElt)

        integer :: j
        real(8) :: A(3,3)
        logical :: updTElt
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(i) <= nElt
        ! if (.not. StatusChk1(iElt) .or. any(iElt>(nElt-3))) return
        if (.not. StatusChk1(iElt)) return

        ! check for null-vector
        A(:,1) = xDir
        A(:,2) = yDir
        A(:,3) = zDir
        if (any(norm2(A,DIM=1)<=EPS)) return

        ! if CF is not orthonormal, orthonormalize it
        if (abs(-A(3,1)*A(2,2) + A(2,1)*A(3,2) - A(1,3))>EPS .or. &
            abs( A(3,1)*A(1,2) - A(1,1)*A(3,2) - A(2,3))>EPS .or. &
            abs(-A(2,1)*A(1,2) + A(1,1)*A(2,2) - A(3,3))>EPS)     &
            call dorthoganalize(A(:,1),A(:,2),A(:,3))

        ! Set Position values (loop over identical elements)
        TElt(:,:,iElt) = 0e0_pr
        do concurrent(j=1:m)
            TElt(1:3,1:3,iElt(j)) = A
            TElt(4:6,4:6,iElt(j)) = A
        end do

        ! update TElt flag
        nECoord(iElt) = 6

        ! Update flag for modifying TElt with perturbations
        LUpdateTElt_FLG(iElt) = (Upd == PASS)

        ! return
        ok = PASS

      end subroutine elt_csys_set


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Returns TElt of local element-coordinate-frame if set
      !
      ! Call     : CALL elt_csys_get(ok,TElt,Upd,iElt,N)
      ! Input    : iElt      [1xN,I]: Elt.ID   (Range: 0 < iElt[j] <= nElt )
      !            N         [1x1,I]: Number of elements from which to retrieve
      ! Output   : ok        [1x1,B]: = (True) if successful; (False) otherwise
      !            TElt    [6x6xN,D]: = Local-coordinate-frame matrix
      !            csys_lcs    [1xN,L]: set (=1,true), if not nECoord/=0 or -6
      !            csys_upd  [1xN,L]: set (=1,true), TElt is updated with element perturbations
      ! Require  : check if pymacos is initialized & Rx loaded
      ! Note     : -- will check if Element is correct Srf. Type
      !            -- GCF => Global Coordinate Frame
      !---------------------------------------------------------------------------------------------
      subroutine elt_csys_get(ok, iElt, csys, csys_lcs, csys_upd, N)

        implicit none
        logical,                   intent(out)  :: ok
        integer, dimension(N),     intent(in)   :: iElt
        real(8), dimension(6,6,N), intent(inout):: csys
        logical, dimension(N),     intent(inout):: csys_lcs    ! Local CS defined if True
        logical, dimension(N),     intent(inout):: csys_upd
        integer,                   intent(in)   :: N
        !   integer intent(hide), depend(iElt,csys_lcs,csys_upd), check(len(iElt)==len(csys_lcs), len(iElt)==len(csys_upd), len(iElt)==shape(csys,2)):: N=len(iElt)
        !   integer intent(hide), depend(iElt), check(len(iElt)==len(csys_lcs), len(iElt)==len(csys_upd), len(iElt)==shape(csys,2)):: N=len(iElt)
        !f2py   integer intent(hide), depend(iElt):: N=len(iElt)

        integer :: j
        ! ------------------------------------------------------
        ok          = FAIL
        csys(:,:,:) = 0e0_pr
        csys_lcs    = FAIL
        csys_upd    = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! query information
        csys = TElt(:, :, iElt)
        where (LUpdateTElt_FLG(iElt)) csys_upd = PASS
        where (nECoord(iElt).EQ.6)    csys_lcs = PASS

        ! completed
        ok = PASS

      end subroutine elt_csys_get


      !---------------------------------------------------------------------------------------------
      ! Removes Local Coordinate System (LCS) Definitions on the defined surfaces
      !---------------------------------------------------------------------------------------------
      subroutine elt_csys_rm(ok,iElt,N)

        implicit none
        logical,               intent(out):: ok
        integer, dimension(N), intent(in) :: iElt    ! Surfaces where to remove LCS
        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt) :: N=len(iElt)
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! remove LCS
        TElt(:,:,iElt) = 0e0_pr          ! delete LCS infos and define matrix as eye(6)
        do concurrent (integer::j=1:6)
          TElt(j,j,iElt) = 1e0_pr
        end do

        LUpdateTElt_FLG(iElt) = .false.  ! set values to default (update LCS with perturbations)
        nECoord(iElt)         = -6       ! set tag defining that no LCS is defined

        ! return
        ok = PASS

      end subroutine elt_csys_rm


      ! ---------------------------------------------------------------------------------------------
      ! Set element surface coordinate frame for elements with Surface type where identical
      ! elements can be defined simultaneously,
      ! ---------------------------------------------------------------------------------------------
      subroutine elt_srf_csys(ok, pMon_, xMon_, yMon_, zMon_, iElt, setter, N)
        use Constants, only: EPS
        use  math_mod, only: dorthoganalize

        implicit none
        logical,                  intent(out)  :: ok
        real(8),  dimension(3,N), intent(inout):: pMon_,xMon_,yMon_,zMon_
        integer,  dimension(N),   intent(in)   :: iElt
        integer,                  intent(in)   :: setter
        integer,                  intent(in)   :: N
        !f2py  integer intent(hide),depend(iElt) :: N=len(iElt)

        integer :: j
        real(8) :: A(3,3)
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) then
          pMon_ = 0e0_pr
          xMon_ = 0e0_pr
          yMon_ = 0e0_pr
          zMon_ = 0e0_pr
        end if

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if EltType require a Srf. Coord. Frame
        if (.not.any( SrfType(iElt) == SrfType_RequireSrfCSYS)) return

        ! if (any( SrfType(iElt(:)) /= SrfType_Monomial .and. &    ! Monomial
        !          SrfType(iElt(:)) /= SrfType_Zernike  .and. &    ! Zernike
        !          SrfType(iElt(:)) /= SrfType_GridData)) return   ! GridData

        if (setter == PASS) then

          ! check for null-vector
          if ((any(norm2(xMon_, DIM=1)<=EPS)) .or. &
              (any(norm2(yMon_, DIM=1)<=EPS)) .or. &
              (any(norm2(zMon_, DIM=1)<=EPS))) return

          do concurrent (j=1:N)

            A(:,1) = xMon_(:,j)
            A(:,2) = yMon_(:,j)
            A(:,3) = zMon_(:,j)

            ! if CF is not orthonormal, orthonormalize it
            !
            ! cross(Vx,Vy) = [-a22*a31 + a21*a32,  a12*a31 - a11*a32, -a12*a21 + a11*a22]
            ! cross(Vy,Vz) = [-a23*a32 + a22*a33,  a13*a32 - a12*a33, -a13*a22 + a12*a23]
            ! cross(Vz,Vx) = [ a23*a31 - a21*a33, -a13*a31 + a11*a33,  a13*a21 - a11*a23]
            !
            ! check:  | cross(xMon,yMon) - zMon | < EPS
            if (abs(-A(3,1)*A(2,2) + A(2,1)*A(3,2) - A(1,3))>EPS .or. &
                abs( A(3,1)*A(1,2) - A(1,1)*A(3,2) - A(2,3))>EPS .or. &
                abs(-A(2,1)*A(1,2) + A(1,1)*A(2,2) - A(3,3))>EPS)     &
              call dorthoganalize(A(:,1),A(:,2),A(:,3))

            ! Set Position values (loop over identical elements)
            xMon(:,iElt(j)) = A(:,1)
            yMon(:,iElt(j)) = A(:,2)
            zMon(:,iElt(j)) = A(:,3)
            pMon(:,iElt(j)) = pMon_(:,j)

          end do

        else

          xMon_(:,:N) = xMon(:,iElt)
          yMon_(:,:N) = yMon(:,iElt)
          zMon_(:,:N) = zMon(:,iElt)
          pMon_(:,:N) = pMon(:,iElt)

        end if

        ok = PASS

      end subroutine elt_srf_csys


      ! ---------------------------------------------------------------------------------------------
      ! Set element surface coordinate frame position for elements with Surface type where identical
      ! elements can be defined simultaneously,
      ! ---------------------------------------------------------------------------------------------
      subroutine elt_srf_csys_pos(ok, pMon_, iElt, setter, N)
        use Constants, only: EPS

        implicit none
        logical,                  intent(out)  :: ok
        real(8),  dimension(3,N), intent(inout):: pMon_
        integer,  dimension(N),   intent(in)   :: iElt
        integer,                  intent(in)   :: setter
        integer,                  intent(in)   :: N
        !f2py integer intent(hide),depend(iElt) :: N=len(iElt)    ! check(shape(pMon_,0)==3, shape(pMon_,1)==len(iElt))
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) pMon_ = 0e0_pr

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if EltType =  Monomial (#4), Zernike (#8), GridData (#9), ...
        if (.not.any(SrfType(iElt) == SrfType_RequireSrfCSYS)) return

        ! check shape: 3 x N
        if (size(pMon_,1) /= 3 .or. size(pMon_,2) /= N) return

        ! set/get
        if (setter == PASS) then
          pMon(:,iElt) = pMon_(:,:)
        else
          pMon_(:,:) = pMon(:,iElt)
        end if

        ok = PASS

      end subroutine elt_srf_csys_pos


      ! ---------------------------------------------------------------------------------------------
      ! Set element surface coordinate frame orientation for elements with Surface type where identical
      ! elements can be defined simultaneously,
      ! ---------------------------------------------------------------------------------------------
      subroutine elt_srf_csys_dir(ok, xMon_, yMon_, zMon_, iElt, setter, N)
        use Constants, only: EPS
        use  math_mod, only: dorthoganalize

        implicit none
        logical,                  intent(out)  :: ok
        real(8),  dimension(3,N), intent(inout):: xMon_,yMon_,zMon_
        integer,  dimension(N),   intent(in)   :: iElt
        integer,                  intent(in)   :: setter
        integer,                  intent(in)   :: N
        !f2py  integer intent(hide),depend(iElt) :: N=len(iElt)

        integer :: j
        real(8) :: A(3,3)
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) then
          xMon_ = 0e0_pr
          yMon_ = 0e0_pr
          zMon_ = 0e0_pr
        end if

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if EltType require a Srf. Coord. Frame
        if (.not.any( SrfType(iElt) == SrfType_RequireSrfCSYS)) return

        ! check shape: 3 x N
        if (size(xMon_,1) /= 3 .or. size(xMon_,2) /= N .or.  &
            size(yMon_,1) /= 3 .or. size(yMon_,2) /= N .or.  &
            size(zMon_,1) /= 3 .or. size(zMon_,2) /= N) return

        if (setter == PASS) then

          ! check for null-vector
          if ((any(norm2(xMon_, DIM=1)<=EPS)) .or. &
              (any(norm2(yMon_, DIM=1)<=EPS)) .or. &
              (any(norm2(zMon_, DIM=1)<=EPS))) return

          do concurrent (j=1:N)
            A(:,1) = xMon_(:,j)
            A(:,2) = yMon_(:,j)
            A(:,3) = zMon_(:,j)

            ! if CSYS is not orthonormal, orthonormalize it
            !
            ! cross(Vx,Vy) = [-a22*a31 + a21*a32,  a12*a31 - a11*a32, -a12*a21 + a11*a22]
            ! cross(Vy,Vz) = [-a23*a32 + a22*a33,  a13*a32 - a12*a33, -a13*a22 + a12*a23]
            ! cross(Vz,Vx) = [ a23*a31 - a21*a33, -a13*a31 + a11*a33,  a13*a21 - a11*a23]
            !
            ! check:  | cross(xMon,yMon) - zMon | < EPS
            if (abs(-A(3,1)*A(2,2) + A(2,1)*A(3,2) - A(1,3))>EPS .or. &
                abs( A(3,1)*A(1,2) - A(1,1)*A(3,2) - A(2,3))>EPS .or. &
                abs(-A(2,1)*A(1,2) + A(1,1)*A(2,2) - A(3,3))>EPS)     &
              call dorthoganalize(A(:,1),A(:,2),A(:,3))

            ! assign
            xMon(:,iElt(j)) = A(:,1)
            yMon(:,iElt(j)) = A(:,2)
            zMon(:,iElt(j)) = A(:,3)
          end do

        else

            xMon_(:,:N) = xMon(:,iElt)
            yMon_(:,:N) = yMon(:,iElt)
            zMon_(:,:N) = zMon(:,iElt)

        end if

        ok = PASS

      end subroutine elt_srf_csys_dir


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Set element surface coordinate frame for elements with Surface type.
      !            Identical elements to be defined simultaneously,
      !               e.g., iElt = [1;-5]  => Element [1;nElt-5] have the same grid data
      !
      ! Call     : CALL elt_srf_csys_set(ok,iElt,pMon,xMon,yMon,zMon,M)
      ! Input    : iElt   [Mx1,I]: Elt.ID   (Range: -nElt < iElt[j] <= nElt )
      !            pMon   [3x1,D]: = [x,y,z]    => origin of grid coord. frame (GCF)
      !            xMon   [3x1,D]: = [Lx,Ly,Lz] => x-axis of grid coord. frame (GCF)
      !            yMon   [3x1,D]: = [Mx,My,Mz] => y-axis of grid coord. frame (GCF)
      !            zMon   [3x1,D]: = [Nx,Ny,Nz] => z-axis of grid coord. frame (GCF)
      !            M      [1x1,I]: Number of equal elements to be updated
      ! Output   : ok     [1x1,B]: = (True) if successful; (False) otherwise
      ! Require  : check if pymacos is initialized & Rx loaded
      ! Note     : -- M of iElt defines identical elements
      !            -- will check if Element is Grid Srf.
      !            -- GCF => Global Coordinate Frame
      !            -- will orthonormalize if not: zMon <= cross(xMon,yMon)
      !                                           yMon <= cross(zMon,xMon)
      !---------------------------------------------------------------------------------------------
      subroutine elt_srf_csys_set(ok,iElt,pMon_,xMon_,yMon_,zMon_,m)
        use Constants, only: EPS
        use  math_mod, only: dorthoganalize

        implicit none
        logical,                intent(out):: ok
        integer,  dimension(m), intent(in) :: iElt
        real(8),  dimension(3), intent(in) :: pMon_,xMon_,yMon_,zMon_
        integer,                intent(in) :: m
        !f2py  integer intent(hide), depend(iElt) :: m=len(iElt)

        integer :: j
        real(8) :: A(3,3)
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(i) <= nElt
        ! if (.not. StatusChk1(iElt) .or. any(iElt>(nElt-3))) return
        if (.not. StatusChk1(iElt)) return

          ! check if EltType =  Monomial (#4), Zernike (#8), GridData (#9)
        if (any( SrfType(iElt(:)) /= SrfType_Monomial .and. &    ! Monomial
                  SrfType(iElt(:)) /= SrfType_Zernike  .and. &    ! Zernike
                  SrfType(iElt(:)) /= SrfType_GridData)) return   ! GridData

        ! check for null-vector
        A(:,1) = xMon_
        A(:,2) = yMon_
        A(:,3) = zMon_
        if (any(norm2(A,DIM=1)<=EPS)) return

        ! if CF is not orthonormal, orthonormalize it
        if (abs(-A(3,1)*A(2,2) + A(2,1)*A(3,2) - A(1,3))>EPS .or. &
            abs( A(3,1)*A(1,2) - A(1,1)*A(3,2) - A(2,3))>EPS .or. &
            abs(-A(2,1)*A(1,2) + A(1,1)*A(2,2) - A(3,3))>EPS)     &
            call dorthoganalize(A(:,1),A(:,2),A(:,3))

        ! Set Position values (loop over identical elements)
        do concurrent (j=1:m)
            xMon(:,iElt(j)) = A(:,1)
            yMon(:,iElt(j)) = A(:,2)
            zMon(:,iElt(j)) = A(:,3)
            pMon(:,iElt(j)) = pMon_
        end do

        ok = PASS

      end subroutine elt_srf_csys_set

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Returns element surface coordinate frame for elements where surface orientation
      !            definition is required, i.e., GridData, Zernike, etc.
      !
      !            Identical elements simultaneously retrieved, e.g.:
      !               iElt = [1,-5]   =>  get data from Elements [1,nElt-5]
      !
      ! Call     : CALL elt_srf_csys_get(ok,pMon,xMon,yMon,zMon,iElt,N)
      ! Input    : iElt   [1xN,I]: Elt.ID   (Range: 0 < iElt[j] <= nElt )
      !            N      [1x1,I]: Number of elements from which to retrieve
      ! Output   : ok     [1x1,B]: = (True) if successful; (False) otherwise
      !            pMon   [3xN,D]: = [x,y,z]    => origin of grid coord. frame (GCF)
      !            xMon   [3xN,D]: = [Lx,Ly,Lz] => x-axis of grid coord. frame (GCF)
      !            yMon   [3xN,D]: = [Mx,My,Mz] => y-axis of grid coord. frame (GCF)
      !            zMon   [3xN,D]: = [Nx,Ny,Nz] => z-axis of grid coord. frame (GCF)
      ! Require  : check if pymacos is initialized & Rx loaded
      ! Note     : -- will check if Element is correct Srf. Type
      !            -- GCF => Global Coordinate Frame
      !---------------------------------------------------------------------------------------------
      subroutine elt_srf_csys_get(ok,pMon_,xMon_,yMon_,zMon_,iElt,N)

        implicit none
        logical,                  intent(out):: ok
        real(8),  dimension(3,N), intent(out):: pMon_,xMon_,yMon_,zMon_
        integer,  dimension(N),   intent(in) :: iElt
        integer,                  intent(in) :: N
        !f2py  integer intent(hide),depend(iElt) :: N=len(iElt)  ! ,check(shape(pMon_,1)==shape(xMon_,1),shape(pMon_,1)==shape(yMon_,1),shape(pMon_,1)==shape(zMon_,1),len(iElt)==shape(pMon_,1))

        integer :: j
        ! ------------------------------------------------------
        ok    = FAIL

        pMon_ = 0e0_pr
        xMon_ = 0e0_pr
        yMon_ = 0e0_pr
        zMon_ = 0e0_pr

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if EltType =  Monomial (#4), Zernike (#8), GridData (#9)
        if (any( SrfType(iElt(:)) /= SrfType_Monomial .and. &    ! Monomial
                 SrfType(iElt(:)) /= SrfType_Zernike  .and. &    ! Zernike
                 SrfType(iElt(:)) /= SrfType_GridData)) return   ! GridData

        do concurrent (j=1:N)
            xMon_(:,j) = xMon(:,iElt(j))
            yMon_(:,j) = yMon(:,iElt(j))
            zMon_(:,j) = zMon(:,iElt(j))
            pMon_(:,j) = pMon(:,iElt(j))
        end do

        ! return
        ok = PASS

      end subroutine elt_srf_csys_get


    ! ============================================================================================
    !
    ! Element Surface Properties: Grid Type
    !
    ! ============================================================================================
    ! [x] elt_srf_grid_any        Determine if any Grid Srf. is defined in Rx
    ! [x] elt_srf_grid_fnd        Return Srf. IDs where a Grid Srf. is defined
    ! [x] elt_srf_grid_fnd_type   Return Srf. IDs where a Grid Srf. with def. Type is defined (AsGrData, GridData, ...)
    ! [x] elt_srf_grid_size       Return Grid Data Size at def. Surfaces
    ! [x] elt_srf_grid_size_max   Return Max. Permitted Grid Sampling (model dependent).
    ! [x] elt_srf_grid_spacing    Set / get Grid Sampling Spacing dx (dx==dy)
    ! [x] elt_srf_grid_data       Set / get Grid Data
    ! [x] elt_srf_grid_data_scale Scales    Grid Data
    ! [x] elt_srf_grid_data_add   Add       Grid Data to existing Grid Data Values

      !-------------------------------------------------------------------------------------------
      ! Determine if at least one Element has a Grid Srf. defined in Rx
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_any(any_elt)

        implicit none
        logical, intent(out):: any_elt   ! success (1) or Fail (0)
        ! ------------------------------------------------------
        any_elt = FAIL

        ! SMACOS and Rx status & nElt > 0
        if (.not. (SystemCheck() .and. nElt>0)) return

        ! check all surfaces if a Grid-Type Srf is defined
        do concurrent(integer::j=1:size(GridTypeAll))
          if (any(SrfType(:nElt) == GridTypeAll(j))) any_elt = PASS
        end do

      end subroutine elt_srf_grid_any


      !-------------------------------------------------------------------------------------------
      ! Determine if a Grid Srf. is/are defined at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_fnd(ok, IsGridSrf, iElt, N)

        implicit none
        logical,                     intent(out):: ok         ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: IsGridSrf  ! 1 (Grp defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N
        !f2py   integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok           = FAIL
        IsGridSrf(:) = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! find Grid Srfs. at Element(s) iElt
        do concurrent(integer::j=1:size(GridTypeAll))
          where ((SrfType(iElt) == GridTypeAll(j))) IsGridSrf = PASS
        end do
        ok = PASS

      end subroutine elt_srf_grid_fnd


      !-------------------------------------------------------------------------------------------
      ! Determine if a Grid Srf. with specific Type is/are defined at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_fnd_type(ok, IsGridSrf, iElt, GridTypeID, N)
        implicit none
        logical,                     intent(out):: ok         ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: IsGridSrf  ! 1 (defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)
        integer,                     intent(in) :: GridTypeID ! Specify SyrfaceType of Type Grid

        integer,                     intent(in) :: N
        !f2py   integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok           = FAIL
        IsGridSrf(:) = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! chk if ID is a valid Grid Srf.
        if (all(GridTypeID /= GridTypeAll)) return

        ! find Grid Srfs. at Element(s) iElt
        where (SrfType(iElt) == GridTypeID) IsGridSrf(iElt) = PASS
        ok = PASS

      end subroutine elt_srf_grid_fnd_type


      !-------------------------------------------------------------------------------------------
      ! Get Max. Permitted Grid Sampling (model dependent).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_size_max(MaxGridSize)

        implicit none
        integer, intent(out):: MaxGridSize ! Max Sampling Grid Size
        ! ------------------------------------------------------

        MaxGridSize = mGridMat

      end subroutine elt_srf_grid_size_max


      !-------------------------------------------------------------------------------------------
      ! Get Grid Sampling at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_size(ok,GridSize, iElt, N)

        implicit none
        logical,               intent(out):: ok         ! success (1) or Fail (0)
        integer, dimension(N), intent(out):: GridSize   ! Grid Value (= -1 Not defined at Srf.)
        integer, dimension(N), intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)

        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt):: N=len(iElt)
        ! ------------------------------------------------------
        ok = FAIL
        GridSize(:) = -1

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! extract data for each Grid Srf. Type
        do concurrent (integer::j=1:size(GridTypeAll))
          where (SrfType(iElt) == GridTypeAll(j)) GridSize = nGridMat(iElt)
        end do
        ok = PASS

      end subroutine elt_srf_grid_size


      !-------------------------------------------------------------------------------------------
      ! Set/Get Grid Srf. Sampling at specified Grid Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_spacing(ok,iElt,GridSrfdx_,setter,N)

        implicit none
        logical,               intent(out)   :: ok           ! success (1) or Fail (0)
        integer, dimension(N), intent(in)    :: iElt         ! Elt ID: (0 < iElt[i] <= nElt)
        real(8), dimension(N), intent(inout) :: GridSrfdx_   ! grid data sampling spacing dx==dy
        logical,               intent(in)    :: setter       ! if PASS, define new values

        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt,GridSrfdx_):: N=len(iElt)

        logical, dimension(N) :: chk
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) GridSrfdx_ = 0d0

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if SrfType /= Grid
        chk(:) = .false.
        do concurrent (integer::j=1:size(GridTypeAll))
          where (SrfType(iElt) == GridTypeAll(j)) chk = .true.
        end do
        if (any(.not. chk)) return

        ! set/get GridSrfdx
        if (setter==PASS) then
          GridSrfdx(iElt) = GridSrfdx_
        else
          GridSrfdx_ = GridSrfdx(iElt)
        end if
        ok = PASS

      end subroutine elt_srf_grid_spacing


      !-------------------------------------------------------------------------------------------
      ! Scale Surface Grid for specified Grid Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_data_scale(ok,iElt,scalar,N)

        implicit none
        logical,               intent(out):: ok       ! success (1) or Fail (0)
        integer, dimension(N), intent(in) :: iElt     ! Elt ID: (0 < iElt[i] <= nElt)
        real(8), dimension(N), intent(in) :: scalar   ! grid scaling factor

        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt,GridSrfdx_):: N=len(iElt)

        logical, dimension(N) :: chk
        integer :: npts
        ! ------------------------------------------------------
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if SrfType /= Grid
        chk(:) = .false.
        do concurrent (integer::j=1:size(GridTypeAll))
          where (SrfType(iElt) == GridTypeAll(j)) chk = .true.
        end do
        if (any(.not. chk)) return

        ! scale grid data
        do concurrent(integer::j=1:SIZE(iElt))
          npts = nGridMat(iElt(j))
          associate (pgrid=>GridMat(1:npts, 1:npts, iEltToGridSrf(iElt(j))))
            pgrid = scalar(j)*pgrid
          end associate
        end do
        ok = PASS

      end subroutine elt_srf_grid_data_scale


      !-------------------------------------------------------------------------------------------
      ! Set/Get Grid Srf. Sampling at specified Grid Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_data(ok,iElt,GridSrfdx_,GridMat_,setter,Nx,Ny)

        implicit none
        logical,                   intent(out)   :: ok           ! success (1) or Fail (0)
        integer,                   intent(in)    :: iElt         ! Elt ID: (0 < iElt[i] <= nElt)
        real(8),                   intent(inout) :: GridSrfdx_   ! grid data sampling spacing dx==dy
        real(8), dimension(Ny,Nx), intent(inout) :: GridMat_     ! displacement at node points from nominal shape (N x N) Grid
        logical,                   intent(in)    :: setter       ! if PASS, define new values

        integer,                   intent(in) :: Nx,Ny
        !f2py  integer intent(hide), depend(GridMat_), check(shape(GridMat_,0)>0, shape(GridMat_,1)>0):: Ny=shape(GridMat_,0), Nx=shape(GridMat_,1)
        ! ------------------------------------------------------
        ok = FAIL
        if (setter==FAIL) then
          GridSrfdx_    = 0d0
          GridMat_(:,:) = 0d0
        end if

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if ((.not. SystemCheck()) .or. (iElt<1) .or. (iElt>nElt)) return
        ! if (.not. StatusChk1(iElt)) return

        ! out-of-boundary & shape check (mGridMat defined in macos_param.txt)
        if ((Nx > mGridMat) .or. (Nx/=Ny) .or. (Nx<3)) return

        ! check if SrfType /= Grid
        if (all(SrfType(iElt) /= GridTypeAll)) return

        ! set/get
        if (setter==PASS) then
                                 GridSrfdx(iElt) = GridSrfdx_
                                 nGridMat(iElt)  = Nx
          GridMat(:,:,iEltToGridSrf(iElt))       = 0d0            ! reset first (just in case)
          GridMat(1:Ny,1:Nx,iEltToGridSrf(iElt)) = GridMat_(:,:)  !Transpose(GridMat_(Ny:-1:1,:))
        else
          GridSrfdx_    = GridSrfdx(iElt)
          GridMat_(:,:) = GridMat(1:Ny,1:Nx,iEltToGridSrf(iElt))
        end if

        ok = PASS

      end subroutine elt_srf_grid_data


      !-------------------------------------------------------------------------------------------
      ! Add Grid Srf. Sampling to Existing Values at specified Grid Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_grid_data_add(ok,iElt,GridMat_,Nx,Ny)

        implicit none
        logical,                   intent(out)   :: ok           ! success (1) or Fail (0)
        integer,                   intent(in)    :: iElt         ! Elt ID: (0 < iElt[i] <= nElt)
        real(8), dimension(Ny,Nx), intent(inout) :: GridMat_     ! displacement at node points from nominal shape (N x N) Grid

        integer,                   intent(in)    :: Nx,Ny
        !f2py  integer intent(hide), depend(GridMat_), check(shape(GridMat_,0)>0, shape(GridMat_,1)>0):: Ny=shape(GridMat_,0), Nx=shape(GridMat_,1)
        ! ------------------------------------------------------
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if ((.not. SystemCheck()) .or. (iElt<1) .or. (iElt>nElt)) return

        ! check if SrfType /= Grid
        if (all(SrfType(iElt) /= GridTypeAll)) return

        ! out-of-boundary & shape check (mGridMat defined in macos_param.txt)
        if ((Nx /= nGridMat(iElt)) .or. (Nx/=Ny) .or. (Nx<3)) return

        ! add
        associate (pgrid=>GridMat(1:Ny,1:Nx,iEltToGridSrf(iElt)))
          pgrid = pgrid + GridMat_(:,:)
        end associate
        ! GridMat(1:Ny,1:Nx,iEltToGridSrf(iElt)) = GridMat(1:Ny,1:Nx,iEltToGridSrf(iElt)) + GridMat_(:,:)  !Transpose(GridMat_(Ny:-1:1,:))
        ok = PASS

      end subroutine elt_srf_grid_data_add


    ! ============================================================================================
    !
    ! Element Surface Properties: Reflective & Transmission Grating
    !
    ! ============================================================================================
    ! [x] elt_srf_grating_any        =>  elt_grating_any       Checks if any Grating Srfs. are defined in Rx
    ! [x] elt_srf_grating_fnd        =>  elt_grating_fnd       Find all elements with Grating Srfs. types
    ! [x] elt_srf_grating_params     =>  elt_grating_params    Linear grating (h1HOE, RuleWidth, Transmission or Reflective)
    ! [x] elt_srf_grating_type       =>  elt_grating_type      Transmission or Reflective Grating
    ! [x] elt_srf_grating_order      =>  elt_grating_order     Linear Grating Order (OrderHOE)
    ! [x] elt_srf_grating_rule_width =>  elt_grating_spacing   RuleWidth
    ! [x] elt_srf_grating_rule_dir   =>  elt_grating_h1HOE     h1HOE vector perpendicular to the ruling dir and psiElt vector.
    ! --------------------------------------------------------------------------------------------

    ! --------------------------------------------------------------------------------------------
    ! Determine if at least one Srf has a Grating defined on Srf.
    ! --------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_any(Any_Elt_With_Grating)

      implicit none
      logical, intent(out):: Any_Elt_With_Grating
      ! ------------------------------------------------------
      Any_Elt_With_Grating = FAIL

      ! SMACOS and Rx status & nElt > 0
      if (.not. (SystemCheck() .and. nElt>0)) return

      ! print *, "=============>", GratingElt, TrGratingElt
      ! print *, "===> EltType: ", EltType(:nElt)
      ! print *, "===> SrfType: ", SrfType(:nElt)
      ! print *, "===>   EltID: ", EltID(:nElt)
      ! print *, "===>   EltID: ", EltID(1:nElt)
      ! check all surfaces if a Zernike Srf is defined
      if (any((EltID(1:nElt) == GratingElt).or. &
              (EltID(1:nElt) == TrGratingElt))) Any_Elt_With_Grating = PASS

    end subroutine elt_srf_grating_any


    !-------------------------------------------------------------------------------------------
    ! Determine if a Grating is defined over a Srf at specified Element(s).
    !-------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_fnd(ok, Grating, iElt, N)

      implicit none
      logical,               intent(out):: ok       ! success (1) or Fail (0)
      integer, dimension(N), intent(out):: Grating  ! 0) No, (1) Refl.  (2) Trans.
      integer, dimension(N), intent(in) :: iElt     ! Elt ID: (0 < iElt[i] <= nElt)

      integer,               intent(in) :: N
      !f2py   integer intent(hide), depend(iElt):: N=len(iElt)

      ! ------------------------------------------------------
      ok         = FAIL
      Grating(:) = FAIL

      ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
      if (.not. StatusChk1(iElt)) return

      ! find Zernike Srfs. at Element(s) iElt
      where (EltID(iElt) == GratingElt)   Grating(iElt) = 1  ! Reflective   Grating
      where (EltID(iElt) == TrGratingElt) Grating(iElt) = 2  ! Transmission Grating
      ok = PASS

    end subroutine elt_srf_grating_fnd


    !-------------------------------------------------------------------------------------------
    ! Set/Get Grating Rule Width
    !-------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_rule_width(ok, iElt, Spacing, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      real(8), intent(inout) :: Spacing    ! RuleWidth (0 < Spacing < Inf)
      logical, intent(in)    :: setter     ! if PASS, define new values
      ! ------------------------------------------------------
      ok = FAIL

      ! SMACOS and Rx status &
      if ((.not. SystemCheck())      .or. &  ! SMACOS and Rx status
          (iElt<1) .or. (iElt>nElt)       &  ! range chk: 0 < iElt <= nElt
         ) return

      ! check if valid Srf.
      if (.not.((EltID(iElt) == GratingElt).or. &
                (EltID(iElt) == TrGratingElt))) return

      ! set/get values
      if (setter == PASS) then
        RuleWidth(iElt) = Spacing
      else
        Spacing = RuleWidth(iElt)
      end if

      ! return
      ok = PASS

    end subroutine elt_srf_grating_rule_width


    !-------------------------------------------------------------------------------------------
    ! Set/Get Grating Order
    !-------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_order(ok, iElt, Order, setter, N)
      logical, intent(out)                :: ok         ! success status (0/1)
      integer, dimension(N), intent(in)   :: iElt       ! Surface 0 <= iElt <= nElt
      integer, dimension(N), intent(inout):: Order      ! Diffraction Order
      logical,               intent(in)   :: setter     ! if PASS, define new values

      integer,               intent(in)   :: N          ! # of Elements
      !f2py integer intent(hide), depend(iElt,Order), check(len(iElt)==len(Order)) :: N=len(iElt)
      logical :: chk(N)

      ! ------------------------------------------------------
      ok = FAIL

      ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
      if (.not. StatusChk1(iElt(:))) return

      ! all listed surfaces must be Gratings
      if (.not.all((EltID(iElt) == GratingElt).or.(EltID(iElt) == TrGratingElt))) return

      ! set/get values
      if (setter == PASS) then
        OrderHOE(iElt(:)) = Order(:)
      else
        Order(:) = OrderHOE(iElt(:))
      end if

      ! return
      ok = PASS

    end subroutine elt_srf_grating_order


    !-------------------------------------------------------------------------------------------
    ! Set/Get Grating Type
    !-------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_type(ok, iElt, reflective, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      logical, intent(inout) :: reflective ! (1) Refl. (0) Trans.
      logical, intent(in)    :: setter     ! if PASS, define new values
      ! ------------------------------------------------------
      ok = FAIL

      ! SMACOS and Rx status &
      if ((.not. SystemCheck())      .or. &  ! SMACOS and Rx status
          (iElt<1) .or. (iElt>nElt)       &  ! range chk: 0 < iElt <= nElt
         ) return

      ! check if valid Srf.
      if (.not.((EltID(iElt) == GratingElt).or. &
                (EltID(iElt) == TrGratingElt))) return

      ! set/get values
      if (setter == PASS) then

        if (reflective==PASS) then
          EltID(iElt) = GratingElt    ! Reflective Grating
        else
          EltID(iElt) = TrGratingElt  ! Transmissive Grating
        end if

      else

        if (EltID(iElt) == GratingElt) then
          reflective = PASS           ! Reflective Grating
        else
          reflective = FAIL           ! Transmissive Grating
        end if

      end if

      ! return
      ok = PASS

    end subroutine elt_srf_grating_type


    !-------------------------------------------------------------------------------------------
    ! Set/Get Grating h1HOE -- defines ruling direction (vector perpend. to its direction and PsiElt)
    !-------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_rule_dir(ok, iElt, h1HOE_, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      real(8), intent(inout) :: h1HOE_(3)  ! h1HOE => perpendicular to the ruling direction and psiElt vector.
      logical, intent(in)    :: setter     ! if PASS, define new values
      ! ------------------------------------------------------
      ok = FAIL

      ! SMACOS and Rx status &
      if ((.not. SystemCheck())      .or. &  ! SMACOS and Rx status
          (iElt<1) .or. (iElt>nElt)       &  ! range chk: 0 < iElt <= nElt
         ) return

      ! check if valid Srf.
      if (.not.((EltID(iElt) == GratingElt).or. &
                (EltID(iElt) == TrGratingElt))) return

      ! set/get values
      if (setter == PASS) then
        h1HOE(:, iElt) = h1HOE_(:)
      else
        h1HOE_(:) = h1HOE(:, iElt)
      end if

      ! return
      ok = PASS

    end subroutine elt_srf_grating_rule_dir


    !-------------------------------------------------------------------------------------------
    ! Set/Get Grating Parameters
    !-------------------------------------------------------------------------------------------
    subroutine elt_srf_grating_params(ok, iElt, Spacing, Diff_Order, h1HOE_, reflective, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      integer, intent(inout) :: Diff_Order ! Diffraction Order
      real(8), intent(inout) :: Spacing    ! RuleWidth (0 < Spacing < Inf)
      real(8), intent(inout) :: h1HOE_(3)  ! h1HOE => perpendicular to the ruling direction and psiElt vector.
      logical, intent(inout) :: reflective ! (1) Refl. (0) Trans.
      logical, intent(in)    :: setter     ! if PASS, define new values
      ! ------------------------------------------------------
      ok = FAIL
      if (.not.setter) then
        Diff_Order = -999
        Spacing    = -999d0
        h1HOE_(:)  = -999d0
        reflective = -1
      end if

      ! SMACOS and Rx status &
      if ((.not. SystemCheck())      .or. &  ! SMACOS and Rx status
          (iElt<1) .or. (iElt>nElt)       &  ! range chk: 0 < iElt <= nElt
         ) return

      ! check if valid Srf.
      if (.not.((EltID(iElt) == GratingElt).or. &
                (EltID(iElt) == TrGratingElt))) return

      ! set/get values
      if (setter == PASS) then

        if (reflective==PASS) then
          EltID(iElt) = GratingElt    ! Reflective Grating
        else
          EltID(iElt) = TrGratingElt  ! Transmissive Grating
        end if

        OrderHOE(iElt)  = real(Diff_Order, kind=8) ! Diffraction Order
        RuleWidth(iElt) = Spacing                  ! Rule Width Lin. Grating
        h1HOE(:, iElt)  = h1HOE_(:)                ! orientation vector

      else

        if (EltID(iElt) == GratingElt) then
          reflective = PASS                    ! Reflective Grating
        else
          reflective = FAIL                    ! Transmissive Grating
        end if

        Diff_Order = int(OrderHOE(iElt)) ! Diffraction Order
        Spacing    = RuleWidth(iElt)     ! Rule Width Lin. Grating
        h1HOE_(:)  = h1HOE(:, iElt)      ! orientation vector

      end if

      ! return
      ok = PASS

    end subroutine elt_srf_grating_params



    ! ============================================================================================
    !
    ! Element Surface Properties: Zernike
    !
    ! ============================================================================================
    ! [ ] elt_srf_zrn_any
    ! [ ] elt_srf_zrn_fnd
    ! [ ] elt_srf_zrn_coef
    ! [ ] elt_srf_zrn_type
    ! [ ] elt_srf_zrn_norm_radius
    !
    ! ToDo: simplify
    !   [ ] elt_srf_zrn_set         Define Zernike Srf. params of equal Zernike Srf. element(s)
    !   [ ] elt_srf_zrn_get         Return Zernike Srf. params of equal Zernike Srf. element(s)
    !   [ ] elt_srf_zrn_mode_set    Define Zernike Srf. params of equal Zernike Srf. element(s)
    ! ============================================================================================

      !-------------------------------------------------------------------------------------------
      ! Determine if at least one Element has a Zernike defined in Rx
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_any(any_elt_Zrn)

        implicit none
        logical, intent(out):: any_elt_Zrn
        ! ------------------------------------------------------
        any_elt_Zrn = FAIL

        ! SMACOS and Rx status & nElt > 0
        if (.not. (SystemCheck() .and. nElt>0)) return

        ! check all surfaces if a Zernike Srf is defined
        if (any(SrfType(:nElt) == SrfType_Zernike)) any_elt_Zrn = PASS

      end subroutine elt_srf_zrn_any


      !-------------------------------------------------------------------------------------------
      ! Determine if a Zernike are defined at specified Element(s).
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_fnd(ok, nZrn, iElt, N)

        implicit none
        logical,                     intent(out):: ok    ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: nZrn  ! 1 (Grp defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt  ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N
        !f2py   integer intent(hide), depend(iElt):: N=len(iElt)

        ! ------------------------------------------------------
        ok      = FAIL
        nZrn(:) = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt(j) <= nElt
        if (.not. StatusChk1(iElt)) return

        ! find Zernike Srfs. at Element(s) iElt
        where (SrfType(iElt) == SrfType_Zernike) nZrn(iElt) = PASS
        ok = PASS

      end subroutine elt_srf_zrn_fnd


      !ToDo: check max mode depending on Zernike Type

      !-------------------------------------------------------------------------------------------
      ! Set/get Zernike Coefficients
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_coef(ok, iElt, ZernMode, ZernCoef_, setter, reset, N)
        use elt_mod,   only: mZernModes

        implicit none
        logical,               intent(out)   :: ok
        integer,               intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
        integer, dimension(N), intent(inout) :: ZernMode   ! [Z_1,Z_2,...,Z_Nm] Zernike Modes
        real(8), dimension(N), intent(inout) :: ZernCoef_  ! [C_1,C_2,...,C_Nc] Zernike Coefficients
        logical,               intent(in)    :: setter     ! if PASS, define new values
        logical,               intent(in)    :: reset      ! if PASS, reset all modes first (only for setter)
        integer,               intent(in)    :: N
        !f2py   integer intent(hide), depend(ZernMode,ZernCoef_), check(len(ZernMode)==len(ZernCoef_), len(ZernMode)>0) :: N=len(ZernMode)

        integer            :: j
        integer, parameter :: mZernCoef = mZernModes             ! hardcoded value taken from module "elt_mod"
        ! ------------------------------------------------------
        ! Note: - tracesub.F: in CTRACE(...)
        !              ZernTypeL = {1,4} => ZernToMon1    Norm/ Malacara
        !                        = {2,5} => ZernToMon2    Norm/ Born & Wolf   => calls ZernToMon1
        !                        = {3,6} => ZernToMon3    Norm/ Fringe        => calls ZernToMon1
        !                        = {  7} => ZernToMon4    NormHex
        !                        = {  8} => ZernToMon6    NormNoll            => calls ZernToMon1
        !                        = {  9} => ZernToMon7    NormAnnularNoll
        !       - surfsub.F
        !              "ZernToMon1" => ZernType = {4,5,6,8} uses var. "ZnNmFac_m" defined via "SetZernNormCoef"

        ! call SetZernNormCoef(0)  ! just in case (def. in elt_mod.F)
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if ((.not. SystemCheck()) .or. (iElt<1) .or. (iElt>nElt)) return
        ! if (.not. StatusChk1(iElt)) return

        ! check if SrfType /= Zernike
        if (SrfType(iElt) /= SrfType_Zernike) return

        ! check mode range: Fringe is limited to 37 Modes
        if (((ZernTypeL(iElt) == ZernType_Fringe) .or.       &
            (ZernTypeL(iElt) == ZernType_NormFringe)) .and. &
              (any(ZernMode > 37))) then
          return
        else
          if (any(ZernMode< 1 .or. ZernMode>mZernCoef)) return
        end if

        if (setter == PASS) then
          if (reset == PASS) ZernCoef(:, iElt) = 0d0
          ZernCoef(ZernMode(:), iElt) = ZernCoef_(:)
        else
          ZernCoef_(:) = ZernCoef(ZernMode(:), iElt)
        end if

        ! return
        ok = PASS

      end subroutine elt_srf_zrn_coef


      !-------------------------------------------------------------------------------------------
      ! Set/get Zernike Type on a Surface
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_type(ok, iElt, ZernType, setter, reset)
        use elt_mod,   only: mZernType

        implicit none
        logical, intent(out)   :: ok         ! success status (0/1)
        integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
        integer, intent(inout) :: ZernType   ! Zernike Type
        logical, intent(in)    :: setter     ! if PASS, define new values
        logical, intent(in)    :: reset      ! if PASS, reset all coefs first (only for setter)

        ! ------------------------------------------------------
        ! Note: - tracesub.F: in CTRACE(...)
        !
        !          ZernTypeL = {1,4} => ZernToMon1    Norm/ ANSI
        !                    = {2,5} => ZernToMon2    Norm/ Born & Wolf   => calls ZernToMon1
        !                    = {3,6} => ZernToMon3    Norm/ Fringe        => calls ZernToMon1
        !                    = {  7} => ZernToMon4    NormHex
        !                    = {  8} => ZernToMon6    NormNoll            => calls ZernToMon1
        !                    = {  9} => ZernToMon7    NormAnnularNoll
        !       - surfsub.F
        !              "ZernToMon1" => ZernType = {4,5,6,8} uses var. "ZnNmFac_m" defined via "SetZernNormCoef"
        ! ------------------------------------------------------

        ! initialisation
        ok = FAIL

        ! SMACOS and Rx status &
        if ((.not. SystemCheck())      .or. &  ! SMACOS and Rx status
            (iElt<1) .or. (iElt>nElt)       &  ! range chk: 0 < iElt <= nElt
            ) return

        if (setter == PASS) then

          if ((ZernType < 1) .or. (ZernType > mZernType) .or. & ! Check Range
              (SrfType(iElt) /= SrfType_Zernike)) return         ! SrfType /= Zernike

          if (reset == PASS) ZernCoef(:, iElt) = 0d0
          ZernTypeL(iElt) = ZernType

        else

          if (SrfType(iElt) /= SrfType_Zernike) then   ! check if SrfType /= Zernike
            ZernType = -1
          else
            ZernType = ZernTypeL(iElt)
          end if
        end if

        ! return
        ok = PASS

      end subroutine elt_srf_zrn_type


      !-------------------------------------------------------------------------------------------
      ! Set/Get Zernike Norm. Radius
      !-------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_norm_radius(ok, iElt, NormRad, setter)
        logical, intent(out)   :: ok         ! success status (0/1)
        integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
        real(8), intent(inout) :: NormRad    ! Zernike Norm. Radius (0 < lMon < Inf)
        logical, intent(in)    :: setter     ! if PASS, define new values
        ! ------------------------------------------------------
        ok = FAIL

        ! SMACOS and Rx status &
        if ((.not. SystemCheck())      .or. &  ! SMACOS and Rx status
            (iElt<1) .or. (iElt>nElt)       &  ! range chk: 0 < iElt <= nElt
           ) return

        ! set/get values
        if (setter == PASS) then
          if ((SrfType(iElt) /= SrfType_Zernike)) return     ! SrfType /= Zernike

          lMon(iElt) = NormRad

        else

          if (SrfType(iElt) /= SrfType_Zernike) then   ! check if SrfType /= Zernike
            NormRad = -1d0
          else
            NormRad = lMon(iElt)
          end if

        end if

        ! return
        ok = PASS

      end subroutine elt_srf_zrn_norm_radius


      !-------------------------------------------------------------------------------------------
      ! Purpose  : Define Zernike Srf. settings of equal Zernike Srf. element(s)
      ! Call     : CALL elt_srf_zrn_set(ok,iElt,lMon,ZernType,ZernMode,ZernCoef,ZernAnnularRatio,M)
      ! Input    : iElt             [Mx1,I]: Elt. ID ( -nElt < iElt[i] <= nElt )
      !            lMon             [1x1,D]: = Zernike Radius
      !            ZernType         [1x1,I]: 1) Malacara  4) NormMalacara  7) NormHex
      !                                      2) BornWolf  5) NormBornWolf  8) NormNoll
      !                                      3) Fringe    6) NormFringe    9) NormAnnularNoll
      !            ZernMode        [1xNm,I]: = [Z_1,Z_2,...,Z_Nm] Zernike Modes
      !            ZernCoef        [1xNc,D]: = [C_1,C_2,...,C_Nc] Zernike Coefficients
      !            ZernAnnularRatio [1x1,D]: = inner/outer radius ratio (0,...,1)
      !                                        (only for ZernType = NormAnnularNoll(#9))
      !            M                [1x1,I]: Number of equal elements
      !            Nm               [1x1,I]: Number of elements in "ZernMode"
      !            Nc               [1x1,I]: Number of elements in "ZernCoef"
      ! Output   : ok               [1x1,B]: = (True,1) if successful; (False,0) otherwise
      ! Require  : PyMACOS initialized & Rx loaded
      ! Note     : -- ZCF => Zernike Coord. Frame: defined via calling elt_srf_csys_set(...)
      !            -- M of iElt defines identical elements, e.g., iElt = [1;-5]
      !               => Element [1;nElt-5] have the same Zernike settings:
      !---------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_set(ok,iElt,lMon_,ZernType,ZernMode,ZernCoef_,ZernAnnularRatio,M,Nm,Nc)
        !use Constants, only: eps
        use elt_mod,   only: mZernType, mZernModes

        implicit none
        logical,                   intent(out):: ok
        integer, dimension(M),     intent(in) :: iElt
        real(8),                   intent(in) :: lMon_
        integer,                   intent(in) :: ZernType
        integer, dimension(Nm),    intent(in) :: ZernMode
        real(8), dimension(Nc),    intent(in) :: ZernCoef_
        real(8),                   intent(in) :: ZernAnnularRatio
        integer,                   intent(in) :: M, Nm, Nc
        !f2py   integer intent(hide), depend(iElt),      check(len(iElt)>0)     :: M=len(iElt)
        !f2py   integer intent(hide), depend(ZernMode),  check(len(ZernMode)>0) :: Nm=len(ZernMode)
        !f2py   integer intent(hide), depend(ZernCoef_), check(len(ZernCoef_)>0):: Nc=len(ZernCoef_)

        integer :: j
        ! ------------------------------------------------------
        ! Note: - tracesub.F: in CTRACE(...)
        !              ZernTypeL = {1,4} => ZernToMon1    Norm/ Malacara
        !                        = {2,5} => ZernToMon2    Norm/ Born & Wolf   => calls ZernToMon1
        !                        = {3,6} => ZernToMon3    Norm/ Fringe        => calls ZernToMon1
        !                        = {  7} => ZernToMon4    NormHex
        !                        = {  8} => ZernToMon6    NormNoll            => calls ZernToMon1
        !                        = {  9} => ZernToMon7    NormAnnularNoll
        !       - surfsub.F
        !              "ZernToMon1" => ZernType = {4,5,6,8} uses var. "ZnNmFac_m" defined via "SetZernNormCoef"

        ! call SetZernNormCoef(0)  ! just in case (def. in elt_mod.F)
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! out-of-boundary & shape check BUT not yet checked: multiple occurances of the same modes (issue with parallelisation)
        if (ZernType<1 .or. ZernType>mZernType .or. Nc/=Nm .or. &
            any(ZernMode<0 .or. ZernMode>mZernModes) .or. lMon_ <= 0e0_pr .or. &
            (ZernAnnularRatio < 0e0_pr .or. ZernAnnularRatio > 1e0_pr)) return

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if EltType /= Zernike
        if (any(SrfType(iElt(:)) /= SrfType_Zernike)) return

        ! set Zernike values

          ! Zernike Modes & corresponding coefficients (incl. Normalisation factor updates)
          ZernCoef(:,iElt(:)) = 0e0_pr
          do concurrent (j=1:M)
            ZernCoef(ZernMode(:),iElt(j)) = ZernCoef_(:)
          end do

          ! Zernike Type
          ZernTypeL(iElt(:)) = ZernType

          ! Zernike Normalisation Radius
          lMon(iElt(:)) = lMon_

          ! Zernike Radii Ratio = inner/outer radius
          if (ZernType==ZernType_NormAnnularNoll) ZernAnnuRatio(iElt(:)) = ZernAnnularRatio

        ! set Srf. Coordinate Frame if not already set to default values (xMon,yMon,zMon,pMon)
        ! ==> done in switchEltSrfType(...)

        ! return
        ok = PASS

      end subroutine elt_srf_zrn_set


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Define Zernike Srf. settings of equal Zernike Srf. element(s)
      ! Call     : CALL elt_srf_zrn_mode_set(ok,iElt,lMon,ZernType,ZernMode,ZernCoef,ZernAnnularRatio,M)
      ! Input    : iElt             [Mx1,I]: Elt. ID ( -nElt < iElt[i] <= nElt )
      !            lMon             [1x1,D]: = Zernike Radius
      !            ZernType         [1x1,I]: 1) Malacara  4) NormMalacara  7) NormHex
      !                                      2) BornWolf  5) NormBornWolf  8) NormNoll
      !                                      3) Fringe    6) NormFringe    9) NormAnnularNoll
      !            ZernMode        [1xNm,I]: = [Z_1,Z_2,...,Z_Nm] Zernike Modes
      !            ZernCoef        [1xNc,D]: = [C_1,C_2,...,C_Nc] Zernike Coefficients
      !            ZernAnnularRatio [1x1,D]: = inner/outer radius ratio (0,...,1)
      !                                        (only for ZernType = NormAnnularNoll(#9))
      !            M                [1x1,I]: Number of equal elements
      !            Nm               [1x1,I]: Number of elements in "ZernMode"
      !            Nc               [1x1,I]: Number of elements in "ZernCoef"
      ! Output   : ok               [1x1,B]: = (True,1) if successful; (False,0) otherwise
      ! Require  : PyMACOS initialized & Rx loaded
      ! Note     : -- ZCF => Zernike Coord. Frame: defined via calling elt_srf_csys_set(...)
      !            -- M of iElt defines identical elements, e.g., iElt = [1;-5]
      !               => Element [1;nElt-5] have the same Zernike settings:
      !---------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_mode_set(ok,iElt,ZernMode,ZernCoef_,M,N)
        !use Constants, only: eps
        use elt_mod,   only: mZernType, mZernModes

        implicit none
        logical,               intent(out):: ok
        integer, dimension(M), intent(in) :: iElt
        integer, dimension(N), intent(in) :: ZernMode
        real(8), dimension(N), intent(in) :: ZernCoef_
        integer,               intent(in) :: M, N

        !f2py   integer intent(hide), depend(iElt), check(len(iElt)>0) :: M=len(iElt)
        !f2py   integer intent(hide), depend(ZernMode,ZernCoef_)       :: N=len(ZernMode)
        !f2py   check(len(ZernMode)>0, len(ZernMode)==len(ZernCoef_))

        ! ------------------------------------------------------
        ! Note: - tracesub.F: in CTRACE(...)
        !              ZernTypeL = {1,4} => ZernToMon1    Norm/ ANSI
        !                        = {2,5} => ZernToMon2    Norm/ Born & Wolf   => calls ZernToMon1
        !                        = {3,6} => ZernToMon3    Norm/ Fringe        => calls ZernToMon1
        !                        = {  7} => ZernToMon4    NormHex
        !                        = {  8} => ZernToMon6    NormNoll            => calls ZernToMon1
        !                        = {  9} => ZernToMon7    NormAnnularNoll
        !       - surfsub.F
        !              "ZernToMon1" => ZernType = {4,5,6,8} uses var. "ZnNmFac_m" defined via "SetZernNormCoef"

        ! call SetZernNormCoef(0)  ! just in case (def. in elt_mod.F)
        ! ------------------------------------------------------
        ! initialisation
        ok = FAIL

        ! out-of-boundary & shape check BUT not yet checked: multiple occurances of the same modes (issue with parallelisation)
        if (any(ZernMode<0 .or. ZernMode>mZernModes)) return

        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt)) return

        ! check if EltType /= Zernike
        if (any(SrfType(iElt(:)) /= 8)) return

        ! Zernike Modes & corresponding coefficients (incl. Normalisation factor updates)
        ZernCoef(:,iElt(:)) = 0e0_pr

        do concurrent (integer::j=1:M)
          ZernCoef(ZernMode(:),iElt(j)) = ZernCoef_(:)
        end do

        ! return
        ok = PASS

      end subroutine elt_srf_zrn_mode_set


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Retrieve Zernike Srf. settings of Zernike Srf. element(s)
      ! Call     : CALL elt_srf_zrn_get(ok,lMon,ZernType,ZernCoef,ZernAnnularRatio,iElt)
      ! Input    : iElt             [1xN,I]: Elt. ID ( -nElt < iElt[i] <= nElt )
      !            N                [1x1,I]: Number of elements to query
      ! Output   : ok               [1x1,B]: = (True,1) if successful; (False,0) otherwise
      !            lMon             [1xN,D]: = Zernike Radius
      !            ZernType         [1xN,I]: 1) Malacara  4) NormMalacara  7) NormHex
      !                                      2) BornWolf  5) NormBornWolf  8) NormNoll
      !                                      3) Fringe    6) NormFringe    9) NormAnnularNoll
      !            ZernCoef         [MxN,D]: = [C_1,C_2,...,C_N] Zernike Coefficients  (M=45)
      !            ZernAnnularRatio [1xN,D]: = inner/outer radius ratio (0,...,1)
      !                                        only important for ZernType = NormAnnularNoll (9)
      ! Require  : PyMACOS initialized & Rx loaded
      !---------------------------------------------------------------------------------------------
      subroutine elt_srf_zrn_get(ok,lMon_,ZernType,ZernCoef_,ZernAnnularRatio,iElt,N)
        use elt_mod, only: mZernModes
        implicit none
        integer, parameter :: mZernCoef = 66             ! hardcoded value taken from module "elt_mod" (issues with parameter from elt_mod)

        logical,                         intent(out):: ok
        real(8), dimension(N),           intent(out):: lMon_
        integer, dimension(N),           intent(out):: ZernType
        real(8), dimension(mZernCoef,N), intent(out):: ZernCoef_
        real(8), dimension(N),           intent(out):: ZernAnnularRatio
        integer, dimension(N),           intent(in) :: iElt
        integer,                         intent(in) :: N

        !f2py  integer intent(hide), depend(iElt,ZernAnnularRatio,ZernType,lMon_) :: N=len(iElt)
        !f2py  check(len(iElt)>0, len(iElt)==len(ZernAnnularRatio), len(iElt)==len(ZernType), len(iElt)==len(lMon_))
        !f2py  check(len(iElt)==shape(ZernCoef_,1))

        ! ------------------------------------------------------
        ! initialize in case of failure
        ok = FAIL

        lMon_(:)         = 0e0_pr
        ZernType(:)      = 0
        ZernCoef_(:,:)   = 0e0_pr
        ZernAnnularRatio = 0e0_pr

        !check shape of array


        ! SMACOS and Rx status & range chk: 0 < iElt <= nElt
        if (.not. StatusChk1(iElt(:))) return

        ! check if EltType /= Zernike
        if (any(SrfType(iElt(:)) /= SrfType_Zernike)) return

        ! get Zernike values
        if (SIZE(ZernCoef_, 1) /= mZernModes) return


        lMon_(:)       = lMon(iElt(:))                 ! Zernike Normalisation Radius
        ZernType(:)    = ZernTypeL(iElt(:))            ! Zernike Type
        ZernCoef_(:,:) = ZernCoef(:mZernModes,iElt(:)) ! coefficients    ToDo: check => (incl. Normalisation factor updates)

        ! Zernike Radii Ratio = inner/outer radius
        do concurrent (integer::j=1:N)
          if (ZernType(j)==ZernType_NormAnnularNoll) then
            ZernAnnularRatio(j) = ZernAnnuRatio(iElt(j))
          else
            ZernAnnularRatio(j) = 0e0_pr
          end if
        end do

        ! return
        ok = PASS

      end subroutine elt_srf_zrn_get


    ! ============================================================================================
    !
    ! Element Surface Properties: Asphere Type
    !
    ! ============================================================================================


    !=============================================================================================
    !
    ! Optical System Analysis / Settings / ...
    !
    !---------------------------------------------------------------------------------------------
    ! [ ] trace_rays
    ! [ ] ray_info_get
    ! [ ] ray_info_set
    ! [ ] opd_val
    ! [ ] xp_fnd / xp_set / xp_get
    ! [ ] stop_info_set / stop_info_get
    !
    !  getEFL
    !  traceChiefRay
    !=============================================================================================

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Retrieve Ray-Trace Data (Pos & Dir) from previous call to trace_rays(...)
      ! Call     : CALL ray_info_get(OK, Pos, Dir, RayOK, nRays)
      ! Input    : nRays (=N)[1x1,I]: Number of traced rays (defined by previous trace_rays(...) call)
      ! Output   : OK        [1x1,B]: = (True) if successful; (False) otherwise
      !            Pos       [3xN,D]: = [[x1,y1,z1],...] global position of ray-surface intersection point
      !            Dir       [3xN,D]: = [[L1,M1,N1],...] direction cosine of ray direction (before Srf)
      !            OPL       [1xN,D]: Optical Path Length from Src. Srf to last traced Srf. (trace_rays)
      !            RayOK     [1xN,L]: (True) if ray is successfully traced; (False) otherwise
      !            RayPass   [1xN,L]: (True) if ray is not blocked; (False) otherwise
      ! Require  : check if Rx loaded
      ! Note     :
      !---------------------------------------------------------------------------------------------
      subroutine ray_info_get(OK, Pos, Dir, OPL, RayOK, RayPass, nRays)

        implicit none
        logical,                     intent(out):: OK
        real(8), dimension(3,nRays), intent(out):: Pos,Dir
        real(8), dimension(nRays),   intent(out):: OPL
        logical, dimension(nRays),   intent(out):: RayOK      ! successfully traced
        logical, dimension(nRays),   intent(out):: RayPass    ! not blocked
        integer,                     intent(in) :: nRays

        !f2py  intent(hide)         :: OK
        !f2py  intent(out,hide,copy):: Pos,Dir,RayOK,OPL,RayPass
        ! ------------------------------------------------------
        OK      = FAIL
        RayOK   = FAIL       ! Ray Status: trace failures
        RayPass = FAIL       ! Ray Status: blocked
        Pos     = 0e0_pr
        Dir     = 0e0_pr

        ! Rx loaded and Ray Size check
        if ((.not. SystemCheck()).or.(nRay/=nRays).or.(nRays<=0)) return

        ! Obtain data from last trace
        Pos(:,:) = RayPos(:,:nRay)            ! intersection point with iElt's Srf.
        Dir(:,:) = RayDir(:,:nRay)            ! direction before Srf.
        OPL(:)   = CumRayL(:nRay)             ! OPL from Src. Srf. to last trace srf.
        where (LRayOK(:nRay))   RayOK=PASS    ! note: Fortran/python returs -1 (True) and 0 (False)
        where (LRayPass(:nRay)) RayPass=PASS  !       => ensures (1) for .TRUE.  and  (0) for .FALSE.

        ! return
        OK = PASS

      end subroutine ray_info_get

      !---------------------------------------------------------------------------------------------
      ! Purpose  :  Replace Ray-Trace Data at current trace location
      ! Call     : CALL ray_info_set(OK, nRays, Pos, Dir, RayOK)
      ! Input    : nRays (=K)[1x1,I]: Number of traced rays (defined by previous trace_rays(...) call)
      !            Pos       [3xK,D]: = [[x1,y1,z1],...] global position of ray-surface intersection point
      !            Dir       [3xK,D]: = [[L1,M1,N1],...] direction cosine of ray direction (before Srf)
      !            OPL       [1xK,D]: Optical Path Length from Src. Srf to last traced Srf.
      !            RayOK     [1xK,I]: (True) if ray is successfully traced; (False) otherwise
      ! Output   : OK        [1x1,B]: = (True) if successful; (False) otherwise
      ! Require  : check if Rx loaded
      ! Note     : K => # of rays to be traced  (== nRays)
      !            WARNING -- you really must understand how to use this functionality correctly
      !---------------------------------------------------------------------------------------------
      subroutine ray_info_set(OK, Pos, Dir, OPL, RayOK, nRays)
        use smacos_vars_mod, only: npts

        implicit none
        logical,                     intent(out):: OK
        integer,                     intent(in) :: nRays
        real(8), dimension(3,nRays), intent(in) :: Pos, Dir
        real(8), dimension(nRays),   intent(in) :: OPL
        integer, dimension(nRays),   intent(in) :: RayOK

        !f2py integer, intent(hide), depend(OPL) :: nRays=len(OPL)
        ! ------------------------------------------------------

        OK = FAIL
        ! Rx loaded and Ray Size check
        if ((.not. SystemCheck()).or. npts*npts<nRays) return

        ! Set data at last trace, assuming trace was performed
        nRay            = nRays
        RayPos(:,:nRay) = Pos(:,:)        ! intersection point with iElt's Srf.
        RayDir(:,:nRay) = Dir(:,:)        ! direction before Srf.
        CumRayL(:nRay)  = OPL(:)          ! OPL from Src. Srf. to last trace srf.
        LRayOK(:nRay)   = .true.
        where (RayOK==0) LRayOK(:nRay)=.false.

        ! return
        OK = PASS

      end subroutine ray_info_set

      !---------------------------------------------------------------------------------------------
      ! Purpose  : Trace Wavefront from source to surface iElt with grid sampling NxN
      ! Call     : CALL trace_rays(OK, WFE, nRays, N, iElt)
      ! Input    : iElt      [1x1,I]: Elt.ID      (Range: 0 < iElt <= nElt)
      ! Output   : OK        [1x1,B]: = (True) if successful; (False) otherwise
      !            rmsWFE    [1x1,D]: rms Wavefront Error
      !            nRays     [1x1,I]: Number of traced rays
      !            nGridPts  [1x1,I]: Wavefront sampling: nGridPts x nGridPts   ( N = nGridPts )
      ! Require  : check if Rx loaded
      ! Note     : --
      !---------------------------------------------------------------------------------------------
      subroutine trace_rays(OK, rms_WFE, nRays, N, iElt)

        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: rms_WFE
        integer, intent(out):: nRays, N
        integer, intent(in) :: iElt
        ! ------------------------------------------------------
        OK = FAIL

        ! Rx loaded and  0 < iElt <= nElt
        if ((.not. SystemCheck()).or.(iElt<1).or.(iElt>nElt)) return

        ! trace rays
        command = 'OPD'
        IARG(1) = iElt
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ! return
        nRays   = nRay
        N       = nGridPts
        rms_WFE = RMSWFE
        if (nRay==0) return   ! all rays are lost

        OK = PASS

      end subroutine trace_rays


      !---------------------------------------------------------------------------------------------
      ! retrieve Optical Path Difference Matrix
      !---------------------------------------------------------------------------------------------
      subroutine opd_val(OK, OPD, N)

        implicit none
        logical,                 intent(out):: OK
        real(8), dimension(N,N), intent(out):: OPD
        integer,                 intent(in) :: N      ! = nGridPts

        !f2py  intent(hide)         :: OK
        !f2py  intent(out,hide,copy):: OPD
        ! ------------------------------------------------------
        OK  = FAIL
        OPD = 0e0_pr

        ! Rx loaded and Ray Size check
        if ((.not. SystemCheck()).or.(nGridPts/=N)) return

        ! Obtain data from last trace
        OPD(:,:) = OPDMat(:N,:N)

        ! return
        OK = PASS

      end subroutine opd_val


      !---------------------------------------------------------------------------------------------
      ! Execute SPOT cmd
      !---------------------------------------------------------------------------------------------
      subroutine spot_cmd(OK, nSpot, iElt, ref_csys, ref_pos, res_trace)
        use smacos_vars_mod, only: iSpot
        implicit none
        logical, intent(out):: OK         ! (PASS=1) if successful; (FAIL=0) otherwise
        integer, intent(out):: nSpot      ! # of successfully traced rays for Spot
        integer, intent(in) :: iElt       ! Element at which spot to be determined
        integer, intent(in) :: ref_csys   ! (1) BEAM, (2) TOUT, (3) TElt
        logical, intent(in) :: ref_pos    ! (PASS=1) @ 'ELT' or (FAIL=0) @ "CHFRAY'
        logical, intent(in) :: res_trace  ! (PASS=1) apply a MODIFY cmd. first or (FAIL=0) not

        character(len=*), parameter :: str_ref_csys(3) = ['BEAM', 'TOUT', 'TELT']
        ! ------------------------------------------------------
        OK    = FAIL
        nSpot = 0

        if ((.not. SystemCheck()) .or.            &
            (iElt<1) .or. (iElt>nElt) .or.        &
            (EltID(iElt).EQ.SegmentElt).or.       &
            (EltID(iElt).EQ.NSRefractorElt) .or.  &
     	      (EltID(iElt).EQ.NSReflectorElt) .or.  &
            (ref_csys<1).or.(ref_csys>3)) return

        if (res_trace==PASS) then
          ! reset trace to start at source
          command = 'MODIFY'
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)
        end if

        ! 'SPCENTER': Set spot centering option: ELT or ChfRay
        command = 'SPC'
        if (ref_pos==PASS) then
          CARG(1) = 'ELT'
        else
          CARG(1) = 'CHFRAY'
        end if
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ! Spot
        command = 'SPOT'
        CARG(1) = str_ref_csys(ref_csys)   ! CSYS: Beam, TOUT, or TELT  (not working: ENTER (user))
        IARG(1) = iElt
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ! Success
        nSpot = iSpot            ! note: 0 <= nSpot <= nRay
        if (nSpot>0) OK = PASS

      end subroutine spot_cmd


      !---------------------------------------------------------------------------------------------
      ! Obtain SPOT Data
      !---------------------------------------------------------------------------------------------
      ! Note:
      !      if SPCENTER == 'ChfRay' => CntrSpot
      !                  == 'Elt'    => RefSpot
      !
      !      ChfRay Location <= CntrSpot-RefSpot
      !
      !   where
      !      RefSpot(1)=DDOTC(xLocal,VptElt(1,iEndElt))
      !      RefSpot(2)=DDOTC(yLocal,VptElt(1,iEndElt))
      !
      !      IF (spcOption.EQ.1) THEN                   ! SPCENTER: ELT
      !        RaySpot(iSpot,1)=xnom(3)-RefSpot(1)
      !        RaySpot(iSpot,2)=xnom(4)-RefSpot(2)
      !      ELSE                                       ! SPCENTER: ChfRay
      !        RaySpot(iSpot,1)=xnom(3)-CntrSpot(1)
      !        RaySpot(iSpot,2)=xnom(4)-CntrSpot(2)
      !---------------------------------------------------------------------------------------------
      subroutine spot_get(OK, SPOT, shift, centroid, csys, N)
        use smacos_vars_mod, only: CntrSpot, RefSpot, xLocal,yLocal,zLocal  ! xcent, ycent

        implicit none
        logical,                 intent(out):: OK           ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), dimension(N,2), intent(out):: SPOT         ! ray-surface intersection points
        real(8), dimension(4),   intent(out):: shift        ! shift from projected Ref. position
        real(8), dimension(2),   intent(out):: centroid     ! centroid position
        real(8), dimension(3,3), intent(out):: csys         ! Coord. Frame orientation

        integer,                 intent(in) :: N            ! = nSpot = successfully traced rays
        !f2py intent(out,hide,copy):: SPOT
        ! ------------------------------------------------------
        if (.not. SystemCheck() .or. (N>nRay) .or. (N<1)) then

          OK          = FAIL
          SPOT(:,:)   = 0e0_pr
          centroid(:) = 0e0_pr
          shift(:)    = 0e0_pr
          csys(:,:)   = 0e0_pr

        else

          OK        = PASS
          SPOT      = RaySpot(1:N,1:2)                      ! ... RaySpot(mRay,2)  N=nSpot
          shift     = (/CntrSpot(:2), &                     ! Chief-Ray Position in selected CSYS
                        RefSpot(:2)/)                       ! VptElt    Position in selected CSYS
          centroid  = sum(RaySpot, dim=1)/real(N, kind=pr)  ! w.r.t. to Reference Position
          csys(:,1) = xLocal
          csys(:,2) = yLocal
          csys(:,3) = zLocal

        end if

      end subroutine spot_get


      !---------------------------------------------------------------------------------------------
      ! Set Exit Pupil (XP) information
      !---------------------------------------------------------------------------------------------
      subroutine xp_set(ok, vpt, psi, rad)
        implicit none
        logical,               intent(out):: ok    ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), dimension(3), intent(in) :: vpt   ! (x,y,z) Srf. position     in global CSYS
        real(8), dimension(3), intent(in) :: psi   ! (L,M,N) Srf. orientation. in global CSYS
        real(8),               intent(in) :: rad   ! ref. sphere radius
        ! ------------------------------------------------------

        ok = FAIL
        if (.not. SystemCheck() .or. (nElt<=3)) return

        VptElt(:, nElt-1) = vpt
        PsiElt(:, nElt-1) = psi
        KrElt(nElt-1)     = rad

        ok = PASS
      end subroutine xp_set


      !---------------------------------------------------------------------------------------------
      ! Get Exit Pupil (XP) information
      !---------------------------------------------------------------------------------------------
      subroutine xp_get(ok, vpt, psi, rad)
        implicit none
        logical,               intent(out) :: ok    ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), dimension(3), intent(out) :: vpt   ! (x,y,z) Srf. position     in global CSYS
        real(8), dimension(3), intent(out) :: psi   ! (L,M,N) Srf. orientation. in global CSYS
        real(8),               intent(out) :: rad   ! ref. sphere radius
        ! ------------------------------------------------------

        ok = FAIL
        if (.not. SystemCheck()) return

        vpt = VptElt(:, nElt-1)
        psi = PsiElt(:, nElt-1)
        rad = KrElt(nElt-1)

        ok = PASS
      end subroutine xp_get


      !---------------------------------------------------------------------------------------------
      ! Find Exit Pupil (XP) -- FEX cmd
      !---------------------------------------------------------------------------------------------
      ! Note     : assume XP Srf is Element nElt-1 & Stop is set
      !---------------------------------------------------------------------------------------------
      subroutine xp_fnd(OK, XP, mode)
        use       macos_mod, only: ifStopSet
        use smacos_vars_mod, only: npts
        use         src_mod, only: nGridPts

        implicit none
        logical, intent(out):: OK        ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), intent(out):: XP(7)     ! [Kr, Psi(L,M,N), Vpt(x,y,z)] @ XP Srf. (= nElt-1)
        integer, intent(in) :: mode      ! (PASS=1): Chief Ray, (FAIL=0): Centroid

        logical :: ifCentroidSave
        integer :: nGridPtsSave
        ! ------------------------------------------------------
        OK    = FAIL
        XP(:) = 0e0_pr
        if (.not. (SystemCheck() .and. ifStopSet)) return

        ! fix: SMACOS "FEX" does not correctly reset nGridPts
        nGridPtsSave = nGridPts

        ! define ref. srf. centering approach
        ifCentroidSave = ifCentroid
        ifCentroid     = (mode == FAIL)  ! mode .eqv. .false.

        command = 'FEX'
        IARG(1) = nElt-1
        CARG(1) = 'YES'
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ! fix: SMACOS "FEX" does not correctly reset nGridPts & npts
        if (nGridPtsSave/=nGridPts .or. npts/=nGridPtsSave-1) then

          nGridPts = nGridPtsSave
          npts     = nGridPts-1

          command = 'MODIFY'
          CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        end if

        ! restore
        ifCentroid = ifCentroidSave

        ! return
        XP = (/KrElt(nElt-1), PsiElt(:,nElt-1), VptElt(:,nElt-1)/)
        OK = PASS

      end subroutine xp_fnd


      !---------------------------------------------------------------------------------------------
      ! Get Stop Information
      !---------------------------------------------------------------------------------------------
      subroutine stop_info_get(OK, iElt, VptOffset)
        use Kinds
        use smacosio_mod, only: StopOffset, EltStopSet
        use    macos_mod, only: ifStopSet

        implicit none
        logical, intent(out):: OK            ! (PASS=1) if successful; (FAIL=0) otherwise
        integer, intent(out):: iElt          ! Element at which Optical System Stop is defined
        real(8), intent(out):: VptOffset(2)  ! [dx,dy]: Offset from Srf. Vertex Position
        ! ------------------------------------------------------

        if (.not. (SystemCheck() .and. ifStopSet .and. EltStopSet)) then
          iElt      = 0
          VptOffset = (/0e0_pr, 0e0_pr/)
          OK        = FAIL
        else
          iElt      = StopElt
          VptOffset = StopOffset
          OK        = PASS
        end if

      end subroutine stop_info_get


      !---------------------------------------------------------------------------------------------
      ! set Element Stop
      !---------------------------------------------------------------------------------------------
      ! Purpose  : Define Optical System Stop Position & update source to ensure Chief Ray goes
      !            through Stop position. Option exist to use current stop definition if already set.
      ! Call     : CALL stop_info_set(OK,iElt,Offset)
      ! Input    : iElt    [1x1,I]: Element at which Optical System Stop to define (0 < iElt <= nElt)
      !
      !            Offset  [2x1,D]: = [dx,dy]: Offset from Srf. Vertex Position
      !                             Special:if iElt<0, Offset data ignored
      ! Output   : OK      [1x1,B]: = (True) if successful; (False) otherwise
      ! Require  : Rx loaded
      ! Note     : -- Stop cannot be set at elements of type NSRefractor,NSReflector or Segment
      !            -- MACOS bug: reading a new Rx does NOT reset Stop information flags, i.e., info
      !                          from previous Rx are retained.
      !            -- if iElt<0 and Element Stop is not set, it will fail, i.e., OK = False
      !---------------------------------------------------------------------------------------------
      subroutine stop_info_set(OK,iElt,VptOffset)
        use smacos_vars_mod, only: npts
        use    smacosio_mod, only: RxStopSet, EltStopSet, StopOffset
        use         src_mod, only: StopElt
        use       macos_mod, only: ifStopSet
        use         elt_mod, only: EltID,NSRefractorElt,NSReflectorElt,SegmentElt

        implicit none
        logical, intent(out):: OK            ! (PASS=1) if successful; (FAIL=0) otherwise
        integer, intent(in) :: iElt          ! Element at which Optical System Stop to be defined (0 < iElt < nElt-2)
        real(8), intent(in) :: VptOffset(2)  ! [dx,dy]: Offset from Srf. Vertex Position
        ! ------------------------------------------------------
        OK = FAIL
        if (.not. SystemCheck() .or. nElt <=3 .or. iElt<1 .or. iElt>=nElt-2) return

        ! cannot set stop at element of type NSRefractor,NSReflector or Segment
        if ((EltID(iElt)==NSRefractorElt).or. &
            (EltID(iElt)==NSReflectorElt).or. &
            (EltID(iElt)==SegmentElt)) return

        ! chk Offset
        if (isnan(VptOffset(1)) .or. isnan(VptOffset(2))) return

        ! update
        StopElt    = iElt
        StopOffset = VptOffset

        RxStopSet  = .true.
        EltStopSet = .true.

        ! set Stop and enforce Chief Ray passing through Stop position
        command   = 'STOP'
        CARG(1)   = 'ELT'
        IARG(1)   = iElt          ! define Stop at Srf. iElt
        DARG(1:2) = VptOffset     ! Offset

        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ! MACOS Bug: After running STOP, nGridPts is not updated/restored
        if ((nGridPts-1)/=npts) nGridPts = npts + 1

        if (.not. ifStopSet) return

        OK = PASS

      end subroutine stop_info_set


    ! ============================================================================================
    !
    ! Element Surface Properties: Tools
    !
    ! ============================================================================================

      !---------------------------------------------------------------------------------------------
      ! return number of defined Elements
      !---------------------------------------------------------------------------------------------
      subroutine n_elt(nElt_out)
        implicit none
        integer, intent(out) :: nElt_out   ! returns # of Elements
        ! ------------------------------------------------------
        nElt_out = 0

        ! Check if system is initialized.
        if (.not. SystemCheck()) return

        nElt_out = nElt
      end subroutine n_elt


    ! ============================================================================================
    !
    ! SMACOS
    !
    ! ============================================================================================


      !---------------------------------------------------------------------------------------------
      !    Purpose : initialize smacos
      !    Call    : call init(ok, modelSize)
      !    Input   : modelSize [1x1,I]: SMACOS model size
      !    Output  : OK        [Logical]: OK = PASS if macos was initialized
      !                                   OK = FAIL otherwise
      !    Require : PyMACOS not be initialized
      !    Note    : This will set PyMACOS package internal setting to
      !                initialized
      !---------------------------------------------------------------------------------------------
      subroutine init(ok, modelsize)
        implicit none
        logical, intent(out) :: ok
        integer, intent(in) :: modelsize
        ! ------------------------------------------------------

        INCLUDE 'pymacos.inc'      ! ToDo: Clean it out

        LOGICAL :: ifInitOptics
        INTEGER :: m_err_pymacos
        INTEGER :: curr_model_size = -1   ! automatic save

        SAVE   ! it's necessary but which variables need to be saved?

        ok = FAIL
        ! -----------------------------------------------------------------------
        ! Initialize flags
        ! -----------------------------------------------------------------------

        DATA ifInitOptics/.FALSE./

        ! Initialize or reset SMACOS model
        firstEntry = modelsize /= curr_model_size

        IF (firstEntry) THEN ! .OR. modelsize /= curr_model_size) THEN
          curr_model_size = modelsize
          CALL macos_init_all(curr_model_size)

          IF (ALLOCATED(PixArray)) THEN
            DEALLOCATE(PixArray,OPDMat, RaySpot, stat=m_err_pymacos)
            IF (m_err_pymacos /= 0) THEN
              CALL macos_memory_failure('init: de-allocate failed.')
              return
            END IF
          END IF
          allocate(PixArray(mPix,mPix), OPDMat(mpts, mpts), RaySpot(mRay,2), &
                    stat=m_err_pymacos)
          IF (m_err_pymacos /= 0) THEN
            call macos_memory_failure('init: allocate failed!')
            stop "*** Memory allocation failed ***"
            return
          END IF

          mPix2=mPix*mPix

          IF (ALLOCATED(OPD)) THEN
            DEALLOCATE(OPD,SPOT,PIX,USER,STAT=m_err_pymacos)
            IF (m_err_pymacos /= 0) THEN
              CALL macos_memory_failure('init: de-allocate failed!')
              return
            END IF
          END IF

          ! Parameters have been defined after init call
          ALLOCATE(OPD(mRay),SPOT(2,mRay), PIX(mPix*mPix),USER(mElt), &
                    STAT=m_err_pymacos)
          IF (m_err_pymacos /= 0) THEN
            CALL macos_memory_failure('init: allocate failed!')
            return
          END IF
        END IF

        OPD(:)=0d0; SPOT(:,:)=0d0;  PIX(:)=0d0;  USER(:)=0d0
        firstEntry = .false.
        ok = PASS

      end subroutine init


      !---------------------------------------------------------------------------------------------
      !
      ! --- load_rx Rx ---
      !    Purpose  : load optical prescription (Rx)
      !    Call     : call load_rx(ok, nElt, rxName)
      !    Input    : rxName     File Name of Rx file
      !    Output   : OK         [Logical]: OK = PASS  if file was loaded
      !                                     OK = FAIL  otherwise
      !               nElt         [1x1,I]: Number of Elements loaded
      !    Require  : PyMACOS initialized
      !    Note     : Will set package internal setting to "Rx loaded"
      !---------------------------------------------------------------------------------------------
      subroutine load_rx(ok, nElt_out, rx)
        implicit none
        logical,            intent(out):: ok
        integer,            intent(out):: nElt_out
        character(len=250), intent(in) :: rx
        ! ------------------------------------------------------
        ok       = FAIL
        nElt_out = 0

        if (firstEntry) then
          print *, 'load_rx: pymacos is not initialized.'
          return
        end if

        if (len_trim(rx) > 256) return

        command = 'OLD'
        CARG(1) = trim(rx)
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        rxLoaded = (nElt > 0)
        if (.not. rxLoaded) return

        ! success
        ok = PASS
        nElt_out = nElt

      end subroutine load_rx


      !---------------------------------------------------------------------------------------------
      ! Save optical prescription
      !---------------------------------------------------------------------------------------------
      ! Warning:
      !    -- volatile operation -- Python needs to check file path existance
      !    -- not all parameters are saved (was not updated for a long time)
      !---------------------------------------------------------------------------------------------
      subroutine save_rx(ok, rx)
        implicit none
        logical,          intent(out):: ok
        character(len=*), intent(in) :: rx
        ! ------------------------------------------------------
        ok = FAIL

        if (.not. SystemCheck() .or. (len_trim(rx) > 256)) return  ! ToDo: update with actual parameter

        command = 'SAVE'
        CARG(1) = trim(rx)
        CARG(2) = 'YES'      ! overwrite if file exists
        CALL SMACOS(command,CARG,DARG,IARG,LARG,RARG,OPDMat,RaySpot,RMSWFE,PixArray)

        ok = PASS
      end subroutine save_rx


      ! ============================================================================================
      !
      ! Utilities
      !
      ! ============================================================================================

      !---------------------------------------------------------------------------------------------
      ! Check: -nElt < iElt <= nElt
      !---------------------------------------------------------------------------------------------
      logical function checkEltID(elt, n, m)
        implicit none
        integer, dimension(n,m), intent(in) :: elt
        integer, intent(in) :: n, m
        ! ------------------------------------------------------
        checkEltID = (all(-nElt < elt(:,:)).and.all(elt(:,:) <= nElt))

      end function checkEltID


      !---------------------------------------------------------------------------------------------
      logical function checkSurfaceID(elt)
        implicit none
        integer, intent(in) :: elt
        ! ------------------------------------------------------
        if (-nElt < elt .and. elt <= nElt) then
          checkSurfaceID = PASS
        else
          checkSurfaceID = FAIL
        end if

      end function checkSurfaceID


      !---------------------------------------------------------------------------------------------
      ! Purpose  : check status
      ! Input    : ----
      ! Output   : (T) if SMACOS loaded and Rx loaded; (F) otherwise
      !---------------------------------------------------------------------------------------------
      logical function SystemCheck()
        implicit none

        if (.not. firstEntry .and. rxLoaded) then
          SystemCheck = PASS
        else
          SystemCheck = FAIL
        end if

      end function SystemCheck


      !---------------------------------------------------------------------------------------------
      ! Purpose  : check if elements are in range:  0 < iElt(i,j) <= nElt
      ! Input    : iElt  [NxM,I]: matrix containing element IDs
      ! Output   : (T) if within range; (F) otherwise
      !---------------------------------------------------------------------------------------------
      logical function EltRangeChk(iElt,LowerLimit)

        implicit none
        integer, dimension(:), intent(in):: iElt
        integer,               intent(in):: LowerLimit
        ! ------------------------------------------------------

        EltRangeChk = .not. any((iElt<LowerLimit).OR.(iElt>nElt))

      end function EltRangeChk


      !---------------------------------------------------------------------------------------------
      ! Purpose  : Combine EltRangeChk & SystemCheck
      ! Input    : iElt  [Nx1,I]: matrix containing element IDs
      ! Output   : (T) if within range; (F) otherwise
      !---------------------------------------------------------------------------------------------
      logical function StatusChk1(iElt)

        implicit none
        integer, dimension(:), intent(in):: iElt
        ! ------------------------------------------------------

        StatusChk1 = (SystemCheck() .and. &    ! SMACOS and Rx status check
                      EltRangeChk(iElt,1))     ! valid range: 0 < iElt(i,j) <= nElt

      end function StatusChk1
      !---------------------------------------------------------------------------------------------


      !---------------------------------------------------------------------------------------------
      ! map scalar element ID from  [-nElt < iElt <= nElt]   to   [0 < iElt <= nElt]
      !---------------------------------------------------------------------------------------------
      subroutine translateSurfaceID(iElt)
        implicit none
        integer, intent(inout) :: iElt
        ! ------------------------------------------------------
        if ( iElt .le. 0 ) then
            iElt = nElt + iElt
        end if

      end subroutine translateSurfaceID


      !---------------------------------------------------------------------------------------------
      ! map array element ID from  [-nElt < iElt <= nElt]   to   [0 < iElt <= nElt]
      !---------------------------------------------------------------------------------------------
      subroutine translateEltID(iElt, n, m)
        implicit none
        integer, dimension(n,m), intent(inout):: iElt
        integer,                 intent(in)   :: n, m
        ! ------------------------------------------------------

        where (iElt(:n, :m) <= 0) iElt(:n, :m) = nElt + iElt(:n, :m)

      end subroutine translateEltID


  end module api
