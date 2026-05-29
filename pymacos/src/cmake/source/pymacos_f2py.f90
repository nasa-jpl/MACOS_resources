! pymacos_f2py.f90 -- f2py-facing wrapper layer
!
! Auto-derived from pymacos.f90 by /tmp/gen_pymacos_refactor.py.
! Defines `module api` (the name Python imports as pymacos.lib.api)
! and forwards every routine to its language-neutral sibling in
! macos_api_mod.  The !f2py annotations live here so the pure-Fortran
! backbone stays free of Python-specific markers.
!
! Functions/state are still exposed via re-exports below; only the
! subroutines need explicit wrappers because f2py looks at the
! subroutine declaration in *this* file to derive the Python signature.


  module api

    use Kinds
    use math_mod
    use param_mod
    use src_mod
    use elt_mod
    use macos_mod
    use smacos_mod

    use macos_api_mod, only: &
      modified_rx_impl => modified_rx, &
      prb_elt_impl => prb_elt, &
      prb_elt_grp_impl => prb_elt_grp, &
      set_src_csys_impl => set_src_csys, &
      get_src_csys_impl => get_src_csys, &
      set_src_sampling_impl => set_src_sampling, &
      get_src_sampling_impl => get_src_sampling, &
      sys_units_impl => sys_units, &
      src_info_impl => src_info, &
      get_src_fov_impl => get_src_fov, &
      src_wvl_impl => src_wvl, &
      set_src_info_impl => set_src_info, &
      set_src_fov_impl => set_src_fov, &
      src_size_impl => src_size, &
      get_src_size_impl => get_src_size, &
      set_src_size_impl => set_src_size, &
      src_finite_impl => src_finite, &
      elt_grp_max_all_impl => elt_grp_max_all, &
      elt_grp_max_impl => elt_grp_max, &
      elt_grp_any_impl => elt_grp_any, &
      elt_grp_fnd_impl => elt_grp_fnd, &
      elt_grp_del_impl => elt_grp_del, &
      elt_grp_del_all_impl => elt_grp_del_all, &
      elt_grp_set_impl => elt_grp_set, &
      elt_grp_get_impl => elt_grp_get, &
      elt_vpt_impl => elt_vpt, &
      elt_psi_impl => elt_psi, &
      elt_rpt_impl => elt_rpt, &
      elt_kc_impl => elt_kc, &
      elt_kr_impl => elt_kr, &
      elt_csys_set_impl => elt_csys_set, &
      elt_csys_get_impl => elt_csys_get, &
      elt_csys_rm_impl => elt_csys_rm, &
      elt_srf_csys_impl => elt_srf_csys, &
      elt_srf_csys_pos_impl => elt_srf_csys_pos, &
      elt_srf_csys_dir_impl => elt_srf_csys_dir, &
      elt_srf_csys_set_impl => elt_srf_csys_set, &
      elt_srf_csys_get_impl => elt_srf_csys_get, &
      elt_srf_grid_any_impl => elt_srf_grid_any, &
      elt_srf_grid_fnd_impl => elt_srf_grid_fnd, &
      elt_srf_grid_fnd_type_impl => elt_srf_grid_fnd_type, &
      elt_srf_grid_size_max_impl => elt_srf_grid_size_max, &
      elt_srf_grid_size_impl => elt_srf_grid_size, &
      elt_srf_grid_spacing_impl => elt_srf_grid_spacing, &
      elt_srf_grid_data_scale_impl => elt_srf_grid_data_scale, &
      elt_srf_grid_data_impl => elt_srf_grid_data, &
      elt_srf_grid_data_add_impl => elt_srf_grid_data_add, &
      elt_srf_grating_any_impl => elt_srf_grating_any, &
      elt_srf_grating_fnd_impl => elt_srf_grating_fnd, &
      elt_srf_grating_rule_width_impl => elt_srf_grating_rule_width, &
      elt_srf_grating_order_impl => elt_srf_grating_order, &
      elt_srf_grating_type_impl => elt_srf_grating_type, &
      elt_srf_grating_rule_dir_impl => elt_srf_grating_rule_dir, &
      elt_srf_grating_params_impl => elt_srf_grating_params, &
      elt_srf_zrn_any_impl => elt_srf_zrn_any, &
      elt_srf_zrn_fnd_impl => elt_srf_zrn_fnd, &
      elt_srf_zrn_coef_impl => elt_srf_zrn_coef, &
      elt_srf_zrn_type_impl => elt_srf_zrn_type, &
      elt_srf_zrn_norm_radius_impl => elt_srf_zrn_norm_radius, &
      elt_srf_zrn_set_impl => elt_srf_zrn_set, &
      elt_srf_zrn_mode_set_impl => elt_srf_zrn_mode_set, &
      elt_srf_zrn_get_impl => elt_srf_zrn_get, &
      elt_srf_ff_any_impl => elt_srf_ff_any, &
      elt_srf_ff_fnd_impl => elt_srf_ff_fnd, &
      elt_srf_mon_zrn_max_modes_impl => elt_srf_mon_zrn_max_modes, &
      ray_info_get_impl => ray_info_get, &
      ray_info_set_impl => ray_info_set, &
      trace_rays_impl => trace_rays, &
      opd_val_impl => opd_val, &
      spot_cmd_impl => spot_cmd, &
      spot_get_impl => spot_get, &
      int_cmd_impl => int_cmd, &
      int_get_impl => int_get, &
      cfield_cmd_impl => cfield_cmd, &
      elt_dx_get_impl => elt_dx_get, &
      base_unit_to_metres_impl => base_unit_to_metres, &
      cfield_get_impl => cfield_get, &
      cfield_apodize_impl => cfield_apodize, &
      cfield_apodize_complex_impl => cfield_apodize_complex, &
      perturb_elt_impl => perturb_elt, &
      perturb_src_impl => perturb_src, &
      xp_set_impl => xp_set, &
      xp_get_impl => xp_get, &
      xp_fnd_impl => xp_fnd, &
      sxp_fnd_impl => sxp_fnd, &
      ors_run_impl => ors_run, &
      srs_run_impl => srs_run, &
      stop_info_get_impl => stop_info_get, &
      stop_info_set_impl => stop_info_set, &
      stop_obj_set_impl => stop_obj_set, &
      n_elt_impl => n_elt, &
      init_impl => init, &
      load_rx_impl => load_rx, &
      save_rx_impl => save_rx, &
      translateSurfaceID_impl => translateSurfaceID, &
      translateEltID_impl => translateEltID

    ! Re-export functions and module state that Python may reach via lib.api.*
    use macos_api_mod, only: &
      currrent_macos_model_size, &
      checkSurfaceID, &
      checkEltID, &
      SystemCheck, &
      EltRangeChk, &
      StatusChk1

    implicit none

  contains

      subroutine modified_rx(ok)

        implicit none
        logical, intent(out):: ok

        CALL modified_rx_impl(ok)
      end subroutine modified_rx

      subroutine prb_elt(OK,iElt,prb,ifGlobal,n)

        implicit none
        logical,                 intent(out):: OK        ! (True) if successful; (False) otherwise
        integer, dimension(n),   intent(in) :: iElt      ! ID (Range: -nElt < iElt[i,j] <= nElt)
        real(8), dimension(6,n), intent(in) :: prb
        integer, dimension(n),   intent(in) :: ifGlobal

        integer,                 intent(in) :: n         ! Number of different elements to be updated
        !f2py  integer intent(hide), depend(iElt) :: n=len(iElt)

        integer :: i

        CALL prb_elt_impl(OK,iElt,prb,ifGlobal,n)
      end subroutine prb_elt

      subroutine prb_elt_grp(OK,iElt,prb,ifGlobal,n)

        implicit none
        logical,                 intent(out):: OK        ! (True) if successful; (False) otherwise
        integer, dimension(n),   intent(in) :: iElt      ! ID (Range: -nElt < iElt[i,j] <= nElt)
        real(8), dimension(6,n), intent(in) :: prb
        logical, dimension(n),   intent(in) :: ifGlobal

        integer,                 intent(in) :: n         ! Number of different elements to be updated
        !f2py  integer intent(hide), depend(iElt) :: n=len(iElt)

        integer :: i

        CALL prb_elt_grp_impl(OK,iElt,prb,ifGlobal,n)
      end subroutine prb_elt_grp

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

        CALL set_src_csys_impl(ok,xDir,yDir,zDir,Axis,xAxis,Rz,filter)
      end subroutine set_src_csys

      subroutine get_src_csys(ok,xDir,yDir,zDir)
        use src_mod, only: xGrid,yGrid,zGrid

        implicit none
        logical,               intent(out):: ok
        real(8), dimension(3), intent(out):: xDir, yDir, zDir

        CALL get_src_csys_impl(ok,xDir,yDir,zDir)
      end subroutine get_src_csys

      subroutine set_src_sampling(OK,N)
        use smacos_vars_mod, only: npts

        implicit none
        logical, intent(out):: OK
        integer, intent(in) :: N

        CALL set_src_sampling_impl(OK,N)
      end subroutine set_src_sampling

      subroutine get_src_sampling(OK,N)

        implicit none
        logical, intent(out):: OK
        integer, intent(out):: N

        CALL get_src_sampling_impl(OK,N)
      end subroutine get_src_sampling

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

        CALL sys_units_impl(OK, BaseUnitID, WaveUnitID)
      end subroutine sys_units

      subroutine src_info(OK,zSrc,SrcPos,SrcDir,IsPtSrc,WL,SrcApe,SrcObs,BaseUnitID,WaveUnitID)

        implicit none
        logical,               intent(out):: OK
        real(8),               intent(out):: zSrc
        real(8), dimension(3), intent(out):: SrcPos,SrcDir
        logical,               intent(out):: IsPtSrc
        real(8),               intent(out):: WL,SrcApe,SrcObs
        integer,               intent(out):: BaseUnitID         ! (1) 'm', (2) 'cm', (3) 'mm', (4) 'in', (0) 'none' = not set
        integer,               intent(out):: WaveUnitID         ! (1) 'm', (2) 'cm', (3) 'mm', (4) 'um', (5) 'nm', (6) 'A', (7) 'in', (9) 'none'= not set

        CALL src_info_impl(OK,zSrc,SrcPos,SrcDir,IsPtSrc,WL,SrcApe,SrcObs,BaseUnitID,WaveUnitID)
      end subroutine src_info

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

        CALL get_src_fov_impl(OK, zSrc, SrcPos, SrcDir, IsPtSrc)
      end subroutine get_src_fov

      subroutine src_wvl(OK, WVL, setter)
        implicit none
        integer, intent(out)   :: OK       ! (PASS) if successful; (FAIL) otherwise
        real(8), intent(inout) :: WVL      ! Wavelength in WaveUnits (WVL > 0)
        integer, intent(in)    :: setter   ! (PASS) to set & (FAIL) to get

        CALL src_wvl_impl(OK, WVL, setter)
      end subroutine src_wvl

      subroutine set_src_info(OK,zSrc,SrcPos,SrcDir,WL,SrcApe,SrcObs)
        use Constants, only: eps

        implicit none
        logical,               intent(out):: OK
        real(8),               intent(in) :: zSrc
        real(8), dimension(3), intent(in) :: SrcPos,SrcDir
        real(8),               intent(in) :: WL,SrcApe,SrcObs
        !f2py  intent(hide):: OK

        integer :: tmp

        CALL set_src_info_impl(OK,zSrc,SrcPos,SrcDir,WL,SrcApe,SrcObs)
      end subroutine set_src_info

      subroutine set_src_fov(OK,zSrc,SrcPos,SrcDir)
        use Constants, only: EPS

        implicit none
        logical,               intent(out):: OK
        real(8),               intent(in) :: zSrc
        real(8), dimension(3), intent(in) :: SrcPos,SrcDir

        real(8) :: tmp(7)

        CALL set_src_fov_impl(OK,zSrc,SrcPos,SrcDir)
      end subroutine set_src_fov

      subroutine src_size(OK, SrcApe, SrcObs, Setter)
        implicit none
        logical, intent(out)  :: OK      ! (PASS) if successful; (FAIL) otherwise
        real(8), intent(inout):: SrcApe  ! Ape. Beam N.A. for Pt.Src.;otherwise, Apt. Beam Diameter
        real(8), intent(inout):: SrcObs  ! Obs. Beam N.A. for Pt.Src.;otherwise, Obs. Beam Diameter
        integer, intent(in)   :: Setter  ! (PASS) to set & (FAIL) to get

        logical :: set_ape, set_obs

        CALL src_size_impl(OK, SrcApe, SrcObs, Setter)
      end subroutine src_size

      subroutine get_src_size(OK, SrcApe, SrcObs)
        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: SrcApe,SrcObs

        CALL get_src_size_impl(OK, SrcApe, SrcObs)
      end subroutine get_src_size

      subroutine set_src_size(OK,SrcApe,SrcObs)
        !use,intrinsic :: ieee_arithmetic  -- not working with gcc to use with ieee_is_nan(), ieee_is_finite(x)

        implicit none
        logical, intent(out):: OK
        real(8), intent(in) :: SrcApe,SrcObs

        CALL set_src_size_impl(OK,SrcApe,SrcObs)
      end subroutine set_src_size

      subroutine src_finite(OK, IsPtSrc)
        use sourcsub_mod, only: isPointSource

        implicit none
        logical, intent(out):: OK
        logical, intent(out):: IsPtSrc

        CALL src_finite_impl(OK, IsPtSrc)
      end subroutine src_finite

      subroutine elt_grp_max_all(maxGrpSize)

        implicit none
        integer, intent(out):: maxGrpSize ! max. defined Grp. Size

        CALL elt_grp_max_all_impl(maxGrpSize)
      end subroutine elt_grp_max_all

      subroutine elt_grp_max(ok,maxGrpSize,iElt,N)

        implicit none
        logical,               intent(out):: ok         ! success (1) or Fail (0)
        integer,               intent(out):: maxGrpSize ! max. defined Grp. Size
        integer, dimension(N), intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)

        integer,               intent(in) :: N

        CALL elt_grp_max_impl(ok,maxGrpSize,iElt,N)
      end subroutine elt_grp_max

      subroutine elt_grp_any(any_elt_grp)

        implicit none
        logical, intent(out):: any_elt_grp

        CALL elt_grp_any_impl(any_elt_grp)
      end subroutine elt_grp_any

      subroutine elt_grp_fnd(ok,nGrp,iElt,N)

        implicit none
        logical,                     intent(out):: ok    ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: nGrp  ! 1 (Grp defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt  ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N

        CALL elt_grp_fnd_impl(ok,nGrp,iElt,N)
      end subroutine elt_grp_fnd

      subroutine elt_grp_del(ok,iElt,N)

        implicit none
        logical,                     intent(out):: ok     ! (True,1) if successful; (False,0) otherwise
        integer, dimension(N),       intent(in) :: iElt   ! Elt. ID ( -nElt < iElt[i] <= nElt )

        integer,                     intent(in) :: N      ! # of Elements in iElt

        CALL elt_grp_del_impl(ok,iElt,N)
      end subroutine elt_grp_del

      subroutine elt_grp_del_all(ok)

        implicit none
        logical, intent(out):: ok  ! (True,1) if successful; (False,0) otherwise

        CALL elt_grp_del_all_impl(ok)
      end subroutine elt_grp_del_all

      subroutine elt_grp_set(ok,iElt,jEltGrp,nEltGrp)

        implicit none
        logical,                     intent(out):: ok       ! (True,1) if successful; (False,0) otherwise
        integer,                     intent(in) :: iElt     ! Elt. ID  (1 <= iElt[i]    <= nElt )
        integer, dimension(nEltGrp), intent(in) :: jEltGrp  ! Grp. Members where (0 <= jEltGrp[i] <= nElt )

        integer,                     intent(in) :: nEltGrp  ! # of Grp. Members (= length(jEltGrp) <= mElt)

        CALL elt_grp_set_impl(ok,iElt,jEltGrp,nEltGrp)
      end subroutine elt_grp_set

      subroutine elt_grp_get(ok,jEltGrp,nEltGrp,iElt,N,mElt_)

        implicit none
        logical,                     intent(out):: ok       ! (True,1) if successful; (False,0) otherwise
        integer, dimension(mElt_,N), intent(out):: jEltGrp  ! Grp. Members where K = mElt (value set to -1 for not used)
        integer, dimension(N),       intent(out):: nEltGrp  ! # of Grp. Members defined in jEltGrp
        integer, dimension(N),       intent(in) :: iElt     ! Elt. ID  (1 <= iElt[i]    <= nElt )

        integer,                     intent(in) :: N        ! # of Elements where to retrieve Grp. IDs
        integer,                     intent(in) :: mElt_    ! Max. # of Elements permitted (for Python = mElt (SMACOS var))

        CALL elt_grp_get_impl(ok,jEltGrp,nEltGrp,iElt,N,mElt_)
      end subroutine elt_grp_get

      subroutine elt_vpt(ok, iElt, Vpt, setter, n)

        implicit none
        logical,                  intent(out)  :: ok        ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n),   intent(in)   :: iElt      ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(3,n), intent(inout):: Vpt       ! Vertex Position [3 x N]
        logical,                  intent(in)   :: setter    ! =PASS for set; =FAIL for get

        integer,                  intent(in)   :: n         ! # of Elements

        CALL elt_vpt_impl(ok, iElt, Vpt, setter, n)
      end subroutine elt_vpt

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

        CALL elt_psi_impl(ok, iElt, Psi, setter, n)
      end subroutine elt_psi

      subroutine elt_rpt(ok, iElt, Rpt, setter, n)

        implicit none
        logical,                  intent(out)  :: ok        ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n),   intent(in)   :: iElt      ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(3,n), intent(inout):: Rpt       ! Rotation Position [3 x N]
        logical,                  intent(in)   :: setter    ! =PASS for set; =FAIL for get

        integer,                  intent(in)   :: n         ! # of Elements

        CALL elt_rpt_impl(ok, iElt, Rpt, setter, n)
      end subroutine elt_rpt

      subroutine elt_kc(ok, iElt, Kc, setter, n)

        implicit none
        logical,                intent(out)  :: ok       ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n), intent(in)   :: iElt     ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(n), intent(inout):: Kc       ! Base Radii of elements
        logical,                intent(in)   :: setter   ! if PASS, set; otherwise, return values

        integer,                intent(in)   :: n        ! # of Elements

        CALL elt_kc_impl(ok, iElt, Kc, setter, n)
      end subroutine elt_kc

      subroutine elt_kr(ok, iElt, Kr, setter, n)

        implicit none
        logical,                intent(out)  :: ok       ! (PASS) if successful; (FAIL) otherwise
        integer,  dimension(n), intent(in)   :: iElt     ! Surface ID: 0 < iElt <= nElt
        real(8),  dimension(n), intent(inout):: Kr       ! Base Radii of elements
        logical,                intent(in)   :: setter   ! if PASS, set; otherwise, return values

        integer,                intent(in)   :: n        ! # of Elements

        CALL elt_kr_impl(ok, iElt, Kr, setter, n)
      end subroutine elt_kr

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

        CALL elt_csys_set_impl(ok,iElt,xDir,yDir,zDir,Upd,m)
      end subroutine elt_csys_set

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

        CALL elt_csys_get_impl(ok, iElt, csys, csys_lcs, csys_upd, N)
      end subroutine elt_csys_get

      subroutine elt_csys_rm(ok,iElt,N)

        implicit none
        logical,               intent(out):: ok
        integer, dimension(N), intent(in) :: iElt    ! Surfaces where to remove LCS
        integer,               intent(in) :: N

        CALL elt_csys_rm_impl(ok,iElt,N)
      end subroutine elt_csys_rm

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

        CALL elt_srf_csys_impl(ok, pMon_, xMon_, yMon_, zMon_, iElt, setter, N)
      end subroutine elt_srf_csys

      subroutine elt_srf_csys_pos(ok, pMon_, iElt, setter, N)
        use Constants, only: EPS

        implicit none
        logical,                  intent(out)  :: ok
        real(8),  dimension(3,N), intent(inout):: pMon_
        integer,  dimension(N),   intent(in)   :: iElt
        integer,                  intent(in)   :: setter
        integer,                  intent(in)   :: N
        !f2py integer intent(hide),depend(iElt) :: N=len(iElt)    ! check(shape(pMon_,0)==3, shape(pMon_,1)==len(iElt))

        integer :: i

        CALL elt_srf_csys_pos_impl(ok, pMon_, iElt, setter, N)
      end subroutine elt_srf_csys_pos

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

        CALL elt_srf_csys_dir_impl(ok, xMon_, yMon_, zMon_, iElt, setter, N)
      end subroutine elt_srf_csys_dir

      subroutine elt_srf_csys_set(ok,iElt,pMon_,xMon_,yMon_,zMon_,N)
        use Constants, only: EPS
        use  math_mod, only: dorthoganalize

        implicit none
        logical,                intent(out):: ok
        integer,  dimension(N), intent(in) :: iElt
        real(8),  dimension(3), intent(in) :: pMon_,xMon_,yMon_,zMon_
        integer,                intent(in) :: N
        !f2py  integer intent(hide), depend(iElt) :: N=len(iElt)

        integer :: j
        real(8) :: A(3,3)

        CALL elt_srf_csys_set_impl(ok,iElt,pMon_,xMon_,yMon_,zMon_,N)
      end subroutine elt_srf_csys_set

      subroutine elt_srf_csys_get(ok,pMon_,xMon_,yMon_,zMon_,iElt,N)

        implicit none
        logical,                  intent(out):: ok
        real(8),  dimension(3,N), intent(out):: pMon_,xMon_,yMon_,zMon_
        integer,  dimension(N),   intent(in) :: iElt
        integer,                  intent(in) :: N
        !f2py  integer intent(hide),depend(iElt) :: N=len(iElt)  ! ,check(shape(pMon_,1)==shape(xMon_,1),shape(pMon_,1)==shape(yMon_,1),shape(pMon_,1)==shape(zMon_,1),len(iElt)==shape(pMon_,1))

        integer :: j

        CALL elt_srf_csys_get_impl(ok,pMon_,xMon_,yMon_,zMon_,iElt,N)
      end subroutine elt_srf_csys_get

      subroutine elt_srf_grid_any(any_elt)

        implicit none
        logical, intent(out):: any_elt   ! success (1) or Fail (0)

        CALL elt_srf_grid_any_impl(any_elt)
      end subroutine elt_srf_grid_any

      subroutine elt_srf_grid_fnd(ok, IsGridSrf, iElt, N)

        implicit none
        logical,                     intent(out):: ok         ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: IsGridSrf  ! 1 (Grp defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N

        CALL elt_srf_grid_fnd_impl(ok, IsGridSrf, iElt, N)
      end subroutine elt_srf_grid_fnd

      subroutine elt_srf_grid_fnd_type(ok, IsGridSrf, iElt, GridTypeID, N)
        implicit none
        logical,                     intent(out):: ok         ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: IsGridSrf  ! 1 (defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)
        integer,                     intent(in) :: GridTypeID ! Specify SyrfaceType of Type Grid

        integer,                     intent(in) :: N

        CALL elt_srf_grid_fnd_type_impl(ok, IsGridSrf, iElt, GridTypeID, N)
      end subroutine elt_srf_grid_fnd_type

      subroutine elt_srf_grid_size_max(MaxGridSize)

        implicit none
        integer, intent(out):: MaxGridSize ! Max Sampling Grid Size

        CALL elt_srf_grid_size_max_impl(MaxGridSize)
      end subroutine elt_srf_grid_size_max

      subroutine elt_srf_grid_size(ok,GridSize, iElt, N)

        implicit none
        logical,               intent(out):: ok         ! success (1) or Fail (0)
        integer, dimension(N), intent(out):: GridSize   ! Grid Value (= -1 Not defined at Srf.)
        integer, dimension(N), intent(in) :: iElt       ! Elt ID: (0 < iElt[i] <= nElt)

        integer,               intent(in) :: N

        CALL elt_srf_grid_size_impl(ok,GridSize, iElt, N)
      end subroutine elt_srf_grid_size

      subroutine elt_srf_grid_spacing(ok,iElt,GridSrfdx_,setter,N)

        implicit none
        logical,               intent(out)   :: ok           ! success (1) or Fail (0)
        integer, dimension(N), intent(in)    :: iElt         ! Elt ID: (0 < iElt[i] <= nElt)
        real(8), dimension(N), intent(inout) :: GridSrfdx_   ! grid data sampling spacing dx==dy
        logical,               intent(in)    :: setter       ! if PASS, define new values

        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt,GridSrfdx_):: N=len(iElt)

        logical, dimension(N) :: chk

        CALL elt_srf_grid_spacing_impl(ok,iElt,GridSrfdx_,setter,N)
      end subroutine elt_srf_grid_spacing

      subroutine elt_srf_grid_data_scale(ok,iElt,scalar,N)

        implicit none
        logical,               intent(out):: ok       ! success (1) or Fail (0)
        integer, dimension(N), intent(in) :: iElt     ! Elt ID: (0 < iElt[i] <= nElt)
        real(8), dimension(N), intent(in) :: scalar   ! grid scaling factor

        integer,               intent(in) :: N
        !f2py   integer intent(hide), depend(iElt,GridSrfdx_):: N=len(iElt)

        logical, dimension(N) :: chk
        integer :: npts

        CALL elt_srf_grid_data_scale_impl(ok,iElt,scalar,N)
      end subroutine elt_srf_grid_data_scale

      subroutine elt_srf_grid_data(ok,iElt,GridSrfdx_,GridMat_,setter,Nx,Ny)

        implicit none
        logical,                   intent(out)   :: ok           ! success (1) or Fail (0)
        integer,                   intent(in)    :: iElt         ! Elt ID: (0 < iElt[i] <= nElt)
        real(8),                   intent(inout) :: GridSrfdx_   ! grid data sampling spacing dx==dy
        real(8), dimension(Ny,Nx), intent(inout) :: GridMat_     ! displacement at node points from nominal shape (N x N) Grid
        logical,                   intent(in)    :: setter       ! if PASS, define new values

        integer,                   intent(in) :: Nx,Ny

        CALL elt_srf_grid_data_impl(ok,iElt,GridSrfdx_,GridMat_,setter,Nx,Ny)
      end subroutine elt_srf_grid_data

      subroutine elt_srf_grid_data_add(ok,iElt,GridMat_,Nx,Ny)

        implicit none
        logical,                   intent(out)   :: ok           ! success (1) or Fail (0)
        integer,                   intent(in)    :: iElt         ! Elt ID: (0 < iElt[i] <= nElt)
        real(8), dimension(Ny,Nx), intent(inout) :: GridMat_     ! displacement at node points from nominal shape (N x N) Grid

        integer,                   intent(in)    :: Nx,Ny

        CALL elt_srf_grid_data_add_impl(ok,iElt,GridMat_,Nx,Ny)
      end subroutine elt_srf_grid_data_add

    subroutine elt_srf_grating_any(Any_Elt_With_Grating)

      implicit none
      logical, intent(out):: Any_Elt_With_Grating

      CALL elt_srf_grating_any_impl(Any_Elt_With_Grating)
    end subroutine elt_srf_grating_any

    subroutine elt_srf_grating_fnd(ok, Grating, iElt, N)

      implicit none
      logical,               intent(out):: ok       ! success (1) or Fail (0)
      integer, dimension(N), intent(out):: Grating  ! 0) No, (1) Refl.  (2) Trans.
      integer, dimension(N), intent(in) :: iElt     ! Elt ID: (0 < iElt[i] <= nElt)

      integer,               intent(in) :: N

      CALL elt_srf_grating_fnd_impl(ok, Grating, iElt, N)
    end subroutine elt_srf_grating_fnd

    subroutine elt_srf_grating_rule_width(ok, iElt, Spacing, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      real(8), intent(inout) :: Spacing    ! RuleWidth (0 < Spacing < Inf)
      logical, intent(in)    :: setter     ! if PASS, define new values

      CALL elt_srf_grating_rule_width_impl(ok, iElt, Spacing, setter)
    end subroutine elt_srf_grating_rule_width

    subroutine elt_srf_grating_order(ok, iElt, Order, setter, N)
      logical, intent(out)                :: ok         ! success status (0/1)
      integer, dimension(N), intent(in)   :: iElt       ! Surface 0 <= iElt <= nElt
      integer, dimension(N), intent(inout):: Order      ! Diffraction Order
      logical,               intent(in)   :: setter     ! if PASS, define new values

      integer,               intent(in)   :: N          ! # of Elements
      !f2py integer intent(hide), depend(iElt,Order), check(len(iElt)==len(Order)) :: N=len(iElt)
      logical :: chk(N)

      CALL elt_srf_grating_order_impl(ok, iElt, Order, setter, N)
    end subroutine elt_srf_grating_order

    subroutine elt_srf_grating_type(ok, iElt, reflective, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      logical, intent(inout) :: reflective ! (1) Refl. (0) Trans.
      logical, intent(in)    :: setter     ! if PASS, define new values

      CALL elt_srf_grating_type_impl(ok, iElt, reflective, setter)
    end subroutine elt_srf_grating_type

    subroutine elt_srf_grating_rule_dir(ok, iElt, h1HOE_, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      real(8), intent(inout) :: h1HOE_(3)  ! h1HOE => perpendicular to the ruling direction and psiElt vector.
      logical, intent(in)    :: setter     ! if PASS, define new values

      CALL elt_srf_grating_rule_dir_impl(ok, iElt, h1HOE_, setter)
    end subroutine elt_srf_grating_rule_dir

    subroutine elt_srf_grating_params(ok, iElt, Spacing, Diff_Order, h1HOE_, reflective, setter)
      logical, intent(out)   :: ok         ! success status (0/1)
      integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
      integer, intent(inout) :: Diff_Order ! Diffraction Order
      real(8), intent(inout) :: Spacing    ! RuleWidth (0 < Spacing < Inf)
      real(8), intent(inout) :: h1HOE_(3)  ! h1HOE => perpendicular to the ruling direction and psiElt vector.
      logical, intent(inout) :: reflective ! (1) Refl. (0) Trans.
      logical, intent(in)    :: setter     ! if PASS, define new values

      CALL elt_srf_grating_params_impl(ok, iElt, Spacing, Diff_Order, h1HOE_, reflective, setter)
    end subroutine elt_srf_grating_params

      subroutine elt_srf_zrn_any(any_elt_Zrn)

        implicit none
        logical, intent(out):: any_elt_Zrn

        CALL elt_srf_zrn_any_impl(any_elt_Zrn)
      end subroutine elt_srf_zrn_any

      subroutine elt_srf_zrn_fnd(ok, nZrn, iElt, N)

        implicit none
        logical,                     intent(out):: ok    ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: nZrn  ! 1 (Grp defined) or 0 (not defined)
        integer, dimension(N),       intent(in) :: iElt  ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N

        CALL elt_srf_zrn_fnd_impl(ok, nZrn, iElt, N)
      end subroutine elt_srf_zrn_fnd

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

        CALL elt_srf_zrn_coef_impl(ok, iElt, ZernMode, ZernCoef_, setter, reset, N)
      end subroutine elt_srf_zrn_coef

      subroutine elt_srf_zrn_type(ok, iElt, ZernType, setter, reset)
        use elt_mod,   only: mZernType

        implicit none
        logical, intent(out)   :: ok         ! success status (0/1)
        integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
        integer, intent(inout) :: ZernType   ! Zernike Type
        logical, intent(in)    :: setter     ! if PASS, define new values
        logical, intent(in)    :: reset      ! if PASS, reset all coefs first (only for setter)

        CALL elt_srf_zrn_type_impl(ok, iElt, ZernType, setter, reset)
      end subroutine elt_srf_zrn_type

      subroutine elt_srf_zrn_norm_radius(ok, iElt, NormRad, setter)
        logical, intent(out)   :: ok         ! success status (0/1)
        integer, intent(in)    :: iElt       ! Surface 0 <= iElt <= nElt
        real(8), intent(inout) :: NormRad    ! Zernike Norm. Radius (0 < lMon < Inf)
        logical, intent(in)    :: setter     ! if PASS, define new values

        CALL elt_srf_zrn_norm_radius_impl(ok, iElt, NormRad, setter)
      end subroutine elt_srf_zrn_norm_radius

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

        CALL elt_srf_zrn_set_impl(ok,iElt,lMon_,ZernType,ZernMode,ZernCoef_,ZernAnnularRatio,M,Nm,Nc)
      end subroutine elt_srf_zrn_set

      subroutine elt_srf_zrn_mode_set(ok,iElt,ZernMode,ZernCoef_,M,N)
        !use Constants, only: eps
        use elt_mod,   only: mZernType, mZernModes

        implicit none
        logical,               intent(out):: ok
        integer, dimension(M), intent(in) :: iElt
        integer, dimension(N), intent(in) :: ZernMode
        real(8), dimension(N), intent(in) :: ZernCoef_
        integer,               intent(in) :: M, N

        CALL elt_srf_zrn_mode_set_impl(ok,iElt,ZernMode,ZernCoef_,M,N)
      end subroutine elt_srf_zrn_mode_set

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

        CALL elt_srf_zrn_get_impl(ok,lMon_,ZernType,ZernCoef_,ZernAnnularRatio,iElt,N)
      end subroutine elt_srf_zrn_get

      subroutine elt_srf_ff_any(any_elt_FF)

        implicit none
        logical, intent(out):: any_elt_FF

        CALL elt_srf_ff_any_impl(any_elt_FF)
      end subroutine elt_srf_ff_any

      subroutine elt_srf_ff_fnd(ok, nFF, iElt, N)

        implicit none
        logical,                     intent(out):: ok    ! success (1) or Fail (0)
        logical, dimension(N),       intent(out):: nFF   ! 1 (FreeForm at this elt) or 0
        integer, dimension(N),       intent(in) :: iElt  ! Elt ID: (0 < iElt[i] <= nElt)

        integer,                     intent(in) :: N

        CALL elt_srf_ff_fnd_impl(ok, nFF, iElt, N)
      end subroutine elt_srf_ff_fnd

      subroutine elt_srf_mon_zrn_max_modes(n_max)
        use elt_mod, only: mMonCoef

        implicit none
        integer, intent(out) :: n_max

        CALL elt_srf_mon_zrn_max_modes_impl(n_max)
      end subroutine elt_srf_mon_zrn_max_modes

      subroutine ray_info_get(OK, Pos, Dir, OPL, RayOK, RayPass, nRays)

        implicit none
        logical,                     intent(out):: OK
        real(8), dimension(3,nRays), intent(out):: Pos,Dir
        real(8), dimension(nRays),   intent(out):: OPL
        logical, dimension(nRays),   intent(out):: RayOK      ! successfully traced
        logical, dimension(nRays),   intent(out):: RayPass    ! not blocked
        integer,                     intent(in) :: nRays

        CALL ray_info_get_impl(OK, Pos, Dir, OPL, RayOK, RayPass, nRays)
      end subroutine ray_info_get

      subroutine ray_info_set(OK, Pos, Dir, OPL, RayOK, nRays)
        use smacos_vars_mod, only: npts

        implicit none
        logical,                     intent(out):: OK
        integer,                     intent(in) :: nRays
        real(8), dimension(3,nRays), intent(in) :: Pos, Dir
        real(8), dimension(nRays),   intent(in) :: OPL
        integer, dimension(nRays),   intent(in) :: RayOK

        CALL ray_info_set_impl(OK, Pos, Dir, OPL, RayOK, nRays)
      end subroutine ray_info_set

      subroutine trace_rays(OK, rms_WFE, nRays, N, iElt)

        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: rms_WFE
        integer, intent(out):: nRays, N
        integer, intent(in) :: iElt

        CALL trace_rays_impl(OK, rms_WFE, nRays, N, iElt)
      end subroutine trace_rays

      subroutine opd_val(OK, OPD, N)

        implicit none
        logical,                 intent(out):: OK
        real(8), dimension(N,N), intent(out):: OPD
        integer,                 intent(in) :: N      ! = nGridPts

        CALL opd_val_impl(OK, OPD, N)
      end subroutine opd_val

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

        CALL spot_cmd_impl(OK, nSpot, iElt, ref_csys, ref_pos, res_trace)
      end subroutine spot_cmd

      subroutine spot_get(OK, SPOT, shift, centroid, csys, N)
        use smacos_vars_mod, only: CntrSpot, RefSpot, xLocal,yLocal,zLocal  ! xcent, ycent

        implicit none
        logical,                 intent(out):: OK           ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), dimension(N,2), intent(out):: SPOT         ! ray-surface intersection points
        real(8), dimension(4),   intent(out):: shift        ! shift from projected Ref. position
        real(8), dimension(2),   intent(out):: centroid     ! centroid position
        real(8), dimension(3,3), intent(out):: csys         ! Coord. Frame orientation

        integer,                 intent(in) :: N            ! = nSpot = successfully traced rays

        CALL spot_get_impl(OK, SPOT, shift, centroid, csys, N)
      end subroutine spot_get

      subroutine int_cmd(OK, N, iElt, res_trace)
        implicit none
        logical, intent(out):: OK        ! (PASS=1) if successful; (FAIL=0) otherwise
        integer, intent(out):: N         ! intensity matrix size per side (= mdttl)
        integer, intent(in) :: iElt      ! element where intensity is to be computed
        logical, intent(in) :: res_trace ! (PASS=1) apply a MODIFY first; (FAIL=0) keep prior trace state

        CALL int_cmd_impl(OK, N, iElt, res_trace)
      end subroutine int_cmd

      subroutine int_get(OK, INT_OUT, N)
        use elt_mod, only: MWFFT

        implicit none
        logical,                 intent(out):: OK
        real(8), dimension(N,N), intent(out):: INT_OUT
        integer,                 intent(in) :: N      ! = mdttl

        CALL int_get_impl(OK, INT_OUT, N)
      end subroutine int_get

      subroutine cfield_cmd(OK, N, iElt, res_trace)
        implicit none
        logical, intent(out):: OK        ! (PASS=1) if successful
        integer, intent(out):: N         ! WFElt's leading dim (= mdttl)
        integer, intent(in) :: iElt      ! element at which complex field
                                         ! is wanted
        logical, intent(in) :: res_trace ! (PASS=1) apply a MODIFY first

        CALL cfield_cmd_impl(OK, N, iElt, res_trace)
      end subroutine cfield_cmd

      subroutine elt_dx_get(OK, dx_out, iElt)
        use elt_mod, only: dxElt, CBM

        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: dx_out
        integer, intent(in) :: iElt

        CALL elt_dx_get_impl(OK, dx_out, iElt)
      end subroutine elt_dx_get

      subroutine base_unit_to_metres(OK, cbm_out)
        use elt_mod, only: CBM

        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: cbm_out

        CALL base_unit_to_metres_impl(OK, cbm_out)
      end subroutine base_unit_to_metres

      subroutine cfield_get(OK, REAL_OUT, IMAG_OUT, N, iElt)
        use elt_mod, only: WFElt, iEltToiWF

        implicit none
        logical,                 intent(out):: OK
        real(8), dimension(N,N), intent(out):: REAL_OUT
        real(8), dimension(N,N), intent(out):: IMAG_OUT
        integer,                 intent(in) :: N      ! = mdttl
        integer,                 intent(in) :: iElt   ! element

        !f2py  intent(hide)         :: OK
        !f2py  intent(out,hide,copy):: REAL_OUT
        !f2py  intent(out,hide,copy):: IMAG_OUT

        integer :: iWF

        CALL cfield_get_impl(OK, REAL_OUT, IMAG_OUT, N, iElt)
      end subroutine cfield_get

      subroutine cfield_apodize(OK, MASK, N, iElt)
        use elt_mod, only: WFElt, iEltToiWF

        implicit none
        logical,                 intent(out):: OK
        real(8), dimension(N,N), intent(in) :: MASK
        integer,                 intent(in) :: N      ! = mdttl
        integer,                 intent(in) :: iElt

        integer :: iWF

        CALL cfield_apodize_impl(OK, MASK, N, iElt)
      end subroutine cfield_apodize

      subroutine cfield_apodize_complex(OK, MASK_RE, MASK_IM, N, iElt)
        use elt_mod, only: WFElt, iEltToiWF

        implicit none
        logical,                 intent(out):: OK
        real(8), dimension(N,N), intent(in) :: MASK_RE, MASK_IM
        integer,                 intent(in) :: N      ! = mdttl
        integer,                 intent(in) :: iElt

        integer :: iWF

        CALL cfield_apodize_complex_impl(OK, MASK_RE, MASK_IM, N, iElt)
      end subroutine cfield_apodize_complex

      subroutine perturb_elt(OK, iElt, th, del, useLocalCoord)

        implicit none
        logical, intent(out):: OK
        integer, intent(in) :: iElt
        real(8), intent(in) :: th(3), del(3)
        logical, intent(in) :: useLocalCoord

        CALL perturb_elt_impl(OK, iElt, th, del, useLocalCoord)
      end subroutine perturb_elt

      subroutine perturb_src(OK, th, del)

        implicit none
        logical, intent(out):: OK
        real(8), intent(in) :: th(3), del(3)

        CALL perturb_src_impl(OK, th, del)
      end subroutine perturb_src

      subroutine xp_set(ok, vpt, psi, rad)
        implicit none
        logical,               intent(out):: ok    ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), dimension(3), intent(in) :: vpt   ! (x,y,z) Srf. position     in global CSYS
        real(8), dimension(3), intent(in) :: psi   ! (L,M,N) Srf. orientation. in global CSYS
        real(8),               intent(in) :: rad   ! ref. sphere radius

        CALL xp_set_impl(ok, vpt, psi, rad)
      end subroutine xp_set

      subroutine xp_get(ok, vpt, psi, rad)
        implicit none
        logical,               intent(out) :: ok    ! (PASS=1) if successful; (FAIL=0) otherwise
        real(8), dimension(3), intent(out) :: vpt   ! (x,y,z) Srf. position     in global CSYS
        real(8), dimension(3), intent(out) :: psi   ! (L,M,N) Srf. orientation. in global CSYS
        real(8),               intent(out) :: rad   ! ref. sphere radius

        CALL xp_get_impl(ok, vpt, psi, rad)
      end subroutine xp_get

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

        CALL xp_fnd_impl(OK, XP, mode)
      end subroutine xp_fnd

      subroutine sxp_fnd(OK, XP, mode)
        use       macos_mod, only: ifStopSet
        use smacos_vars_mod, only: npts
        use         src_mod, only: nGridPts

        implicit none
        logical, intent(out):: OK
        real(8), intent(out):: XP(7)
        integer, intent(in) :: mode

        logical :: ifCentroidSave
        integer :: nGridPtsSave

        CALL sxp_fnd_impl(OK, XP, mode)
      end subroutine sxp_fnd

      subroutine ors_run(OK, iElt)

        implicit none
        logical, intent(out):: OK
        integer, intent(in) :: iElt

        CALL ors_run_impl(OK, iElt)
      end subroutine ors_run

      subroutine srs_run(OK, iSlv1, iSlv2, link)

        implicit none
        logical, intent(out):: OK
        integer, intent(in) :: iSlv1
        integer, intent(in) :: iSlv2
        logical, intent(in) :: link

        CALL srs_run_impl(OK, iSlv1, iSlv2, link)
      end subroutine srs_run

      subroutine stop_info_get(OK, iElt, VptOffset)
        use Kinds
        use smacosio_mod, only: StopOffset, EltStopSet
        use    macos_mod, only: ifStopSet

        implicit none
        logical, intent(out):: OK            ! (PASS=1) if successful; (FAIL=0) otherwise
        integer, intent(out):: iElt          ! Element at which Optical System Stop is defined
        real(8), intent(out):: VptOffset(2)  ! [dx,dy]: Offset from Srf. Vertex Position

        CALL stop_info_get_impl(OK, iElt, VptOffset)
      end subroutine stop_info_get

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

        CALL stop_info_set_impl(OK,iElt,VptOffset)
      end subroutine stop_info_set

      subroutine stop_obj_set(OK, x, y, z)
        use smacos_vars_mod, only: npts
        use       macos_mod, only: ifStopSet
        implicit none
        logical, intent(out):: OK
        real(8), intent(in) :: x, y, z

        CALL stop_obj_set_impl(OK, x, y, z)
      end subroutine stop_obj_set

      subroutine n_elt(nElt_out)
        implicit none
        integer, intent(out) :: nElt_out   ! returns # of Elements

        CALL n_elt_impl(nElt_out)
      end subroutine n_elt

      subroutine init(ok, modelsize)
        implicit none
        logical, intent(out) :: ok
        integer, intent(in) :: modelsize

        CALL init_impl(ok, modelsize)
      end subroutine init

      subroutine load_rx(ok, nElt_out, rx)
        implicit none
        logical,            intent(out):: ok
        integer,            intent(out):: nElt_out
        character(len=250), intent(in) :: rx

        CALL load_rx_impl(ok, nElt_out, rx)
      end subroutine load_rx

      subroutine save_rx(ok, rx)
        implicit none
        logical,          intent(out):: ok
        character(len=*), intent(in) :: rx

        CALL save_rx_impl(ok, rx)
      end subroutine save_rx

      subroutine translateSurfaceID(iElt)
        implicit none
        integer, intent(inout) :: iElt

        CALL translateSurfaceID_impl(iElt)
      end subroutine translateSurfaceID

      subroutine translateEltID(iElt, n, m)
        implicit none
        integer, dimension(n,m), intent(inout):: iElt
        integer,                 intent(in)   :: n, m

        CALL translateEltID_impl(iElt, n, m)
      end subroutine translateEltID

  end module api

