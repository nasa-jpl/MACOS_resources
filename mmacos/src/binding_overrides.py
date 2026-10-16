"""binding_overrides.py -- per-routine inputs to gen_mex_wrappers.py's f2py target.

This is generator INPUT (hand-maintained), not output.  Extracted once
(2026-10-16, BRIEF_to_bindings A.2) from the then hand-written
pymacos_f2py.f90 so the generated layer keeps every Python signature
pymacos had; new routines need no entry (the default rules apply:
intent(inout) -> intent(in,out), i.e. read AND returned, as the mex does).
"""

# !f2py lines, verbatim, emitted after the wrapper's declarations.
F2PY_HINTS = {'calib_run': ['!f2py integer intent(hide), depend(old_wfe_out):: max_fov=shape(old_wfe_out,0), '
               'max_wl=shape(old_wfe_out,1)'],
 'calib_set_beam_pos_fov': ['!f2py integer intent(hide), depend(pos):: n=shape(pos,1)'],
 'calib_set_target': ['!f2py integer intent(hide), depend(wf_zern_modes):: '
                      'n_wf_zern=len(wf_zern_modes)'],
 'calib_set_var_elt': ['!f2py integer intent(hide), depend(zern_modes):: n_zern=len(zern_modes)'],
 'cfield_get': ['!f2py  intent(hide)         :: OK',
                '!f2py  intent(out,hide,copy):: REAL_OUT',
                '!f2py  intent(out,hide,copy):: IMAG_OUT'],
 'cfield_plane_get': ['!f2py  intent(hide)         :: OK',
                      '!f2py  intent(out,hide,copy):: REAL_OUT',
                      '!f2py  intent(out,hide,copy):: IMAG_OUT'],
 'coat_get': ['!f2py  intent(out,hide,copy):: IndRef_',
              '!f2py  intent(out,hide,copy):: Extinc_',
              '!f2py  intent(out,hide,copy):: Thk_'],
 'elt_asph_get': ['!f2py integer intent(in) :: n'],
 'elt_csys_get': ['!f2py   integer intent(hide), depend(iElt):: N=len(iElt)'],
 'elt_csys_set': ['!f2py  integer intent(hide), depend(iElt), check(len(iElt)>0), '
                  'check(len(xDir)==3, len(yDir)==3, len(zDir)==3) :: m=len(iElt)'],
 'elt_psi': ['!f2py integer intent(hide), depend(iElt,Psi), check(len(iElt)==shape(Psi,1), '
             'shape(Psi,0)==3) :: n=len(iElt)'],
 'elt_srf_csys': ['!f2py  integer intent(hide),depend(iElt) :: N=len(iElt)'],
 'elt_srf_csys_dir': ['!f2py  integer intent(hide),depend(iElt) :: N=len(iElt)'],
 'elt_srf_csys_get': ['!f2py  integer intent(hide),depend(iElt) :: N=len(iElt)  ! '
                      ',check(shape(pMon_,1)==shape(xMon_,1),shape(pMon_,1)==shape(yMon_,1),shape(pMon_,1)==shape(zMon_,1),len(iElt)==shape(pMon_,1))'],
 'elt_srf_csys_pos': ['!f2py integer intent(hide),depend(iElt) :: N=len(iElt)    ! '
                      'check(shape(pMon_,0)==3, shape(pMon_,1)==len(iElt))'],
 'elt_srf_csys_set': ['!f2py  integer intent(hide), depend(iElt) :: N=len(iElt)'],
 'elt_srf_ff_zrn_coef': ['!f2py   integer intent(hide), depend(ZernMode,FFZernCoef_), '
                         'check(len(ZernMode)==len(FFZernCoef_), len(ZernMode)>0) :: '
                         'N=len(ZernMode)'],
 'elt_srf_grating_order': ['!f2py integer intent(hide), depend(iElt,Order), '
                           'check(len(iElt)==len(Order)) :: N=len(iElt)'],
 'elt_srf_grid_data_scale': ['!f2py   integer intent(hide), depend(iElt,GridSrfdx_):: N=len(iElt)'],
 'elt_srf_grid_spacing': ['!f2py   integer intent(hide), depend(iElt,GridSrfdx_):: N=len(iElt)'],
 'elt_srf_mon_zrn_coef': ['!f2py   integer intent(hide), depend(ZernMode,MonZernCoef_), '
                          'check(len(ZernMode)==len(MonZernCoef_), len(ZernMode)>0) :: '
                          'N=len(ZernMode)'],
 'elt_srf_zrn_FreeForm': ['!f2py  integer intent(hide), depend(GridMat_), '
                          'check(shape(GridMat_,0)>0, shape(GridMat_,1)>0):: Ny=shape(GridMat_,0), '
                          'Nx=shape(GridMat_,1)',
                          '!f2py  integer intent(hide), depend(FFZernModes_,FFZernCoef_):: '
                          'nFF=len(FFZernModes_)',
                          '!f2py  integer intent(hide), depend(MonZernModes_,MonZernCoef_):: '
                          'nMon=len(MonZernModes_)'],
 'elt_srf_zrn_coef': ['!f2py   integer intent(hide), depend(ZernMode,ZernCoef_), '
                      'check(len(ZernMode)==len(ZernCoef_), len(ZernMode)>0) :: N=len(ZernMode)'],
 'elt_srf_zrn_set': ['!f2py   integer intent(hide), depend(iElt),      check(len(iElt)>0)     :: '
                     'M=len(iElt)',
                     '!f2py   integer intent(hide), depend(ZernMode),  check(len(ZernMode)>0) :: '
                     'Nm=len(ZernMode)',
                     '!f2py   integer intent(hide), depend(ZernCoef_), check(len(ZernCoef_)>0):: '
                     'Nc=len(ZernCoef_)'],
 'prb_elt': ['!f2py  integer intent(hide), depend(iElt) :: n=len(iElt)'],
 'prb_elt_grp': ['!f2py  integer intent(hide), depend(iElt) :: n=len(iElt)'],
 'rayfield_get': ['!f2py  intent(out,hide,copy):: ExRe',
                  '!f2py  intent(out,hide,copy):: ExIm',
                  '!f2py  intent(out,hide,copy):: EyRe',
                  '!f2py  intent(out,hide,copy):: EyIm',
                  '!f2py  intent(out,hide,copy):: EzRe',
                  '!f2py  intent(out,hide,copy):: EzIm',
                  '!f2py  intent(out,hide,copy):: kx',
                  '!f2py  intent(out,hide,copy):: ky',
                  '!f2py  intent(out,hide,copy):: kz',
                  '!f2py  intent(out,hide,copy):: nx',
                  '!f2py  intent(out,hide,copy):: ny',
                  '!f2py  intent(out,hide,copy):: nz',
                  '!f2py  intent(out,hide,copy):: status'],
 'set_src_info': ['!f2py  intent(hide):: OK'],
 'sys_units': ['!f2py  intent(hide):: OK,BaseUnitID,WaveUnitID']}

# Routines whose f2py-facing interface predates the generator and keeps
# in-place intent(inout) semantics (no default intent(in,out) rewrite).
# Every routine bound in the hand-written layer is listed.
F2PY_LEGACY = set(['base_unit_to_metres', 'beam_get', 'beam_set', 'calib_buffer_dims', 'calib_clear_var_elts',
 'calib_run', 'calib_set_beam', 'calib_set_beam_pos_fov', 'calib_set_beam_wt', 'calib_set_iter',
 'calib_set_target', 'calib_set_tol', 'calib_set_var_elt', 'calib_stats_get', 'cfield_apodize',
 'cfield_apodize_complex', 'cfield_cmd', 'cfield_get', 'cfield_plane_get', 'coat_get', 'coat_set',
 'compose_add', 'compose_get', 'compose_start', 'elt_asph_get', 'elt_csys_get', 'elt_csys_rm',
 'elt_csys_set', 'elt_dx_get', 'elt_grp_any', 'elt_grp_del', 'elt_grp_del_all', 'elt_grp_fnd',
 'elt_grp_get', 'elt_grp_max', 'elt_grp_max_all', 'elt_grp_set', 'elt_kc', 'elt_kr', 'elt_psi',
 'elt_rpt', 'elt_srf_csys', 'elt_srf_csys_dir', 'elt_srf_csys_get', 'elt_srf_csys_pos',
 'elt_srf_csys_set', 'elt_srf_ff_any', 'elt_srf_ff_fnd', 'elt_srf_ff_zrn_coef',
 'elt_srf_grating_any', 'elt_srf_grating_fnd', 'elt_srf_grating_order', 'elt_srf_grating_params',
 'elt_srf_grating_rule_dir', 'elt_srf_grating_rule_width', 'elt_srf_grating_type',
 'elt_srf_grid_any', 'elt_srf_grid_data', 'elt_srf_grid_data_add', 'elt_srf_grid_data_scale',
 'elt_srf_grid_fnd', 'elt_srf_grid_fnd_type', 'elt_srf_grid_size', 'elt_srf_grid_size_max',
 'elt_srf_grid_spacing', 'elt_srf_mon_zrn_coef', 'elt_srf_mon_zrn_max_modes',
 'elt_srf_zrn_FreeForm', 'elt_srf_zrn_any', 'elt_srf_zrn_coef', 'elt_srf_zrn_fnd',
 'elt_srf_zrn_get', 'elt_srf_zrn_mode_set', 'elt_srf_zrn_norm_radius', 'elt_srf_zrn_set',
 'elt_srf_zrn_type', 'elt_vpt', 'ffcut_get', 'ffcut_set', 'ffp', 'get_src_csys', 'get_src_fov',
 'get_src_sampling', 'get_src_size', 'grid_dropped_get', 'grid_idx_ovf_get', 'init', 'int_cmd',
 'int_get', 'load_rx', 'model_size_get', 'modes_get', 'modified_rx', 'n_elt', 'obs_set',
 'opd_mask_get', 'opd_ref_get', 'opd_ref_set', 'opd_val', 'ors_run', 'perturb_elt', 'perturb_src',
 'pfp', 'pol_get', 'pol_set', 'prb_elt', 'prb_elt_grp', 'ray_info_get', 'ray_info_set',
 'rayfield_get', 'save_rx', 'set_src_csys', 'set_src_fov', 'set_src_info', 'set_src_sampling',
 'set_src_size', 'spot_cmd', 'spot_get', 'src_finite', 'src_flux', 'src_info', 'src_size',
 'src_wvl', 'srs_run', 'stop_info_get', 'stop_info_set', 'stop_obj_set', 'sxp_fnd', 'sys_units',
 'sysprop', 'trace_rays', 'translateEltID', 'translateSurfaceID', 'vecdif_set', 'window_off',
 'window_set', 'xp_fnd', 'xp_get', 'xp_set'])

# Dimension rewrites for the f2py declaration (f2py must derive the size
# from the array it is handed; module parameters are resolved to literals
# and the generator asserts the literal against the module source).
F2PY_DIMS = {
    'calib_set_var_elt': {'dofs': ['8'], 'zern_modes': ['n_zern']},
    'calib_set_target': {'wf_zern_modes': ['n_wf_zern']},
    'calib_set_beam_pos_fov': {'pos': ['3', 'n']},
}
# (module, parameter, literal) asserted at generation time.
F2PY_PARAM_ASSERTS = [('dopt_mod', 'mVarDOF', '8')]

# Wrappers emitted verbatim (the interface differs from the api routine).
F2PY_VERBATIM = {
    'calib_run': '      subroutine calib_run(OK, rtn_flag_out, nFov_out, nWl_out,            &\n                           old_wfe_out, new_wfe_out, max_fov, max_wl)\n\n        implicit none\n        logical, intent(out)  :: OK\n        integer, intent(out)  :: rtn_flag_out\n        integer, intent(out)  :: nFov_out\n        integer, intent(out)  :: nWl_out\n        integer, intent(in)   :: max_fov, max_wl\n        real(8), intent(inout):: old_wfe_out(max_fov,max_wl)\n        real(8), intent(inout):: new_wfe_out(max_fov,max_wl)\n        !f2py integer intent(hide), depend(old_wfe_out):: max_fov=shape(old_wfe_out,0), max_wl=shape(old_wfe_out,1)\n\n        CALL calib_run_impl(OK, rtn_flag_out, nFov_out, nWl_out,           &\n                            old_wfe_out, new_wfe_out)\n      end subroutine calib_run\n',
}

# f2py-only routines (no api counterpart), emitted verbatim at the end.
F2PY_EXTRA = ["      ! Subroutine wrapper around the model-size FUNCTION so f2py exposes\n      ! it (the bare function is imported but never wrapped, so macos.py's\n      ! model_size() -> lib.api.currrent_macos_model_size AttributeErrors).\n      subroutine model_size_get(N)\n        implicit none\n        integer, intent(out):: N\n\n        N = currrent_macos_model_size()\n      end subroutine model_size_get\n"]
