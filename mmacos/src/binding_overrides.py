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

# --- One default per option (BRIEF_to_bindings A.6, Dave 2026-10-10) ------------
# The generator's `defaults` target writes cmdref/25_defaults.md from this list.
# Every `engine` / `mmacos` / `pymacos` cell that is a (file, regex) pair is READ
# from that source at generation time (group 1 = the default as written); a
# pattern that no longer matches stops the generator (fail closed), so the table
# cannot keep describing a default the code no longer has.  Paths: engine under
# macos/macos_f90, mmacos under mmacos/src/+macos, pymacos under pymacos/src/pymacos.
# `get` / `set` must name macos_api_mod routines (checked against the parse).
# `status`: 'inherits' (the veneer reads the engine), 'agrees' (restated, same
# value, no api getter to inherit from), 'fixed' (a restatement removed by A.6),
# 'Dave' (a judgment call, routed with the measurement).
OPTIONS = [
    dict(option='Session modes: POLARIZATION, VECTOR/SCALAR, CENTROID, LNEG, CIR',
         engine=('macos_mod.F', r'^\s*ifCentroid\s*=\s*(\.\w+\.)'),
         engine_note='ifPol/ifVecDif3/ifCentroid/ifPGColor .FALSE., ifLNegOK .TRUE., '
                     'at allocation (macos_mod_init); survive LOAD',
         get=['modes_get'], set=['pol_set', 'vecdif_set'],
         cli='POLARIZATION, VECTOR/SCALAR, CENTROID/CHIEFRAY, LNEG/NOLNEG, CIR/GIR; '
             'LOAD names the non-default ones',
         mmacos='`macos.modes()`; `load_rx` warns `mmacos:modesInForce`',
         pymacos='`modes()`; `load` warns (RuntimeWarning)',
         override='`polarization(...)`; CENTROID/LNEG/CIR have no api setter (CLI only)',
         status='inherits'),
    dict(option='OPD reference (chief ray / aperture mean)',
         engine=('traceutil_mod.F', r'Logical\s*::\s*LOpdRefChief\s*=\s*(\.\w+\.)'),
         engine_note='session LOpdRefChief (chief), at allocation; per deck '
                     '`UseChfRay4OPD= Y|N`; per loaded deck `opd_ref_set`',
         get=['opd_ref_get'], set=['opd_ref_set'],
         cli='OPDREF CHIEF|MEAN (session); `UseChfRay4OPD=` (deck)',
         mmacos=('dw_dx.m', r"opts\.opd_ref \(1,:\) char \{.*\} = '(\w*)'"),
         mmacos_note='`opd()` / `opd_ref()` inherit; the dw_* harvests\' `opd_ref`: '
                     'the deck\'s own as loaded (was `\'chief\'`, re-applied after the '
                     'reload: it overrode `UseChfRay4OPD= N` and left the session on chief)',
         pymacos='`opd()` / `opd_ref()` inherit (no harvest veneer)',
         override='`opd_ref(\'mean\'|\'chief\')` for the loaded deck; a harvest\'s `opd_ref`',
         status='fixed'),
    dict(option='FEX / SXP axis (chief ray / beam centroid)',
         engine=('macos_mod.F', r'^\s*ifCentroid\s*=\s*(\.\w+\.)'),
         engine_note='session mode ifCentroid (.FALSE. = chief), at allocation',
         get=['modes_get'], set=['xp_fnd', 'sxp_fnd'],
         cli='CENTROID / CHIEFRAY, then FEX / SXP',
         mmacos=('fex.m', r'^\s*mode double \{.*\} = (\S+)'),
         mmacos_note='`fex(mode)`, `sxp(mode)`: empty = the session axis read through '
                     '`modes_get` (was a fixed 1; `sxp` also refused 0)',
         pymacos=('macos.py', r'^def fex\(mode: int \| None = (\w+)\)'),
         pymacos_note='`fex(mode)`, `sxp(mode)`: None = the session axis (was a fixed 1)',
         override='`fex(0)` / `fex(1)` per call',
         status='fixed'),
    dict(option='SPOT centre (element vertex / chief ray)',
         engine=('macos_init.inc', r'^\s*spcOption\s*=\s*(\d+)'),
         engine_note='spcOption (1 = element vertex), at allocation',
         get=['spc_option_get'], set=['spot_cmd'],
         cli='SPCENTER ELT|CHFRAY',
         mmacos=('spot.m', r"opts\.at\s+\(1,:\) char \{.*\} = '(\w*)'"),
         mmacos_note='`spot(srf, at=)`: empty = the session centre (`spc_option_get`)',
         pymacos=('macos.py', r'^\s+vpt_center: bool \| int \| None = (\w+),'),
         pymacos_note='`spot(srf, vpt_center=)`: None = the session centre',
         override='per call (`at=\'chief\'` / `vpt_center=False`); `spot_cmd` '
                  'restores the session option after the call',
         status='inherits'),
    dict(option='Obscuration option (OBS ALL / POSITIVE / NEGATIVE)',
         engine=('macos_init.inc', r'^\s*iObsOpt\s*=\s*(\d+)'),
         engine_note='iObsOpt (1 = POSITIVE: unobscured rays), at allocation',
         get=['obs_opt_get'], set=['obs_set'],
         cli='OBS',
         mmacos='`obs()` queries (`obs_opt_get`), `obs(opt)` sets',
         pymacos=('macos.py', r"^def obs_set\(option: str \| None = (\w+)\)"),
         pymacos_note='`obs_set()` queries (was: SET positive), `obs_set(opt)` sets',
         override='`obs(\'all\'|\'negative\')` / `obs_set(...)`',
         status='inherits'),
    dict(option='Far-field evanescent cut',
         engine=('dftsub.F', r'LOGICAL\s*::\s*ifFFEvanCut\s*=\s*(\.\w+\.)'),
         engine_note='dft_mod ifFFEvanCut (off), at allocation',
         get=['ffcut_get'], set=['ffcut_set'],
         cli='(none: api only)',
         mmacos='`ffcut()` queries, `ffcut(on)` sets',
         pymacos='`ffcut()` queries, `ffcut(on)` sets',
         override='`ffcut(true)`',
         status='inherits'),
    dict(option='OPD / mask array orientation',
         engine=None, engine_note='no engine state: the raw array is [i = global X, j = global Y]',
         get=['opd_val', 'opd_mask_get'], set=[],
         cli='(OPD writes the raw array)',
         mmacos=('opd.m', r"opts\.orient \(1,:\) char \{.*\} = '(\w+)'"),
         mmacos_note='`opd(orient=)`, `opd_mask(orient=)`',
         pymacos=('macos.py', r'^def opd\(orient: str = "(\w+)"'),
         pymacos_note='`opd(orient=)`, `opd_mask(orient=)`',
         override='`orient=\'xy\'` (the transpose)',
         status='agrees'),
    dict(option='Valid-pixel mask (opdValid)',
         engine=None, engine_note='opdValid from the last OPD trace (zero is NOT "no ray")',
         get=['opd_mask_get'], set=[],
         cli='(OPD)', mmacos='`opd_mask()`', pymacos='`opd_mask()`',
         override='--', status='inherits'),
    dict(option='remove_ptt in the sensitivity harvests',
         engine=None, engine_note='no engine state',
         get=[], set=[],
         cli='--',
         mmacos=('dw_dsurf.m', r'opts\.remove_ptt \(1,1\) logical = (\w+)'),
         mmacos_note='`dw_dsurf`, `dw_dsurf_multi`',
         pymacos='(no harvest veneer; tests/sensitivities drivers)',
         override='`remove_ptt=true`', status='agrees'),
    dict(option='Model size',
         engine=('macos_mod.F', r'default_problem_size\s*=\s*(\d+)'),
         engine_note='the CLI\'s start-up prompt: Return = this default; SMACOS '
                     'with no size: 256 too (param_mod)',
         get=[], set=['init'],
         cli='prompted at start-up; Return = 256',
         mmacos=('init.m', r'^\s*model_size \(1,1\) double \{.*\} = (\d+)'),
         mmacos_note='`init(model_size)` default',
         pymacos=('macos.py', r'^def init\(model_size: int = (\d+)\)'),
         pymacos_note='`init(model_size=)` default',
         override='`init(N)`',
         status='agrees'),   # Dave 2026-10-16: 256 everywhere (pymacos was 512, mmacos required)
    dict(option='Polarization source state on POLARIZATION ON',
         engine=None, engine_note='Ex0/Ey0 = 0 (unset) until set; `pol_get` reads it',
         get=['pol_get'], set=['pol_set'],
         cli='POLARIZATION prompts for Ex, Ey',
         mmacos=('polarization.m', r'opts\.Ex double = (\[[^\]]*\])'),
         mmacos_note='`polarization(\'on\', Ex=, Ey=)` Ex default',
         pymacos=('macos.py', r'Ex: complex \| Tuple\[float, float\] = (\([^)]*\))'),
         pymacos_note='`polarization(\'on\', Ex=, Ey=)` Ex default',
         override='`Ex=`, `Ey=`', status='agrees'),
]

# --- The cmdref BINDING column (BRIEF_to_bindings B.1) --------------------------
# Every CLI command of the cmdref (macos_help.inc entries, keyed by the entry's
# FIRST name as HELP writes it) is either detected as an 'SMACOS command' -- an
# api routine issues it through SMACOS (`command = '...'` in macos_api_mod.F90) --
# or classified HERE.  An entry in neither stops binding_classes() (fail closed).
# Classes: 'direct setter' (name the api routines), 'CLI-only' (why),
# 'CLI-only, legacy' (Dave 2026-10-10: the linear model and the differential
# commands, superseded by the dw_dx route), 'missing' (the B.2 worklist group).
_LEGACY_LIN = ('CLI-only, legacy', 'the internal linear model, superseded by the '
               'dw_dx route (harvest through the api, build the model outside); the '
               'analytic sensitivity forms cover only some surfaces (Dave 2026-10-10)')
_LEGACY_DIFF = ('CLI-only, legacy', 'differential FD inside the CLI; the api route '
                '(dw_dx and family over perturb_elt + opd_val) does it with the step, '
                'reference and mask under the caller\'s control (Dave 2026-10-10)')
_SESSION = ('CLI-only', 'session / shell convenience')
_PLOT = ('CLI-only', 'plot style (the bindings return the arrays)')
_FILE = ('CLI-only', 'CLI file-output mode (the bindings return the arrays)')
CLI_BINDINGS = {
    # Session & Files
    'Quit': _SESSION, 'HELP': _SESSION, 'STatus': _SESSION, 'SUMmarize': _SESSION,
    'EXEcute': _SESSION, 'JOUrnal': _SESSION, 'ECHo': _SESSION, '!': _SESSION,
    'PWD': _SESSION, 'RX': _SESSION, 'VI': _SESSION,
    'REset': ('direct setter', '`init` (re-allocation restores every startup default)'),
    'MREset': ('direct setter', '`init(model_size)`'),
    'ELTS': ('missing', 'Zernike / analysis set (B.2, second)'),
    # Prescription I/O
    'NEW': ('CLI-only', 'interactive element-entry dialog'),
    'FID': ('missing', 'field / beam set'),
    'VALidate': ('missing', 'the validator (validate_prescription.F90) has no api entry'),
    'EXPort': _LEGACY_LIN,
    'SHOw': ('direct setter', 'the `elt_*` getters (`elt_info_get`, `elt_obs_get`, '
                              '`elt_vpt`, `elt_psi`, `elt_kc`, `elt_kr`, ...)'),
    # Source & Wavelength
    'CHIefray': ('direct setter', '`xp_fnd(mode=1)` / `sxp_fnd(1)` per call; the '
                                  'session axis is read by `modes_get`'),
    'WLENS': ('missing', 'field / beam set'), 'SWL': ('missing', 'field / beam set'),
    'MULtispec': ('missing', 'through `compose_*` (B.2, image-plane set)'),
    'NFIlt': ('missing', 'image-plane post-processing (B.2, third)'),
    'ATMosphere': ('missing', 'image-plane post-processing (B.2, third)'),
    'SETC': ('direct setter', '`calib_set_iter`, `calib_set_tol`, `calib_set_*`'),
    'SAOpt': ('missing', 'source-adjustment mode'),
    # Ray Tracing
    # RAY is an SMACOS command (ray_trace, B.2 2026-10-17): the entry classifies itself.
    # RRAY / PRAY are SMACOS commands (ref_rays, B.2 2026-10-17): the entries classify themselves.
    'TPR': ('missing', 'image-plane post-processing (B.2, third)'),
    'MAP': ('missing', 'Zernike / analysis set (B.2, second)'),
    'COOrd': ('missing', 'beam coordinates at an element (the frames alone are '
                         '`elt_csys_get`)'),
    'ACOor': ('missing', 'all-element coordinates to xyzLocal.txt'),
    'SYSprop': ('direct setter', '`sysprop`'),
    'BWK': ('missing', 'chief-ray walk'),
    'FSR': ('missing', 'image-plane post-processing (B.2, third)'),
    'CENter': ('missing', 'STOP\'s sibling (center the beam in the stop); no '
                          'verified api twin'),
    'CENTRoid': ('direct setter', '`xp_fnd(mode=0)` / `sxp_fnd(0)` per call; the '
                                  'session axis is read by `modes_get`'),
    'FDP': ('missing', 'image-plane post-processing (B.2, third)'),
    # Behavior Toggles
    'LNEg/NOLNeg': ('missing', 'no api setter; read by `modes_get` (a session '
                               'mode -- NOT the linear model, whatever its L)'),
    'REGrid/NOREGrid': ('missing', 'a per-element resampling flag'),
    'NONe': _PLOT,
    # Surface Data
    'SINt': ('missing', 'single-ray tracing (B.2, first)'),
    'UDSinit': ('missing', 'builder'),
    # Perturbation
    'PREad': ('direct setter', '`prb_elt` / `perturb_elt` (the file is read in the caller)'),
    # Linear Model
    'BUild': _LEGACY_LIN, 'DMBuild': _LEGACY_LIN, 'PARtials': _LEGACY_LIN,
    'LPErturb': _LEGACY_LIN, 'LPRead': _LEGACY_LIN, 'LREset': _LEGACY_LIN,
    'LSPot': _LEGACY_LIN, 'LPIxilate': _LEGACY_LIN,
    # Diffraction
    'PROpagate': ('direct setter', 'not a command: `int_cmd` / `cfield_cmd` propagate'),
    'CADD': ('missing', 'through `compose_*` (B.2, image-plane set)'),
    'NOIse': ('missing', 'image-plane post-processing (B.2, third)'),
    'STRetch': _PLOT,
    # Outputs
    'OPDREF': ('direct setter', '`opd_ref_set` / `opd_ref_get` (the loaded deck)'),
    'ZABerr': ('missing', 'Zernike / analysis set (B.2, second)'),
    'ZCOef': ('missing', 'Zernike / analysis set (B.2, second)'),
    'LOS': ('missing', 'line of sight (the harvests\' `compute_los` reads it from spots)'),
    'METcalc': ('direct setter', '`met_calc`, `met_get`, `met_geom_get`'),
    'PIXel': ('direct setter', 'the INT block: `int_cmd` / `int_get`'),
    'AMPlitude': ('direct setter', 'the INT block: `cfield_cmd` / `cfield_get`'),
    'PHAse': ('direct setter', 'the INT block: `cfield_cmd` / `cfield_get`'),
    'REAl': ('direct setter', 'the INT block: `cfield_cmd` / `cfield_get`'),
    'LOG': ('direct setter', 'the INT block: `int_cmd` / `int_get`'),
    'MTF': ('missing', 'image-plane post-processing (B.2, third)'),
    'IMG': ('direct setter', 'the INT block: `int_cmd` / `int_get`'),
    # Plot Style
    'GRay': _PLOT, 'WIRe': _PLOT, 'SLIce': _PLOT, 'COLumn': _PLOT, 'CONtour': _PLOT,
    'IMGmode': _PLOT, 'CIR': _PLOT, 'PGP': _PLOT,
    # File Output
    'TEXt': _FILE, 'BINary': _FILE, 'FITs': _FILE, 'MAT': _FILE,
    'GETMatlabmatrix': ('CLI-only', 'disabled (a no-op stub); the bindings return arrays'),
    # Window & Post-Processing
    'PLOcate': ('missing', 'plot pixel-array location'),
    'PWIn': ('CLI-only', 'external plot GUI'),
    'SZCo': ('missing', 'Zernike / analysis set (B.2, second)'),
    'GBS': ('missing', 'field / beam set'),
    'BLUr': ('missing', 'image-plane post-processing (B.2, third)'),
    'GAIn': ('missing', 'image-plane post-processing (B.2, third)'),
    'ODRaw': _PLOT,
    'ROW': ('missing', 'image-plane post-processing (B.2, third)'),
    # System Optimization
    'AVAR': _LEGACY_DIFF, 'MVAR': _LEGACY_DIFF, 'DVAR': _LEGACY_DIFF, 'DFOV': _LEGACY_DIFF,
    'VARS': ('missing', 'lists the CALIB variables; `calib_set_var_elt` sets them, '
                        'no getter'),
    'AFOV': ('missing', 'field / beam set'), 'FOVS': ('missing', 'field / beam set'),
    'SFOV': ('missing', 'field / beam set'),
    # Misc / Debug
    # 'SRAy, SRT, SRTrace' is one HELP entry; SRT is an SMACOS command (srt_trace,
    # B.2 2026-10-17), so the entry classifies itself -- SRAy / SRTrace stay unbound.
    'DOPD': _LEGACY_DIFF,
    'JWST_v3d': ('missing', 'builder; the Vis3d ray history is `ray_hist_set` / '
                            '`ray_pos_hist_get`'),
}
