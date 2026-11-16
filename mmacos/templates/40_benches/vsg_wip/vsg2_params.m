function p = vsg2_params(opts)
%VSG2_PARAMS  Single source of truth for every VSG2 hardware number.
%   p = vsg2_params() returns a struct of all VSG2 optical/mechanical
%   parameters used by the IFO (Twyman-Green) and ZWFS models.  Every
%   number lives HERE and nowhere else in the vsg2 drivers -- change a
%   value once, and the whole model tracks it.
%
%   p = vsg2_params('dm','BMC', 'model_size',512, 'spot_id',9) overrides
%   selected fields (see the arguments block).
%
%   Units convention (mirrors the plan + mmacos/CLAUDE.md):
%     - The Rx prescriptions use BaseUnits = mm (p.base_units).
%     - User-facing driver surfaces take/return SI metres; convert with
%       1/macos.cbm() at the mmacos boundary.
%     - Fields tagged "(mm)" / "(m)" / "(nm)" / "(um)" state their unit.
%
%   Authoritative sources (see PLAN_VSG2_MODELS.md header):
%     - VSG2_Info.pptx (bench build)
%     - "VSG2 Zernike Wavefront Sensor Update -v2.pptx" (ZWFS upgrade
%       deck) -- the DM pupil diameter (D = N_act*pitch*sqrt(2)), F/#,
%       and the 9-spot etched-depth mask array all come from this deck.
%
%   TODO(Dave) markers flag placeholders still awaiting confirmation.

arguments
    opts.dm         (1,:) char {mustBeMember(opts.dm,{'AOX','BMC'})} = 'AOX'
    opts.model_size (1,1) double {mustBeInteger, mustBePositive}     = 256
    opts.spot_id    (1,1) double {mustBeInteger, mustBePositive}     = 9
    opts.stage      (1,:) char {mustBeMember(opts.stage,{'A','B'})}  = 'A'
end

% =====================================================================
%  0. Bookkeeping
% =====================================================================
p.base_units  = 'mm';          % Rx BaseUnits / WaveUnits
p.model_size  = opts.model_size;  % 256 default; 512 for fidelity runs
p.stage       = opts.stage;    % 'A' idealized unfolded; 'B' as-built

% =====================================================================
%  1. Source (frequency-stabilized HeNe via SM fiber)
% =====================================================================
p.lambda_nm   = 632.8;                 % HeNe (nm)
p.lambda_mm   = p.lambda_nm * 1e-6;    % (mm) -- Rx WaveUnits value
p.lambda_m    = p.lambda_nm * 1e-9;    % (m)  -- SI
% Finite-band generalization hook (single-line HeNe by default):
p.lambda_band_nm = p.lambda_nm;        % scalar -> monochromatic
% TODO(Dave): fiber NA / beam diameter at DM (sets overfill vs pupil).

% =====================================================================
%  2. Deformable mirror (the system pupil, test-arm optic)
% =====================================================================
switch opts.dm
    case 'AOX'      % Xinetics/AOX, 1.0 mm pitch, 48x48 (deck default)
        p.dm.name       = 'AOX 48x48';
        p.dm.n_act      = 48;
        p.dm.pitch_mm   = 1.0;
        p.dm.stroke_nm  = 200;         % +/- surface (nm); OPD = 2x
    case 'BMC'      % Boston Micromachines, 0.4 mm pitch, 96x96
        p.dm.name       = 'BMC 96x96';
        p.dm.n_act      = 96;
        p.dm.pitch_mm   = 0.4;
        p.dm.stroke_nm  = 200;         % TODO(Dave): confirm BMC stroke
end
% Illuminated pupil = circle circumscribing the square actuator grid
% (deck slides 5/16): beam fills the DIAGONAL, not the square edge.
p.dm.edge_mm     = p.dm.n_act * p.dm.pitch_mm;         % square edge (mm)
p.dm.diam_mm     = p.dm.edge_mm * sqrt(2);             % pupil D (mm)
p.dm.radius_mm   = p.dm.diam_mm / 2;
p.dm.stroke_mm   = p.dm.stroke_nm * 1e-6;              % surface (mm)
% DM grid model: an nGridMat FreeForm/GridData figure, zero at nominal.
% Match the engine grid to the actuator grid at Stage A (1 node/act);
% GridSrfdx spans (nGridMat-1)*dx.  n_act+1 nodes bracket n_act cells.
p.dm.grid_n      = p.dm.n_act + 1;                     % nodes across pupil
p.dm.grid_dx_mm  = p.dm.pitch_mm;                      % 1 node per actuator

% =====================================================================
%  3. Collimator L1 + imager L2 (Newport achromats)
% =====================================================================
% Layout figure mislabels L1 "700mm EFL"; part number + text = 750 mm.
p.L1.name    = 'Newport PAC097AR.14';
p.L1.efl_mm  = 750.0;
p.L1.diam_mm = 76.2;                    % 3 inch
p.L2.name    = 'Newport PAC095AR.14';
p.L2.efl_mm  = 250.0;
p.L2.diam_mm = 76.2;
% Stage-B doublet prescriptions (radii/glasses/thicknesses) are fetched
% from Newport then; Stage A uses ideal thin-lens surfaces.
% TODO(Dave)/Stage B: real PAC097/PAC095 radii + glasses.

% =====================================================================
%  4. Beam splitter (our selection: wedged 50/50 plate, s-pol)
% =====================================================================
% Twyman-Green: BOTH detected arms carry exactly one R and one T at the
% BS, so arm amplitudes balance for ANY split ratio; 50/50 maximizes
% throughput.  Stage A = zero-thickness plane, scalar amplitude factors.
p.bs.a_test  = 0.5;     % test-arm amplitude factor (|a|^2 = 0.25 intensity)
p.bs.a_ref   = 0.5;     % reference-arm amplitude factor
% Stage B: fused silica, ~10 mm thick, 30 arcmin wedge, dielectric 50/50
% @ 632.8nm/45 deg, AR second surface.  TODO(Dave)/Stage B: substrate.
p.bs.thickness_mm = 10.0;      % (Stage B)
p.bs.wedge_arcmin = 30.0;      % (Stage B)

% =====================================================================
%  5. Distances (mm) -- unfolded path lengths (Stage A)
% =====================================================================
% Nominal from the slide/thin-lens; arm split refined via pupil_quality.
p.dist.dm_to_L2_mm  = 1175.0;   % DM -> L2 (includes fold legs in reality)
p.dist.L2_to_cam_mm = 317.0;    % L2 -> camera (DM image conjugate)
p.dist.L2_to_foc_mm = p.L2.efl_mm;          % 250: internal source focus
p.dist.foc_to_cam_mm = p.dist.L2_to_cam_mm - p.dist.L2_to_foc_mm; % ~67
% Arm path lengths BS->DM and BS->RefMirr: settable INDEPENDENTLY (arm
% OPD matters for finite band).  Placeholders -- scale from VSG2_layout.png
% then refine (see plan Sec 4 arm-distance procedure).
% TODO(Dave): estimate BS split from the layout drawing.
p.dist.bs_to_dm_mm  = 300.0;    % PLACEHOLDER (scale from drawing)
p.dist.bs_to_ref_mm = 300.0;    % PLACEHOLDER (matched arm at Stage A)
p.dist.L1_to_bs_mm  = 200.0;    % PLACEHOLDER
p.dist.src_to_L1_mm = p.L1.efl_mm;          % collimator: source at focus

% =====================================================================
%  6. Camera (Andor NEO sCMOS)
% =====================================================================
p.cam.npix_x     = 2560;
p.cam.npix_y     = 2160;
p.cam.pixel_um   = 6.5;
p.cam.pixel_mm   = p.cam.pixel_um * 1e-3;
p.cam.binning    = 3;                       % 3x3 binning
p.cam.binpix_mm  = p.cam.pixel_mm * p.cam.binning;
% Magnification DM->camera (thin-lens): m = L2_to_cam / dm_to_L2.
p.cam.mag        = p.dist.L2_to_cam_mm / p.dist.dm_to_L2_mm;  % ~0.27
p.cam.scale_binpx_per_mm = 13.4;   % deck (CGI DM2 data), DM-plane scale
% TODO(Dave): camera flux / QE for photon-noise runs.

% =====================================================================
%  7. Optical scale derived at the focus
% =====================================================================
p.fnum   = p.L2.efl_mm / p.dm.diam_mm;      % ~3.57-3.7 for AOX
p.lamF_mm = p.lambda_mm * p.fnum;           % lambda*F/# (mm) = lambda/D unit

% =====================================================================
%  8. PZT phase shifter (reference-arm flat)
% =====================================================================
% Normal-incidence double pass: phase = 4*pi*Tz/lambda.
% Step size follows the PSI algorithm's per-frame phase alpha (NEVER
% hardcode nm): Tz = alpha*lambda/(4*pi).
p.pzt.algorithm = 'zygo13';     % default bench algorithm (13 frames)
switch p.pzt.algorithm
    case 'zygo13'
        p.pzt.alpha = pi/4;     % 45 deg/frame -> Tz = lambda/16 ~ 39.6 nm
        p.pzt.nframe = 13;
    case {'hariharan5','degroot7','4step'}
        p.pzt.alpha = pi/2;     % 90 deg/frame -> Tz = lambda/8
        p.pzt.nframe = struct('hariharan5',5,'degroot7',7,'4step',4).(p.pzt.algorithm);
end
p.pzt.step_mm = p.pzt.alpha * p.lambda_mm / (4*pi);   % Tz per frame (mm)
p.pzt.analytic = false;         % true = multiply E_r by exp(i*delta) (x-check)

% =====================================================================
%  9. ZWFS mask (transmissive etched substrate; 9-spot array)
% =====================================================================
% Deck slide 6: 3x3 array on one 25.4 mm FS substrate, 5 mm pitch, all
% etched to the SAME depth (346.2 nm -> pi/2 at HeNe).  "-" = no dimple.
p.zwfs.substrate    = 'Thorlabs W4101FT1';  % 1 mm FS, one-side AR
p.zwfs.subst_mm     = 1.0;
p.zwfs.n_fs         = 1.45702;              % FS index @ 632.8 nm (deck)
p.zwfs.etch_nm      = 346.2;                % physical etch depth (nm)
p.zwfs.etch_mm      = p.zwfs.etch_nm * 1e-6;
% Phase from etch depth (NOT assumed pi/2): phase = 2*pi*(n-1)*t/lambda.
% pi/2 by construction at 632.8 nm; chromatic elsewhere.
p.zwfs.phase_rad    = 2*pi * (p.zwfs.n_fs - 1) * p.zwfs.etch_mm / p.lambda_mm;
% Spot table (spot_id -> dimple diameter).  NaN = clear (no dimple).
p.zwfs.spot_lamD = [3.0, 2.0, NaN, 2.0, NaN, 1.0, NaN, 1.22, 1.06];
p.zwfs.spot_um   = [6.78,4.52, NaN,4.52, NaN,2.26, NaN,2.76, 2.40];
p.zwfs.spot_id   = opts.spot_id;            % selected spot (default 9)
p.zwfs.dimple_lamD = p.zwfs.spot_lamD(opts.spot_id);
p.zwfs.dimple_dia_mm = p.zwfs.dimple_lamD * p.lamF_mm;   % if a real dimple
p.zwfs.leakage_eps = 0.0;                   % ref-blocker amplitude leakage
% Substrate bias (common to both modes): ~1.7 nm rms SA + 0.31 mm focus
% shift (deck slide 5).  Stage A ignores; Stage B adds as a bias term.
p.zwfs.subst_sa_nm  = 1.7;
p.zwfs.subst_focus_shift_mm = 0.31;

% =====================================================================
% 10. Mode-switch hardware (deck; realizes the ref-blocking discussion)
% =====================================================================
p.hw.mask_stage  = 'Xeryon XYZ (5 nm res)';  % positions the mask
p.hw.ref_blocker = 'Aerotech ATS-100';       % blocks ref arm in ZWFS mode

end
