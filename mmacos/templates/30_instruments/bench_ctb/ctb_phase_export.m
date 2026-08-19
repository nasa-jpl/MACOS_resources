function out = ctb_phase_export(opts)
%CTB_PHASE_EXPORT  Export the CTB full model's per-plane phase factors for
%   an EXTERNAL PROPER user (roadmap item 5, deck_ctb "Next").
%
%   Runs the FULL surface-to-surface model (ctb_s2s_dcr.in, 44 elts) at
%   N=1024, 500 nm, and writes a self-describing .mat that lets a PROPER
%   user (a) consume this model's fields/surfaces plane by plane as an
%   INTERFACE CHECK, and (b) either REPLAY every inter-optic leg in their
%   own PROPER run, or IGNORE the s2s legs (collapsed jumps, compact-model
%   style) using our exported field as the hand-off.  See proper_ctb_check.m
%   for the consumer and README.md ("Phase-factor export") for the format.
%
%   The export carries NO mmacos dependency downstream: proper_ctb_check.m
%   reads ONLY the .mat.  Everything here is a runtime engine query on the
%   as-committed deck (no engine edit; if any gap surfaces the run STOPs).
%
%   CONVENTIONS (also stamped in the .mat meta):
%     - Fields are complex E on each plane's OWN grid; |E|^2 == intensity.
%     - OPD_m = -angle(E) * lambda/(2*pi)  -- the macos->PROPER SIGN FLIP
%       (macos OPD is opposite prop_add_phase; pymacos opd_sign_flip=true).
%       An external user calls prop_add_phase(bm, OPD_m) directly.
%     - Units: METRES everywhere.  dx_m is the runtime dx_at (SI via CBM),
%       NOT dxElt.  Lengths are chief-ray Euclidean (get_ray_info * CBM).
%     - Center pixel = floor(N/2)+1 (the FFT DC pixel, focus lands there).
%     - Grid orientation: E(row,col), row = +Y (first index), col = +X;
%       a +X PUPIL phase ramp exp(+i 2pi k X/D) sends the FPA peak to
%       col < center (-X side) -- the FT sign this engine uses.  The export
%       RUNS that probe and stores the measured shift in meta.orientation
%       so the consumer can assert the same handedness.
%     - Single wavelength (500 nm); no chromatic leg is touched, so no
%       per-lambda grids are needed (stated in meta).
%
%   out = CTB_PHASE_EXPORT() writes ctb_phase_export_N1024.mat (-v7.3, full)
%   + ctb_phase_export_preview.mat (128-downsampled, committed) + the
%   fingerprint .fp.json.  Name-value:
%     'rx'        full deck (default ctb_s2s_dcr.in).
%     'model_size' grid (1024).
%     'preview_n' preview downsample size (default 128).
%     'outdir'    output dir (default this example dir).
%     'write'     actually write the big .mat (default true; false = dry
%                 run returning the struct for inspection).
%
%   See also: proper_ctb_check, ctb_proper_compare, jac_fingerprint.
    arguments
        opts.rx         (1,:) char   = ''
        opts.model_size (1,1) double = 1024
        opts.preview_n  (1,1) double = 96
        opts.outdir     (1,:) char   = ''
        opts.write      (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_s2s_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    N = opts.model_size;

    % ---- station map (real optics + masks + FPA) for the FULL deck -----
    % iElt, name, kind.  Verified against ctb_s2s_dcr.in element list.
    S = { 1,'OAP1','optic'; 2,'DM1','pupil'; 5,'DM2','pupil'; 8,'OAP2','optic'; ...
          11,'Focus23','focus'; 13,'OAP3','optic'; 16,'Apodizer','pupil'; ...
          19,'OAP4','optic'; 22,'FPM','focus'; 24,'OAP5','optic'; ...
          27,'Lyot','pupil'; 30,'OAP6','optic'; 33,'FieldStop','focus'; ...
          35,'OAP7','optic'; 38,'CheckPoint','pupil'; 41,'OAP8','optic'; ...
          43,'ExitPupil','pupil'; 44,'FPA','focus' };
    nS = size(S,1);

    % ---- legs: propagation table (from/to station, chief length, type) -
    % Each pair of consecutive stations is one leg; the PropType is read
    % from the deck structure between them (NFPlane p2p, quartet through-
    % focus NF1+NF2, or the terminal FarField).  Sphere radii come from the
    % EPreturn zElt of a through-focus quartet / the ExitPupil FF sphere.
    macos.init(N);
    nE = macos.load_rx(opts.rx);
    assert(nE == 44, 'expected the 44-elt full deck; got nElt=%d', nE);
    cbm      = macos.cbm();
    lambda_m = macos.get_src_wvl() * cbm;

    t = macos.trace();  nRay = t.nRays;
    fprintf('[export] full deck nElt=%d  nRay=%d  lambda=%.4e m  N=%d\n', nE, nRay, lambda_m, N);

    % ---- one forward trace, read every station ------------------------
    % EFL_m: focal length of a POWERED optic-kind station (an off-axis
    % parabola), f = |Kr|/2 in SI -- what a PROPER user's prop_lens needs to
    % model that OAP.  Populated ONLY for optic-kind stations with a finite
    % Kr; NaN for pupils/foci and for the ExitPupil (whose FarField-sphere
    % focusing radius R lives in legs.sphere_R_m / spheres, not |Kr|/2).
    stations = struct('name',{},'iElt',{},'kind',{},'E',{},'OPD_m',{}, ...
                      'AMP',{},'dx_m',{},'z_along_chief_m',{},'chief_pos_m',{}, ...
                      'EFL_m',{});
    macos.intensity(S{1,1});                              % first full trace
    prev = []; cum = 0;
    for k = 1:nS
        s = S{k,1};
        cf = mmacos('complex_field', double(s), 0);       % reset_trace=0
        macos.intensity(s, 'reset_trace', false);         % keep trace state
        ri = macos.get_ray_info(nRay);
        p  = ri.pos(:,1) * cbm;                           % chief (ray 1), metres
        if isempty(prev), leg = 0; else, leg = norm(p - prev); cum = cum + leg; end
        prev = p;
        efl = NaN;                                        % powered-OAP focal length
        if strcmp(S{k,3},'optic')
            kr = macos.get_elt_kr(s);
            if abs(kr) < 1e21, efl = abs(kr) * cbm / 2; end   % OAP f = |Kr|/2
        end
        stations(k) = struct('name',S{k,2}, 'iElt',s, 'kind',S{k,3}, ...
            'E', cf, ...
            'OPD_m', -angle(cf) * lambda_m/(2*pi), ...    % macos->PROPER sign flip
            'AMP', abs(cf), ...
            'dx_m', abs(macos.dx_at(s)), ...
            'z_along_chief_m', cum, ...
            'chief_pos_m', p(:).', ...
            'EFL_m', efl);
    end

    % ---- legs table ---------------------------------------------------
    legs = build_legs_(stations, opts.rx, cbm);

    % ---- feeding reference spheres (REPLAY ENABLER for through-focus /  -
    % FarField legs) ----------------------------------------------------
    % A through-focus (NF1 sphere->plane + NF2 plane->sphere) or terminal
    % FarField leg is NOT replayable in PROPER from the OPTIC-plane field:
    % PROPER reproduces our focus ONLY when seeded at the FEEDING REFERENCE
    % SPHERE (the EPreturn element one step before the NF2 focus, or the
    % ExitPupil FF sphere) with THAT sphere's dx and radius R -- then
    % prop_lens(R)+prop_propagate(R) matches macos at intensity peak-norm
    % corr 1.000000 (the ctb_proper_compare arbiter result).  So for every
    % focus/FarField station we export its feeding sphere: field E_sphere,
    % dx_sphere_m, R_m, and the target station it feeds.  (Collimated
    % NFPlane p2p legs need no sphere -- see meta.convention_p2p.)
    sphere_feed = struct( ...
        'Focus23',   10, 'FPM',       21, ...
        'FieldStop', 32, 'FPA',       43);                % feeds -> station
    spheres = build_spheres_(sphere_feed, cbm, lambda_m, nRay);

    % ---- per-optic phase screens (difference-of-consecutive-plane-OPD) -
    % The engine accumulates real-optic OPD geometrically onto the field
    % BETWEEN diffraction legs, so a clean "this optic alone" screen is not
    % directly readable.  We ship the DIFFERENCE construction: the screen
    % attributed to reaching station k is OPD(k) resampled onto a common
    % grid minus OPD(k-1) -- what an external user adds to their wavefront
    % at that plane to reproduce our accumulated phase.  Documented as the
    % diff construction (meta.screen_method).  Screens live on each optic's
    % OWN grid (same as stations(k).OPD_m); the consumer resamples as needed.
    screens = build_screens_(stations);

    % ---- coronagraph masks AS USED by the shipped chain ---------------
    % The four masks the ctb_coro_compare / mask-family drivers apply, built
    % on THE SHIPPED configuration (post centering fix, focus pixel
    % floor(N/2)+1), stored as arrays so the .mat stands alone (no builder
    % dependency downstream).  Each carries its plane's dx_m, its station
    % name, and the provenance (builder + parameters).  The FPM is COMPLEX
    % in general (a phase occulter); the shipped default is the real 2.70
    % lambda/D hard occulter, so we store amplitude + phase split (both real)
    % and note when the phase is trivially zero.  See proper_ctb_run.m.
    masks = build_masks_(stations, N, lambda_m);

    % ---- orientation probe (self-contained, stored in meta) -----------
    orient = orientation_probe_(opts, N, cbm);

    % ---- provenance ---------------------------------------------------
    commit = git_commit_(here);
    meta = struct( ...
        'product','ctb_phase_export', ...
        'format_version', 2, ...                          % v2: + stations.EFL_m + masks block (v1 = stations/legs/spheres/screens)
        'lambda_m', lambda_m, 'N', N, 'center_px', floor(N/2)+1, ...
        'base_unit_to_metre', cbm, ...
        'opd_sign', 'OPD_m = -angle(E)*lambda/(2*pi); macos OPD is OPPOSITE prop_add_phase (pymacos opd_sign_flip=true). Consumer: prop_add_phase(bm, OPD_m).', ...
        'opd_wrapping', 'OPD_m derives from angle(E) in [-pi,pi], so it is WRAPPED (|OPD| <= lambda/2). For the interface check the consumer compares the COMPLEX field E (wrap-immune); to use OPD_m as an additive screen on a smooth pupil, unwrap first (e.g. Goldstein) -- the raw AMP+angle(E) reconstruct E exactly, so E is the primary carrier, OPD_m a convenience view.', ...
        'grid_orientation', 'E(row,col): row=+Y (first index), col=+X. +X pupil phase ramp -> FPA peak on -X side (col<center).', ...
        'orientation', orient, ...
        'units','metres throughout; dx_m = runtime dx_at (SI via CBM), not dxElt; lengths = chief-ray Euclidean (get_ray_info*CBM).', ...
        'wavelength_note','single wavelength (500 nm); no chromatic leg touched, so no per-lambda grids.', ...
        'screen_method','per-optic screen = difference of consecutive-station OPD_m (clean per-optic split not directly readable; diff construction shipped).', ...
        'convention_focus', ['THROUGH-FOCUS + FarField legs: replay in PROPER from the FEEDING ' ...
            'SPHERE (struct `spheres`), NOT the optic-plane field.  Seed prop_begin(N*dx_sphere_m,' ...
            ' lambda, N) with the sphere E, then prop_lens(R_m)+prop_propagate(R_m) -> reproduces ' ...
            'macos at the focus at intensity peak-norm corr 1.000000 (the ctb_proper_compare arbiter).'], ...
        'convention_p2p', ['COLLIMATED NFPlane plane-to-plane legs (pupil->pupil): the RAW COMPLEX ' ...
            'FIELD does NOT match after PROPER prop_propagate.  macos NFPlane reads the field on a ' ...
            'PLANAR reference (local-plane curvature re-zeroed); PROPER prop_propagate accumulates ' ...
            'the full Fresnel quadratic reference-sphere phase.  So over a collimated leg the ' ...
            'INTENSITY agrees (corr ~0.95) but the raw complex fields differ by a large quadratic ' ...
            'reference-phase term (measured DM1->DM2 raw-field corr ~ -0.8).  This is a REFERENCE-' ...
            'PHASE CONVENTION difference, NOT an error (the NF p2p propagator itself is validated to ' ...
            '2.4e-14 macos-vs-macos).  CONSEQUENCE for a consumer: do NOT validate a collimated leg ' ...
            'by raw-complex-field correlation.  Either (RECOMMENDED) consume our exported field E ' ...
            'directly at each plane as the hand-off (the "collapsed" mode -- always valid, no ' ...
            'reference-phase ambiguity), or compare on intensity + reference-sphere-removed phase.'], ...
        'masks_note', ['`masks` carries the four shipped coronagraph masks as stand-alone arrays + ' ...
            'physical (metres) parameters.  Pupil masks (Apodizer/Lyot) transfer directly; the FPM is ' ...
            'a FOCUS-plane occulter whose array is on the macos focal grid for reference -- rebuild it ' ...
            'at the consumer focal dx from radius_m.  stations.EFL_m = |Kr|/2 for powered OAPs (prop_lens).'], ...
        'source_rx', opts.rx, 'upstream_commit', commit, ...
        'nRay', nRay, 'built_by','ctb_phase_export.m');

    out = struct('meta',meta, 'stations',stations, 'legs',legs, ...
                 'spheres',spheres, 'screens',screens, 'masks',masks);

    % ---- write full + preview + fingerprint ---------------------------
    if opts.write
        fullpath = fullfile(opts.outdir, sprintf('ctb_phase_export_N%d.mat', N));
        save(fullpath, '-struct', 'out', '-v7.3');
        fprintf('[export] wrote %s\n', fullpath);

        prev_struct = downsample_(out, opts.preview_n);   %#ok<NASGU>
        prevpath = fullfile(opts.outdir, 'ctb_phase_export_preview.mat');
        save(prevpath, '-struct', 'prev_struct', '-v7');   % small, committed
        fprintf('[export] wrote %s (%d-downsampled preview)\n', prevpath, opts.preview_n);

        write_fingerprint_(out, fullfile(opts.outdir, sprintf('ctb_phase_export_N%d.fp.json', N)), here);
    else
        fprintf('[export] dry run (write=false); struct returned, nothing written.\n');
    end
end

% ======================================================================
function legs = build_legs_(stations, rx, cbm)
%BUILD_LEGS_  One leg per consecutive station pair; classify PropType from
%   the deck structure between the two station iElts, and read sphere radii
%   from the intervening EPreturn/FarField zElt.
    txt = fileread(rx);
    legs = struct('from',{},'to',{},'from_iElt',{},'to_iElt',{}, ...
                  'chief_len_m',{},'prop_type',{},'sphere_R_m',{});
    for k = 2:numel(stations)
        a = stations(k-1); b = stations(k);
        len = norm(b.chief_pos_m - a.chief_pos_m);
        [ptype, R] = classify_leg_(txt, a.iElt, b.iElt, cbm);
        legs(k-1) = struct('from',a.name, 'to',b.name, ...
            'from_iElt',a.iElt, 'to_iElt',b.iElt, ...
            'chief_len_m',len, 'prop_type',ptype, 'sphere_R_m',R);
    end
end

function spheres = build_spheres_(feed, cbm, lambda_m, nRay)
%BUILD_SPHERES_  Feeding reference sphere for each through-focus / FarField
%   leg: read E, dx, R at the EPreturn (or ExitPupil) element that seeds the
%   focus, so a PROPER user replays the leg as prop_lens(R)+prop_propagate(R)
%   from THAT plane (arbiter fidelity).  `feed` maps target-station-name ->
%   feeding-sphere iElt.  Requires a live trace (call after the export trace).
    names = fieldnames(feed);
    spheres = struct('feeds_station',{},'sphere_iElt',{},'E',{},'AMP',{}, ...
                     'OPD_m',{},'dx_sphere_m',{},'R_m',{},'note',{});
    for i = 1:numel(names)
        se = feed.(names{i});
        cf = mmacos('complex_field', double(se), 0);
        macos.intensity(se, 'reset_trace', false);
        spheres(i) = struct('feeds_station',names{i}, 'sphere_iElt',se, ...
            'E', cf, 'AMP', abs(cf), 'OPD_m', -angle(cf)*lambda_m/(2*pi), ...
            'dx_sphere_m', abs(macos.dx_at(se)), ...
            'R_m', abs(macos.get_elt_z(se))*cbm, ...
            'note', sprintf(['seed PROPER at this sphere: prop_begin(N*dx_sphere_m,lambda,N); ' ...
                'inject E; prop_lens(R_m); prop_propagate(R_m) -> %s at corr 1.0'], names{i}));
    end
end

function [ptype, R] = classify_leg_(txt, ia, ib, cbm)
%CLASSIFY_LEG_  Inspect the elements strictly between ia and ib (their
%   PropType lines) to name the leg; pull the EPreturn/FarField sphere R.
    R = NaN;
    seg = elt_block_(txt, ia+1, ib);                     % text of elts (ia,ib]
    hasNF1 = contains(seg,'NF1'); hasNF2 = contains(seg,'NF2');
    hasNFP = contains(seg,'NFPlane'); hasFF = contains(seg,'FarField');
    if hasFF
        ptype = 'FarField (sphere->plane)';
        R = sphere_R_(txt, ib-1, cbm);                   % ExitPupil zElt (=b-1)
    elseif hasNF1 && hasNF2
        ptype = 'through-focus quartet (NF1 EPsphere->plane + NF2 plane->EPsphere)';
        R = sphere_R_near_(seg, cbm);                    % EPreturn zElt in-block
    elseif hasNFP
        ptype = 'NFPlane plane-to-plane';
    else
        ptype = 'geometric jump';
    end
end

function s = elt_block_(txt, i0, i1)
%ELT_BLOCK_  Return the text spanning "iElt=  i0" .. "iElt=  i1" (exclusive
%   of the i1 header line's body is fine -- we only scan PropType/zElt tags).
    lines = regexp(txt, '\r?\n', 'split');
    cur = -1; keep = false; s = '';
    for i = 1:numel(lines)
        m = regexp(lines{i}, '^\s*iElt=\s*(\d+)', 'tokens', 'once');
        if ~isempty(m), cur = str2double(m{1}); keep = (cur>=i0 && cur<=i1); end
        if keep, s = [s, lines{i}, newline]; end %#ok<AGROW>
    end
end

function R = sphere_R_near_(seg, cbm)
%SPHERE_R_NEAR_  First EPreturn-class zElt (a large +R) in a quartet block.
    zs = regexp(seg, 'zElt\s*=\s*([-\d.DEed+]+)', 'tokens');
    R = NaN;
    for i = 1:numel(zs)
        v = str2double(strrep(zs{i}{1},'D','E'));
        if isfinite(v) && abs(v) < 1e21 && abs(v) > 1   % a real sphere radius (not 1e22 plane)
            R = abs(v)*cbm; return;
        end
    end
end

function R = sphere_R_(txt, iElt, cbm)
%SPHERE_R_  zElt of a specific element (the FF ExitPupil sphere).
    blk = elt_block_(txt, iElt, iElt);
    z = regexp(blk, 'zElt\s*=\s*([-\d.DEed+]+)', 'tokens', 'once');
    if isempty(z), R = NaN; return; end
    v = str2double(strrep(z{1},'D','E'));
    R = abs(v)*cbm;
end

% ======================================================================
function screens = build_screens_(stations)
%BUILD_SCREENS_  Per-optic phase screen (OPD it ADDS), diff construction.
%   screen(k) = OPD_m at station k minus OPD_m at station k-1 resampled to
%   station-k's grid.  For k=1 the screen is the OPD at the first plane.
    screens = struct('at_station',{},'iElt',{},'OPD_add_m',{},'dx_m',{},'note',{});
    for k = 1:numel(stations)
        b = stations(k);
        if k == 1
            add = b.OPD_m; note = 'first plane: absolute OPD (no predecessor)';
        else
            a = stations(k-1);
            add = b.OPD_m - resample_opd_(a.OPD_m, a.dx_m, b.dx_m);
            note = sprintf('OPD(%s) - OPD(%s) resampled to this grid', b.name, a.name);
        end
        screens(k) = struct('at_station',b.name, 'iElt',b.iElt, ...
            'OPD_add_m',add, 'dx_m',b.dx_m, 'note',note);
    end
end

% ======================================================================
function masks = build_masks_(stations, N, lambda_m)
%BUILD_MASKS_  The coronagraph masks AS USED by the shipped chain
%   (ctb_coro_compare / the mask-family drivers), captured as stand-alone
%   arrays + physical parameters so proper_ctb_run.m can apply them with NO
%   builder dependency.  Runs its OWN init/trace (safe: called after the
%   export trace + spheres are captured).
%
%   Shipped configuration (ctb_coro_compare defaults):
%     Apodizer  soft circle  r0 = 15 mm, sigma = 2 mm     (amplitude, pupil)
%     FPM       hard occulter 2.70 lambda/D               (amplitude, focus)
%     Lyot      hard disk     0.50 * bare geometric pupil  (amplitude, pupil)
%     FieldStop OPEN in the shipped chain (no mask applied) -- stored as an
%               explicit all-ones entry with a note, so the four coronagraph
%               planes are all represented and a user can drop in their own.
%
%   THE KEY STAND-ALONE FACT: every mask's defining size is stored in METRES
%   (radius_m), which is GRID-INDEPENDENT.  Pupil-plane mask arrays (Apodizer,
%   Lyot) are built on their macos plane dx, which the PROPER cascade
%   reproduces at a pupil (~1.0 sampling ratio), so they transfer directly.
%   The FPM sits at a FOCUS, where the PROPER cascade's focal pitch differs
%   from macos's by ~10x (the interface finding); its array here is on the
%   macos focal grid FOR REFERENCE, and proper_ctb_run REBUILDS the occulter
%   at the consumer's own focal dx from radius_m (a hard disk of a physical
%   radius -- three lines, no builder needed).  All arrays are REAL (the
%   shipped FPM is a real hard occulter); a complex phase FPM would ship an
%   AMP + phase split, both real.
    here = fileparts(mfilename('fullpath'));
    addpath(here);                                        % ctb_mask_disk/softcircle

    % which macos station each mask sits on (this deck's full-model indices)
    e = struct('Apodizer',16, 'FPM',22, 'Lyot',27, 'FieldStop',33);

    % shipped mask parameters (ctb_coro_compare defaults)
    r_apod_m = 15e-3; r_apod_taper_m = 2e-3;             % soft circle
    r_fpm_lamD = 2.70;                                    % occulter, lambda/D
    r_lyot_frac = 0.50;                                   % of bare pupil

    % ---- deterministic mask-sizing geometry (own init; = ctb_coro_compare)
    macos.init(N);
    macos.load_rx(fullfile(here,'ctb_s2s_dcr.in'));
    cbm = macos.cbm();

    % FPM leg: NF1 sphere is FPM-1; R = its zElt; feed pitch = dx_at there.
    macos.intensity(e.FPM);
    Isph   = macos.intensity(e.FPM-1, 'reset_trace', false);
    dx_sph = abs(macos.dx_at(e.FPM-1));
    R_fpm  = abs(macos.get_elt_z(e.FPM-1)) * cbm;
    Dbeam  = 2 * beam_radius_local_(Isph, dx_sph);
    dx_f       = lambda_m * R_fpm / (N * dx_sph);         % Fraunhofer focal pitch
    lamD_fpm_m = lambda_m * R_fpm / Dbeam;                % FPM-local lambda/D (m)
    r_fpm_m    = r_fpm_lamD * lamD_fpm_m;

    % bare geometric pupil radius at the Lyot (no FPM applied on this pre-pass)
    Ily = macos.intensity(e.Lyot, 'reset_trace', false);
    dx_lyot = abs(macos.dx_at(e.Lyot));
    r_lyot_geom_m = beam_radius_local_(Ily, dx_lyot);
    r_lyot_m = r_lyot_frac * r_lyot_geom_m;

    % apodizer pupil dx + field-stop plane dx
    macos.intensity(e.Apodizer);
    dx_apod = abs(macos.dx_at(e.Apodizer));
    macos.intensity(e.FieldStop);
    dx_fstop = abs(macos.dx_at(e.FieldStop));

    % ---- build the arrays (shipped builders, beam-centred) ------------
    M_apod  = ctb_mask_softcircle(N, dx_apod, r_apod_m, r_apod_taper_m, 8);
    M_fpm   = 1 - ctb_mask_disk(N, dx_f, r_fpm_m, 8);     % opaque occulter
    M_lyot  = ctb_mask_disk(N, dx_lyot, r_lyot_m, 8);
    M_fstop = ones(N);                                    % open in the shipped chain

    masks = struct('name',{},'station',{},'plane_kind',{},'M',{}, ...
                   'dx_m',{},'radius_m',{},'active',{},'builder',{}, ...
                   'params',{},'note',{});
    masks(1) = struct('name','Apodizer','station','Apodizer', ...
        'plane_kind','amplitude_pupil', 'M',M_apod, 'dx_m',dx_apod, ...
        'radius_m',r_apod_m, 'active',true, 'builder','ctb_mask_softcircle', ...
        'params',struct('r0_m',r_apod_m,'sigma_m',r_apod_taper_m,'K',8), ...
        'note','soft-circle amplitude apodizer; PUPIL plane -> apply array directly (dx matches PROPER pupil sampling).');
    masks(2) = struct('name','FPM','station','FPM', ...
        'plane_kind','amplitude_focus', 'M',M_fpm, 'dx_m',dx_f, ...
        'radius_m',r_fpm_m, 'active',true, 'builder','1 - ctb_mask_disk', ...
        'params',struct('r_fpm_lamD',r_fpm_lamD,'lamD_fpm_m',lamD_fpm_m, ...
                        'R_fpm_m',R_fpm,'D_beam_m',Dbeam,'K',8), ...
        'note',['opaque hard occulter, 2.70 lambda/D.  FOCUS plane: array is on the macos focal grid ' ...
                'for reference; REBUILD at the consumer focal dx from radius_m (grid-independent metres).']);
    masks(3) = struct('name','Lyot','station','Lyot', ...
        'plane_kind','amplitude_pupil', 'M',M_lyot, 'dx_m',dx_lyot, ...
        'radius_m',r_lyot_m, 'active',true, 'builder','ctb_mask_disk', ...
        'params',struct('r_lyot_frac',r_lyot_frac,'r_lyot_geom_m',r_lyot_geom_m,'K',8), ...
        'note','Lyot stop, 0.50 of the bare geometric pupil; PUPIL plane -> apply array directly.');
    masks(4) = struct('name','FieldStop','station','FieldStop', ...
        'plane_kind','amplitude_focus', 'M',M_fstop, 'dx_m',dx_fstop, ...
        'radius_m',NaN, 'active',false, 'builder','none (open)', ...
        'params',struct(), ...
        'note','OPEN in the shipped chain (no mask applied); all-ones placeholder so the four coronagraph planes are represented.');
end

function rr = beam_radius_local_(I, dx)
    thr = 0.02*max(I(:)); [yy,xx] = find(I>thr);
    if isempty(xx), rr = 0; return; end
    c = floor(size(I,1)/2) + 1; rr = max(hypot(xx-c,yy-c))*dx;
end

% ======================================================================
function J = resample_opd_(W, dx_src, dx_dst)
%RESAMPLE_OPD_  Bilinear resample W (pitch dx_src, array-centre origin) onto
%   the same size with pitch dx_dst (no flux weighting -- OPD is intensive).
    if abs(dx_src/dx_dst - 1) < 1e-12, J = W; return; end
    N=size(W,1); c=(N-1)/2; [xs,ys]=meshgrid((0:N-1)-c,(0:N-1)-c);
    xi = xs*(dx_dst/dx_src)+c; yi = ys*(dx_dst/dx_src)+c;
    J = interp2(0:N-1,(0:N-1).',W,xi,yi,'linear',0);
end

% ======================================================================
function orient = orientation_probe_(opts, N, cbm)
%ORIENTATION_PROBE_  Inject a +X pupil phase ramp at DM1 and record the
%   FPA peak shift, so the consumer can assert the same handedness from the
%   .mat alone.  Runs on a FRESH trace (does not perturb the export trace).
    e_dm1 = 2; e_fpa = 44; k_cycles = 8;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e_dm1);
    I2 = macos.intensity(e_dm1,'reset_trace',false);
    c = floor(N/2)+1; thr = 0.02*max(I2(:)); [yy,xx] = find(I2>thr);
    R = max(hypot(xx-c,yy-c));                            % pupil radius, px
    [X,~] = meshgrid((0:N-1)-floor(N/2),(0:N-1)-floor(N/2));
    ramp = exp(1i*2*pi*k_cycles*(X/(2*R)));              % +X ramp
    macos.apodize_complex(e_dm1, ramp);
    macos.intensity(e_dm1,'reset_trace',false);
    Ifpa = macos.intensity(e_fpa,'reset_trace',false);
    [~,idx] = max(Ifpa(:)); [pr,pc] = ind2sub(size(Ifpa),idx);
    orient = struct('probe','+X pupil phase ramp exp(+i 2pi k X/D) at DM1', ...
        'k_cycles',k_cycles, 'pupil_R_px',R, ...
        'fpa_peak_row',pr, 'fpa_peak_col',pc, 'center_px',c, ...
        'dcol', pc-c, 'drow', pr-c, ...
        'assertion','+X pupil ramp -> FPA peak at col < center (dcol<0), row unchanged');
end

% ======================================================================
function prev = downsample_(out, n)
%DOWNSAMPLE_  Build a small preview: E/OPD/AMP per station block-averaged to
%   n x n; legs + meta carried verbatim; screens downsampled too.
    prev.meta = out.meta;  prev.meta.preview = true;  prev.meta.preview_n = n;
    prev.legs = out.legs;
    prev.stations = out.stations;
    for k = 1:numel(out.stations)
        prev.stations(k).E     = imresize_(out.stations(k).E, n);
        prev.stations(k).OPD_m = imresize_(out.stations(k).OPD_m, n);
        prev.stations(k).AMP   = imresize_(out.stations(k).AMP, n);
    end
    prev.spheres = out.spheres;
    for k = 1:numel(out.spheres)
        prev.spheres(k).E     = imresize_(out.spheres(k).E, n);
        prev.spheres(k).OPD_m = imresize_(out.spheres(k).OPD_m, n);
        prev.spheres(k).AMP   = imresize_(out.spheres(k).AMP, n);
    end
    prev.screens = out.screens;
    for k = 1:numel(out.screens)
        prev.screens(k).OPD_add_m = imresize_(out.screens(k).OPD_add_m, n);
    end
    prev.masks = out.masks;
    for k = 1:numel(out.masks)
        prev.masks(k).M = imresize_(out.masks(k).M, n);   % arrays only; params carried verbatim
    end
end

function B = imresize_(A, n)
%IMRESIZE_  Block-mean downsample A (NxN) to n x n (no Image Toolbox dep).
    N = size(A,1);
    if n >= N, B = A; return; end
    e = floor(N/n)*n;  A = A(1:e,1:e);
    B = squeeze(mean(mean(reshape(A, e/n, n, e/n, n), 1), 3));
end

% ======================================================================
function write_fingerprint_(out, fp_path, here)
%WRITE_FINGERPRINT_  Small committed sidecar (jac_fingerprint pattern).
%   Flattens the per-station AMP/OPD into named matrices so the generic
%   jac_fingerprint('write') captures dims + per-column norms + frobenius.
    S = struct();
    for k = 1:numel(out.stations)
        nm = matlab.lang.makeValidName(out.stations(k).name);
        S.(['AMP_' nm]) = out.stations(k).AMP;
        S.(['OPD_' nm]) = out.stations(k).OPD_m;
    end
    meta = struct('product','ctb_phase_export', ...
        'upstream_commit', out.meta.upstream_commit, ...
        'N', out.meta.N, 'lambda_m', out.meta.lambda_m, ...
        'nStations', numel(out.stations), 'nLegs', numel(out.legs));
    fpfun = fullfile(here, 'jac_fingerprint.m');
    if isfile(fpfun)
        addpath(here);
        jac_fingerprint('write', fp_path, S, meta);
    else
        % self-contained fallback (same JSON shape) if the helper is absent
        fp = local_fp_build_(S, meta);
        fid = fopen(fp_path,'w'); fwrite(fid, jsonencode(fp,'PrettyPrint',true)); fclose(fid);
    end
    fprintf('[export] wrote %s\n', fp_path);
end

function fp = local_fp_build_(S, meta)
    fp.meta = meta; fp.fields = struct([]);
    fn = fieldnames(S);
    for i = 1:numel(fn)
        v = double(S.(fn{i})); if isempty(v), continue; end
        e = struct('name',fn{i},'size',size(v),'fro',norm(v(:)));
        cn = sqrt(sum(reshape(v,size(v,1),[]).^2,1));
        idx = unique(round(linspace(1,numel(cn),min(64,numel(cn)))));
        e.ncol = numel(cn); e.col_idx = idx; e.col_nrm = cn(idx);
        fp.fields = [fp.fields, e]; %#ok<AGROW>
    end
end

function c = git_commit_(here)
    c = 'unknown';
    try
        [st,o] = system(sprintf('git -C "%s" rev-parse --short HEAD', here));
        if st==0, c = strtrim(o); end
    catch
    end
end
