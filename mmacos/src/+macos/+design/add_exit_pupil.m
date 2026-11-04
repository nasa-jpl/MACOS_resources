function info = add_exit_pupil(rx_in, rx_out, opts)
%ADD_EXIT_PUPIL  Give an imported prescription the exit-pupil pair the chain reads.
%
%   info = MACOS.DESIGN.ADD_EXIT_PUPIL(RX_IN, RX_OUT) writes RX_OUT: RX_IN
%   with two Return elements inserted immediately BEFORE its terminal
%   FocalPlane --
%     [nElt]    FP_return   a FLAT Return at the focal-plane vertex, facing
%                           the arriving beam (the image reference);
%     [nElt+1]  ExitPupil   a SPHERICAL Return at the exit pupil: radius =
%                           the chief-ray distance FP -> EP, psi =
%                           -unit(FP -> EP), pointing back at the image,
%                           the sphere's centre of curvature;
%   and the original FocalPlane preserved at nElt+2.  This is the pair
%   `Telescope.add_pupil` emits for a designed telescope, written for a deck
%   that arrived as text: a Zemax conversion (rx_converter/zemax/zmx2macos.py),
%   a CODE V conversion (rx_converter/codev), any hand-written .in.  Zemax and
%   CODE V have no such element, and every wavefront READ in the design
%   layer -- macos.dw_dx and the other dw_d* supervisors, the stage runners
%   run_sensitivities / run_met, PSFs -- takes the OPD at element nElt-1 and
%   REFUSES a deck whose nElt-1 is an optic (`+macos/private/wf_elt_auto.m`;
%   the FP OPD is the path to each ray's landing point and is blind to tilt,
%   Dave 2026-09-08).  So this is the first step from an imported deck into
%   the segmentation -> sensitivities -> MET chain (design/runners/README.md).
%
%   The exit pupil is located by the ENGINE's FEX finder: the deck is loaded,
%   its stop set, the chief ray traced, and FEX crosses it with its
%   field-differential partner.  On the deck's own field this returns the
%   paraxial pupil exactly (measured on the f/6 Cassegrain fixture: vertex
%   and radius to 1e-9 mm of the textbook value, psi on the axis); a tilted
%   probe field ('field_rad', what Telescope.add_pupil uses) moves the sphere
%   onto that field's chief ray instead.  The seed pair is written first so
%   that FEX writes into the pupil slot and never over an optic; the result
%   is reloaded and traced, and the ray count at the focal plane must equal
%   the input's.
%
%   Name-value:
%     'stop_elt'    element to declare as the stop when the deck carries no
%                   ApStop= (a converter deck carries it on its stop
%                   element; a CODE V deck in the header).  Default []:
%                   the deck's own ApStop=, and an error when it has none.
%     'stop_offset' [dx dy] local offset for 'stop_elt' (default [0 0]).
%     'field_rad'   probe field (rad) off the deck's chief direction, as
%                   Telescope.add_pupil probes (1 arcmin = 2.908882e-4).
%                   Default 0: FEX on the deck's own chief ray.  The
%                   source is restored after.  The order matters: the
%                   field is set BEFORE the stop, so the probe's chief
%                   ray is re-aimed through the stop (set after, it keeps
%                   the on-axis aim and the pupil lands off the axis by
%                   the pupil-magnified chief offset).
%     'mode'        FEX axis: 1 chief ray, 0 centroid; [] = the session's
%                   mode, as macos.fex reads it (default []).
%     'model_size'  engine model size (default 128); 'init' (default true)
%                   false reuses the running session at that size.
%     'verbose'     print the result (default true).
%
%   info: .rx_out .n_in .n_out .img_elt .ep_elt .fp_elt .ep_vpt (3x1)
%         .ep_psi (3x1) .ep_radius .nrays_in .nrays_out .rms_ep (the OPD at
%         the pupil, WaveUnits) .stop (how the stop was set) .units.
%
%   Example (the Zemax Cassegrain of rx_converter/zemax/examples):
%       python3 zmx2macos.py cassegrain_example.zmx -o cassegrain.in
%       info = macos.design.add_exit_pupil('cassegrain.in', 'cassegrain_xp.in');
%       out  = macos.dw_dx(macos.Session(128), info.rx_out);   % now reads at the EP
%
%   See also macos.design.Telescope/add_pupil, macos.fex, macos.dw_dx,
%   run_segmentation, run_sensitivities, run_met.

    arguments
        rx_in  (1,:) char
        rx_out (1,:) char = ''
        opts.stop_elt    double {mustBeScalarOrEmpty, mustBeInteger} = []
        opts.stop_offset (1,2) double = [0 0]
        opts.field_rad   (1,1) double {mustBeNonnegative} = 0
        opts.mode        double {mustBeScalarOrEmpty, mustBeMember(opts.mode, [0 1])} = []
        opts.model_size  (1,1) double {mustBeInteger, mustBePositive} = 128
        opts.init        (1,1) logical = true
        opts.verbose     (1,1) logical = true
    end
    assert(isfile(rx_in), 'add_exit_pupil: %s not found', rx_in);
    if isempty(rx_out)
        [d, nm] = fileparts(rx_in);
        rx_out = fullfile(d, [nm '_xp.in']);
    end
    say = @(varargin) fprintf(varargin{:});
    if ~opts.verbose, say = @(varargin) []; end

    % ---- the text -------------------------------------------------------
    R = rx_blocks(rx_in, 'add_exit_pupil');
    n = numel(R.elts);
    if n < 2
        error('macos:design:add_exit_pupil:noOptic', ...
            'add_exit_pupil: %s has %d element(s); need an optic before the focal plane.', rx_in, n);
    end
    fp_blk = R.elts{n};
    if isempty(regexp(fp_blk, '(?m)^\s*Element=\s*FocalPlane\>', 'once'))
        error('macos:design:add_exit_pupil:notFP', ...
            ['add_exit_pupil: the last element of %s is not a FocalPlane (%s).  The pair ' ...
             'goes before a terminal FocalPlane; an afocal deck takes a Reference ' ...
             'terminal instead (Telescope.add_exit_reference).'], rx_in, blk_kind_(fp_blk));
    end
    if ~isempty(regexp(R.elts{n-1}, '(?m)^\s*Element=\s*(Return|Reference)\>', 'once')) ...
            && abs(blk_kr_(R.elts{n-1})) < 1e21
        error('macos:design:add_exit_pupil:hasPupil', ...
            ['add_exit_pupil: element %d of %s is already a curved %s before the focal ' ...
             'plane -- the deck carries an exit pupil.'], n-1, rx_in, blk_kind_(R.elts{n-1}));
    end

    % ---- the engine: load, stop, baseline ----------------------------------
    if opts.init, macos.init(opts.model_size); end
    macos.load_rx(rx_in);
    if ~macos.has_rx() || macos.num_elt() ~= n
        error('macos:design:add_exit_pupil:load', ...
            'add_exit_pupil: the engine loaded %d elements from %s, the text has %d.', ...
            macos.num_elt(), rx_in, n);
    end
    stop_how = set_stop_(R, opts, rx_in);
    t0 = macos.trace(n);
    ray = macos.get_ray_info(t0.nRays);
    FP_Vpt = macos.get_elt_vpt(n);  FP_Vpt = FP_Vpt(:);
    uIn = ray.dir(:,1);  uIn = uIn / norm(uIn);            % the chief ray arriving at the FP
    chief_miss = norm(ray.pos(:,1) - FP_Vpt);

    % ---- pass 1: the seed pair, so FEX writes into the pupil slot ----------
    prev_vpt = macos.get_elt_vpt(n-1);
    rSeed = norm(prev_vpt(:) - FP_Vpt);  if rSeed <= 0, rSeed = 1; end
    ind_line = blk_line_(fp_blk, 'IndRef', '           IndRef=   1.0000000000000000E+00');
    ext_line = blk_line_(fp_blk, 'Extinc', '           Extinc=   0.0000000000000000E+00');
    flat = ret_block_('FP_return', 'Flat',  -1.0e22, -uIn, FP_Vpt,              ind_line, ext_line, rSeed);
    sph  = ret_block_('ExitPupil', 'Conic', -rSeed,   uIn, FP_Vpt - rSeed*uIn,  ind_line, ext_line, rSeed);
    write_(rx_out, R, [R.elts(1:n-1), {flat, sph}, R.elts(n)]);
    macos.load_rx(rx_out);
    nE = n + 2;

    % ---- FEX: on the deck's chief ray, or on a probe field about it --------
    cur = macos.get_src_fov();
    if opts.field_rad > 0
        d0 = cur.src_dir(:) / norm(cur.src_dir);
        px = [1;0;0];
        if abs(dot(px, d0)) > 0.9, px = [0;1;0]; end
        px = px - dot(px, d0)*d0;  px = px / norm(px);
        macos.set_src_fov('src_dir', d0*cos(opts.field_rad) + px*sin(opts.field_rad));
    end
    set_stop_(R, opts, rx_in);                 % AFTER the field: re-aims the chief through the stop
    macos.trace(nE);
    if isempty(opts.mode), f = macos.fex(); else, f = macos.fex(opts.mode); end
    if opts.field_rad > 0, macos.set_src_fov('src_dir', cur.src_dir); end
    EP_Vpt = f.vpt(:);

    % ---- pass 2: the pair per the add_pupil contract -----------------------
    d = EP_Vpt - FP_Vpt;
    radius = norm(d);
    psi = -d / radius;
    flat = ret_block_('FP_return', 'Flat',  -1.0e22, -uIn, FP_Vpt, ind_line, ext_line, radius);
    sph  = ret_block_('ExitPupil', 'Conic', -radius,  psi, EP_Vpt, ind_line, ext_line, radius);
    write_(rx_out, R, [R.elts(1:n-1), {flat, sph}, R.elts(n)]);

    % ---- validate the artifact by reload + trace ---------------------------
    macos.load_rx(rx_out);
    set_stop_(R, opts, rx_in);
    ie = macos.get_elt_info(nE-1);
    if ~any(ie.elt_id == [3, 8])
        error('macos:design:add_exit_pupil:reload', ...
            'add_exit_pupil: element %d of %s reloaded as %s, not a Return.', nE-1, rx_out, ie.type);
    end
    t1 = macos.trace(nE);
    tep = macos.trace(nE-1);
    if t1.nRays ~= t0.nRays
        warning('macos:design:add_exit_pupil:rays', ...
            ['add_exit_pupil: %s traces %d rays to the focal plane, the input %d -- ' ...
             'the pupil pair costs rays (a backward step onto the flat return?).  Inspect.'], ...
            rx_out, t1.nRays, t0.nRays);
    end

    info = struct('rx_out', rx_out, 'n_in', n, 'n_out', nE, ...
                  'img_elt', n, 'ep_elt', nE-1, 'fp_elt', nE, ...
                  'ep_vpt', EP_Vpt, 'ep_psi', psi, 'ep_radius', radius, ...
                  'nrays_in', t0.nRays, 'nrays_out', t1.nRays, 'rms_ep', tep.rmsWFE, ...
                  'chief_miss_fp', chief_miss, 'stop', stop_how, 'units', R.units);
    say('add_exit_pupil: %s -> %s: %d -> %d elements; EP vertex [%.6g %.6g %.6g] %s, radius %.6g %s;\n', ...
        rx_in, rx_out, n, nE, EP_Vpt, R.units, radius, R.units);
    say('  rays at the focal plane %d -> %d; OPD rms at the pupil %.3g (WaveUnits); stop: %s\n', ...
        t0.nRays, t1.nRays, tep.rmsWFE, stop_how);
end

% =========================================================================
function how = set_stop_(R, opts, rx_in)
%SET_STOP_  The deck's own ApStop= (element form, then header form), else 'stop_elt'.
    if ~isempty(opts.stop_elt)
        macos.stop(opts.stop_elt, opts.stop_offset);
        how = sprintf('stop_elt %d', opts.stop_elt);  return
    end
    for k = 1:numel(R.elts)
        t = regexp(R.elts{k}, '(?m)^\s*ApStop=\s*(\S+)\s+(\S+)', 'tokens', 'once');
        if ~isempty(t)
            macos.stop(k, [str2double(t{1}) str2double(t{2})]);
            how = sprintf('ApStop= on element %d', k);  return
        end
    end
    t = regexp(R.header, '(?m)^\s*ApStop=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
    if ~isempty(t)
        macos.stop_obj(str2double(t{1}), str2double(t{2}), str2double(t{3}));
        how = 'header ApStop= position';  return
    end
    error('macos:design:add_exit_pupil:noStop', ...
        ['add_exit_pupil: %s declares no ApStop= and no ''stop_elt'' was given; FEX ' ...
         'needs a stop to aim the chief ray.'], rx_in);
end

function s = ret_block_(name, surf, kr, psi, vpt, ind_line, ext_line, zElt)
    v3 = @(v) sprintf('%24.16E%24.16E%24.16E', v(1), v(2), v(3));   % the converter's column layout
    L = { '             iElt=  0', ...
          ['          EltName=  ' name], ...
           '          Element=  Return', ...
          ['          Surface=  ' surf], ...
          sprintf('            KrElt=%24.16E', kr), ...
           '            KcElt=   0.0000000000000000E+00', ...
          ['           psiElt=' v3(psi)], ...
          ['           VptElt=' v3(vpt)], ...
          ['           RptElt=' v3(vpt)], ...
          ind_line, ext_line, ...
           '             nObs=  0', ...
           '           ApType=  None', ...
           '         PropType=  Geometric', ...
          sprintf('             zElt=%24.16E', zElt), ...
           '          nECoord=  -6' };
    s = strjoin(L, newline);
end

function write_(rx_out, R, elts)
    n = numel(elts);
    for k = 1:n
        elts{k} = regexprep(elts{k}, '(?m)^(\s*iElt=\s*)\S+', sprintf('$1 %d', k), 'once');
    end
    hdr = regexprep(R.header, '(?m)^(\s*nElt=\s*)\S+', sprintf('$1 %d', n), 'once');
    txt = [hdr newline newline strjoin(elts, [newline newline]) newline R.tail];
    fid = fopen(rx_out, 'w');
    assert(fid > 0, 'add_exit_pupil: cannot write %s', rx_out);
    fprintf(fid, '%s', txt);  fclose(fid);
end

function k = blk_kind_(blk)
    t = regexp(blk, '(?m)^\s*Element=\s*(\S+)', 'tokens', 'once');
    if isempty(t), k = '?'; else, k = t{1}; end
end

function kr = blk_kr_(blk)
    t = regexp(blk, '(?m)^\s*KrElt=\s*(\S+)', 'tokens', 'once');
    if isempty(t), kr = -1e22; else, kr = str2double(t{1}); end
end

function s = blk_line_(blk, key, default)
    t = regexp(blk, ['(?m)^\s*' key '=[^\n]*'], 'match', 'once');   % MATLAB's . matches newline
    if isempty(t), s = default; else, s = t; end
end
