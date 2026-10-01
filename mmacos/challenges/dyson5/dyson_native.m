function N = dyson_native(P, tag, Pr, opts)
%DYSON_NATIVE  The native optimize: CALIB (the engine's multi-field, multi-
%   wavelength least squares) on a ladder rung's deck, with the smile and
%   keystone operands as WALLS held on the exact chain between chunks of
%   iterations.
%
%   N = dyson_native(P, tag, Pr) takes the rung's parameter set Pr (the R4
%   of record), freezes the chain's solved quantities (groove period, FPA
%   plane) so the chain can carry an engine-moved design, emits the deck
%   with the CALIB block (SPOT target = the max ray distance to the chief
%   per (slit position, wavelength), weighted equally) and the double-pass
%   links (the return-pass copy of a surface follows its first pass), and
%   runs CALIB in chunks of opts.chunk iterations.  After every chunk the
%   engine's element state is READ BACK and mapped into the chain's
%   parameters (an identity check proves the mapping: every vertex, normal,
%   radius and conic agrees with the engine to 1e-9 m); the chunk is
%   accepted only if, scored in the ENGINE over P.score_nx x P.score_nlam,
%   the smile and keystone stay inside opts.wall_px and the clearance gate
%   passes.  A rejected chunk restores the last accepted state and ends the
%   solve; the solve also ends when the CRF stops improving by opts.tol_px.
%
%   Why walls on the chain and not operands in CALIB: CALIB's SPOT target
%   carries no centroid-position operand (smacos_compute.inc computes one
%   radius per field; design_optim.F sizes the objective 1 per (FOV,
%   lambda)), so distortion cannot enter its merit without an engine change
%   (reported to CC in BRIEF_dyson5_beat4c.md).  The walls are Dave's rule
%   in its coarse form: on ITERATES (every chunk), never on reports.
%
%   The CALIB variables (opts.var; the default set):
%     Grating         DY, PIST   -- the grating's position vs the block+slit+FPA
%                                   assembly (a real alignment knob; in the
%                                   chain's grating-centred frame it is the
%                                   slit plane off the grating's centre)
%     BlockSphereOut  ROC, CONIC -- the block's convex face (its return-pass
%                                   copy linked); the h^4/h^6 asphere is HELD
%                                   unless opts.asph (see the CC note: the
%                                   OptAsph slice in smacos_compute.inc)
%     MenA_out, MenB_out  PIST, ROC -- the meniscus faces (copies linked)
%     FPA             PIST       -- focus (the pre-FPA Reference linked)
%   The groove period is not a CALIB variable (RuleWidth has no DOF); the
%   band's span on the FPA is reported after the solve as a check.
%
%   Returns N.rung (the ladder's rung struct: .name .vars .x .P .chain
%   .engine .file .merit -- the trade table reads it), N.history (one row
%   per chunk), N.identity_max_m, N.span_px, N.calib (what was varied).
    arguments
        P struct
        tag (1,:) char
        Pr struct
        opts.nx (1,1) double = 5
        opts.nlam (1,1) double = 6
        opts.chunk (1,1) double = 5
        opts.max_chunks (1,1) double = 8
        opts.wall_px (1,1) double = 0.05
        opts.tol_px (1,1) double = 0.002
        opts.asph (1,1) logical = false
        opts.var = []
        opts.varset (1,:) char {mustBeMember(opts.varset, {'all', 'blur'})} = 'blur'
        opts.quiet (1,1) logical = false
    end
    pr = @(varargin) print_(opts.quiet, varargin{:});
    % --- the rung's chain, with its solved quantities frozen into parameters
    G0 = spectrometer_geom('dyson', Pr);
    Pn0 = Pr;  Pn0.grating_d = G0.grating.d;  Pn0.grating_m = G0.grating.m;  Pn0.fpa_z = G0.fpa.z;  Pn0.slit_dz = 0;
    G1 = spectrometer_geom('dyson', Pn0);
    dd = surf_diff_(G0, G1);
    assert(dd < 1e-12, 'dyson_native: freezing the solved quantities changed the chain (%.3g m)', dd);
    % --- fields and wavelengths for CALIB (<= 12 FOV x 6 lambda)
    W = P.npix(1)*P.pixel_m;
    xs = linspace(-W/2, W/2, opts.nx);  lams = linspace(P.band_m(1), P.band_m(2), opts.nlam);
    assert(opts.nx <= 12 && opts.nlam <= 6, 'dyson_native: CALIB caps at 12 FOV x 6 wavelengths');
    fovs = struct('slit', {}, 'dir', {});
    for i = 1:opts.nx
        sl = G1.slit + [xs(i); 0; 0];
        fovs(end+1) = struct('slit', sl, 'dir', G1.aim(sl, G1.src.lambda_c));   %#ok<AGROW>
    end
    V = opts.var;
    if isempty(V)
        % 'all' includes the grating's position (DY, PIST).  MEASURED
        % (2026-10-01, first run after macos 0d257ff): five CALIB iterations with
        % it took the keystone from 0.003 to 15 px -- a grating moved along the
        % dispersion direction changes the dispersion geometry, which the
        % blur-only SPOT merit cannot see -- and the wall rejected the chunk.
        % 'blur' (the default) leaves the dispersion geometry alone: block
        % face ROC + CONIC, meniscus faces PIST + ROC, FPA PIST.
        if strcmp(opts.varset, 'all')
            V = struct('name', {'Grating', 'BlockSphereOut', 'MenA_out', 'MenB_out', 'FPA'}, ...
                       'mask', {[0 0 0 0 1 1 0 0], [0 0 0 0 0 0 1 1], [0 0 0 0 0 1 1 0], [0 0 0 0 0 1 1 0], [0 0 0 0 0 1 0 0]}, ...
                       'asph', {[], [], [], [], []});
            ia = 2;
        else
            V = struct('name', {'BlockSphereOut', 'MenA_out', 'MenB_out', 'FPA'}, ...
                       'mask', {[0 0 0 0 0 0 1 1], [0 0 0 0 0 1 1 0], [0 0 0 0 0 1 1 0], [0 0 0 0 0 1 0 0]}, ...
                       'asph', {[], [], [], []});
            ia = 1;
        end
        if opts.asph, V(ia).asph = [1 2]; end
    end
    O = struct('fovs', fovs, 'wavelens', lams, 'weights', ones(1, opts.nx), 'target', 'SPOT', ...
               'wf_elt', [], 'max_iters', opts.chunk, 'var', V);
    file_seed = sprintf('%s_s4_r4n_seed.in', tag);
    M = spectrometer_rx(G1, file_seed, 'ngridpts', P.ngridpts, 'name', [P.tag '_r4n_seed'], ...
                        'apertures', true, 'margin', ap_margin_(P), 'links', true, 'opt', O);
    macos.load_rx(file_seed);
    assert(macos.num_elt() == M.nElt, 'dyson_native: the seed deck did not load with every element');
    ndof = sum(cellfun(@(m) nnz(m), {V.mask})) + sum(cellfun(@numel, {V.asph}));
    pr('dyson_native: %d fields x %d wavelengths, %d CALIB variables (%s), chunks of %d iterations, walls %.3f px\n', ...
       opts.nx, opts.nlam, ndof, strjoin({V.name}, ', '), opts.chunk, opts.wall_px);
    pr('  links: %s\n', strjoin(arrayfun(@(k) sprintf('%s->%s', M.names{k}, M.names{M.link(k)}), find(M.link), 'uni', 0), ', '));
    % --- the seed scored in the engine (the record's R4 row, re-measured on this deck)
    Re0 = spectrometer_score(G1, M, Pn0, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
    hist = struct('chunk', {}, 'iters', {}, 'converged', {}, 'smile', {}, 'keystone', {}, 'crf', {}, 'srf', {}, 'ee', {}, ...
                  'clear_mm', {}, 'identity_m', {}, 'accepted', {}, 'note', {});
    hist(1) = struct('chunk', 0, 'iters', 0, 'converged', true, 'smile', Re0.smile_max, 'keystone', Re0.keystone_max, ...
                     'crf', Re0.crf_max, 'srf', Re0.srf_max, 'ee', Re0.ee_min, 'clear_mm', G1.fpa.clear_to_slit*1e3, ...
                     'identity_m', 0, 'accepted', true, 'note', 'seed (R4 of record)');
    pr('  %-5s %5s %8s %8s %7s %7s %6s %8s %9s  %s\n', 'chunk', 'iters', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'clear', 'identity', '');
    pr('  %-5d %5d %8.4f %8.4f %7.3f %7.3f %6.3f %8.2f %9.1e  %s\n', 0, 0, Re0.smile_max, Re0.keystone_max, Re0.crf_max, Re0.srf_max, Re0.ee_min, ...
       G1.fpa.clear_to_slit*1e3, 0, 'seed');
    tmpd = tempname;  mkdir(tmpd);
    last_good = fullfile(tmpd, 'accepted_0');  macos.save_rx(last_good);  last_good = [last_good '.in'];
    Pacc = Pn0;  Gacc = G1;  Racc = Re0;  it_total = 0;
    for c = 1:opts.max_chunks
        macos.calib_set_iter(opts.chunk);
        r = macos.calib();
        it_total = it_total + opts.chunk;
        % read the engine's elements back and map them into the chain
        E = engine_state_(M.nElt);
        [Pn, s] = map_(Pn0, E, M, G0);
        note = '';  ok = true;
        try
            Gk = spectrometer_geom('dyson', Pn);
            idm = identity_(Gk, E, M, s);
        catch err
            Gk = [];  idm = Inf;  ok = false;  note = ['chain: ' err.message];
        end
        if ok && idm > 1e-9, ok = false;  note = sprintf('identity %.2e m', idm); end
        if ok && (Pn.men_t < 1e-3 || norm(s) > 20e-3 || abs(Pn.block_r/Pr.block_r - 1) > 0.2)
            ok = false;  note = 'sanity: meniscus thickness / grating shift / block radius';
        end
        if ok
            Re = spectrometer_score(Gk, M, Pn, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
            Cl = spectrometer_clearance(Gk, P, 'quiet', true);
            if Re.smile_max > opts.wall_px || Re.keystone_max > opts.wall_px
                ok = false;  note = sprintf('WALL: smile %.4f keystone %.4f > %.3f px', Re.smile_max, Re.keystone_max, opts.wall_px);
            elseif ~Cl.pass
                ok = false;  note = sprintf('CLEARANCE %+.2f mm (%s vs %s)', Cl.min_mm, Cl.table.leg{1}, Cl.table.body{1});
            end
        else
            Re = Re0;  Cl = struct('min_mm', NaN);
        end
        hist(end+1) = struct('chunk', c, 'iters', it_total, 'converged', r.converged, 'smile', Re.smile_max, 'keystone', Re.keystone_max, ...
                             'crf', Re.crf_max, 'srf', Re.srf_max, 'ee', Re.ee_min, 'clear_mm', Cl.min_mm, 'identity_m', idm, ...
                             'accepted', ok, 'note', note);   %#ok<AGROW>
        pr('  %-5d %5d %8.4f %8.4f %7.3f %7.3f %6.3f %8.2f %9.1e  %s\n', c, it_total, Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, ...
           Re.ee_min, Cl.min_mm, idm, tern_(ok, 'accepted', ['REJECTED -- ' note]));
        if ~ok
            macos.load_rx(last_good);  break
        end
        gain = Racc.crf_max - Re.crf_max;
        Pacc = Pn;  Gacc = Gk;  Racc = Re;
        last_good = fullfile(tmpd, sprintf('accepted_%d', c));  macos.save_rx(last_good);  last_good = [last_good '.in'];
        if gain < opts.tol_px && c > 1
            pr('  CRF gain %.4f px < %.3f px: converged\n', gain, opts.tol_px);  break
        end
    end
    rmdir(tmpd, 's');
    % --- the design of record for this rung: a CLEAN emitter deck from the
    % mapped chain (the identity check makes it the engine's design), loaded
    % and scored in the engine
    file = sprintf('%s_s4_r4n.in', tag);
    Mn = spectrometer_rx(Gacc, file, 'ngridpts', P.ngridpts, 'name', [P.tag '_r4n'], 'apertures', true, 'margin', ap_margin_(P));
    macos.load_rx(file);
    Re = spectrometer_score(Gacc, Mn, Pacc, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
    Rc = spectrometer_score_chain(Gacc, Pacc, 'nx', P.score_nx, 'nlam', P.score_nlam, 'nring', 6);
    span = abs(Re.V(ceil(end/2), end) - Re.V(ceil(end/2), 1));     % band span on the FPA, px, slit centre
    N.rung = struct('name', 'R4n native CALIB blur on R4, walls smile/keystone <= 0.05 px (chain)', 'vars', {{V.name}}, 'x', [], ...
                    'P', Pacc, 'chain', Rc, 'engine', Re, 'file', file, 'merit', NaN);
    N.seed = struct('file', file_seed, 'engine', Re0, 'P', Pn0);
    N.history = struct2table(hist);
    N.identity_max_m = max([hist.identity_m]);
    N.span_px = span;  N.span_spec_px = P.npix(2);
    N.calib = struct('fields_x_m', xs, 'wavelens_m', lams, 'var', V, 'ndof', ndof, 'chunk', opts.chunk, 'iters', it_total, 'wall_px', opts.wall_px);
    N.shift_m = (Pacc.slit_dz ~= 0)*[0; Pr.y_slit - Pacc.y_slit; -Pacc.slit_dz];
    pr('dyson_native: R4n engine smile %.4f keystone %.4f CRF %.3f SRF %.3f EE %.3f (seed CRF %.3f EE %.3f); band span %.1f px of %d; deck %s\n', ...
       Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, Re0.crf_max, Re0.ee_min, span, P.npix(2), file);
end

function E = engine_state_(n)
    E.vpt = zeros(3, n);  E.psi = zeros(3, n);  E.kr = zeros(1, n);  E.kc = zeros(1, n);
    for k = 1:n
        E.vpt(:,k) = reshape(macos.get_elt_vpt(k), 3, 1);  E.psi(:,k) = reshape(macos.get_elt_psi(k), 3, 1);
        E.kr(k) = macos.get_elt_kr(k);  E.kc(k) = macos.get_elt_kc(k);
    end
end

function [Pn, s] = map_(Pn0, E, M, G0)
%MAP_  Engine element state -> chain parameters, in the chain's grating-
%   centred frame.  Conventions: engine KrElt = -R with psi toward the
%   centre of curvature (the emitter's rule), so centre = Vpt + |Kr| psi; the
%   chain's meniscus curvature c has its centre at the vertex + 1/c along
%   +z, so c = psi_z/|Kr|.  Everything is translated by -s, the grating's
%   centre, so the grating stays at the origin and the slit plane carries
%   the shift (P.slit_dz, P.y_slit).
    ix = @(nm) find(strcmp(M.names, nm), 1);
    iG = ix('Grating');  Rg = abs(E.kr(iG));  s = E.vpt(:,iG) + Rg*E.psi(:,iG);
    Pn = Pn0;
    iB = ix('BlockSphereOut');  r = abs(E.kr(iB));  Cb = E.vpt(:,iB) + r*E.psi(:,iB) - s;
    Pn.block_r = r;  Pn.block_dz = Cb(3);  Pn.block_dy = Cb(2);  Pn.block_Kc = E.kc(iB);
    n0 = G0.n(Pn0.lambda_ref_m);  Pn.Rg_factor = Rg/(n0*r/(n0 - 1));
    Pn.y_slit = Pn0.y_slit - s(2);  Pn.slit_dz = Pn0.slit_dz - s(3);  Pn.face_offset = Pn0.face_offset - s(3);
    iA = ix('MenA_out');  iBm = ix('MenB_out');
    if ~isempty(iA)
        za = E.vpt(3,iA) - s(3);  zb = E.vpt(3,iBm) - s(3);
        Pn.men_z = za;  Pn.men_t = zb - za;
        Pn.men_ca = E.psi(3,iA)/abs(E.kr(iA));  Pn.men_cb = E.psi(3,iBm)/abs(E.kr(iBm));
    end
    iF = ix('FPA');  Pn.fpa_z = E.vpt(3,iF) - s(3);
end

function d = identity_(Gk, E, M, s)
%IDENTITY_  Max disagreement (m, with radii in m and conics as numbers)
%   between the mapped chain's surfaces (+s) and the engine's elements.
    d = 0;
    for k = 1:numel(Gk.surf)
        nm = Gk.surf(k).name;  j = find(strcmp(M.names, nm), 1);
        if isempty(j), continue; end
        if strcmp(Gk.surf(k).kind, 'plane'), v = Gk.surf(k).C(:); else, v = Gk.surf(k).vpt(:); end   % the emitter writes a plane's point
        d = max(d, norm(v + s - E.vpt(:,j)));
        d = max(d, abs(1 - Gk.surf(k).psi(:)'*E.psi(:,j)));
        if ~strcmp(Gk.surf(k).kind, 'plane')
            d = max(d, abs(Gk.surf(k).R - abs(E.kr(j))));
            Kc = 0;  if isfield(Gk.surf(k), 'Kc') && ~isempty(Gk.surf(k).Kc), Kc = Gk.surf(k).Kc; end
            d = max(d, abs(Kc - E.kc(j)));
        end
    end
    % the FPA plane itself (the chain's last surface is the detector)
    j = find(strcmp(M.names, 'FPA'), 1);
    d = max(d, abs(Gk.fpa.z + s(3) - E.vpt(3,j)));
end

function d = surf_diff_(Ga, Gb)
    d = 0;
    for k = 1:numel(Ga.surf)
        d = max(d, norm(Ga.surf(k).vpt(:) - Gb.surf(k).vpt(:)));
        d = max(d, norm(Ga.surf(k).C(:) - Gb.surf(k).C(:)));
    end
    d = max(d, abs(Ga.fpa.z - Gb.fpa.z));  d = max(d, abs(Ga.grating.d - Gb.grating.d));
end

function m = ap_margin_(P)
    if isfield(P, 'ap_margin_m'), m = P.ap_margin_m; else, m = 5e-3; end
end

function s = tern_(c, a, b)
    if c, s = a; else, s = b; end
end

function print_(quiet, varargin)
    if ~quiet, fprintf(varargin{:}); end
end
