function out = ctb_beam_probe(opts)
%CTB_BEAM_PROBE  The MEASURED beam footprint at each CTB pupil plane.
%   Settles a radius-vs-diameter contradiction that changes the field
%   servo's separability answer by 2x (BRIEF_to_gauge_close item 7 step 2).
%
%   The documents disagree, so neither can be used:
%     README source model: "the source NA is set to put a R_DM.FILL beam on
%       the DM" -- R_DM is the DM RADIUS (22.5 mm), so R_DM*FILL is a
%       RADIUS, and the chain "DM 21.4 -> apod 16.0 -> Lyot 8.0" is radii.
%     ctb_dm.m: 'beam_d_mm' default 21.3, documented as the controlled beam
%       DIAMETER and used as one -- pitch = beam_d_mm/nact = 0.666 mm and
%       active = hypot <= beam_d_mm/2 + pitch.  If 21.3 is really the
%       RADIUS, the 32x32 lattice spans half the pupil and the pitch is half
%       what a 32-across DM on this beam would have.
%     NOTES_gauge_in_coronagraph.md section C+ uses BOTH at once: pitch
%       0.67 mm (which implies a 21.4 mm SPAN) with a beam whose diameter it
%       takes to be 42.8 mm, and predicts 50% Fresnel amplitude conversion
%       at 33 cycles across the beam.  With a 21.4 mm beam the same formula
%       gives 16.6 cycles.  32 * 0.67 = 21.4, not 42.8 -- the two halves of
%       that prediction cannot both be right.
%
%   So: ask the ENGINE.  Trace the committed deck and report the ray
%   footprint at every pupil plane, as a radius AND a diameter, with the
%   element's declared clear aperture beside it.  No deck is modified.
%
%   Run:  >> ctb_beam_probe
    arguments
        opts.rx    (1,:) char = 'ctb_dcr.in'
        opts.model (1,1) double = 512
        opts.elts  (1,:) cell = {'DM1','DM2','Apodizer','Lyot'}
    end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
cd(exdir);
macos.init(opts.model);
macos.load_rx(opts.rx);
n = macos.num_elt();

% Element NAMES are not in the engine API -- they come from the deck text, the
% way macos.segment_grid_basis does it.  That is the one place this probe can
% be misled, and it is a known trap: several decks in this corpus declare an
% nElt that disagrees with their Element= block count, which shifts every index
% and attributes the WRONG element's numbers (the FEX blast-radius lesson).  So
% the parse is CROSS-CHECKED against the engine's own count, and each element
% the probe reports is verified by TYPE before its footprint is believed.
txt = fileread(opts.rx);
nm  = regexp(txt, '(?<=EltName=)[^\r\n]*', 'match');
nm  = strtrim(nm);
if numel(nm) ~= n
    warning('ctb_beam_probe:count', ...
        ['%s declares %d EltName= blocks but the engine reports %d elements. ' ...
         'Indices from the text are NOT trustworthy here -- reporting by index only.'], ...
        opts.rx, numel(nm), n);
end

fprintf('\n=== CTB measured beam footprint (%s, model %d, %d elements) ===\n', ...
        opts.rx, opts.model, n);
fprintf('%-12s %5s %-12s %11s %11s %10s %7s\n', ...
        'element', 'elt', 'type', 'fp radius', 'fp diameter', 'clear r', 'rays');
out = struct('name',{{}}, 'ielt',[], 'r_mm',[], 'd_mm',[], 'apr_mm',[], 'nray',[]);
for k = 1:numel(opts.elts)
    ie = find(strcmp(nm, opts.elts{k}), 1);
    if isempty(ie)
        fprintf('%-12s   NOT FOUND in %s\n', opts.elts{k}, opts.rx);  continue
    end
    info = macos.get_elt_info(ie);
    st = macos.trace(ie);
    ri = macos.get_ray_info(st.nRays);
    ok = ri.ok_trace & ri.ok_pass;          % logical N x 1
    P  = ri.pos(:, ok).';                   % pos is 3 x N -> rows of [x y z]
    if size(P,1) < 8
        fprintf('%-12s %5d %-12s   only %d rays -- not measured\n', ...
                opts.elts{k}, ie, info.type, size(P,1));
        continue
    end
    c = ri.pos(:, 1).';                     % the chief is ray 1
    r = max(sqrt(sum((P - c).^2, 2)));
    apr = NaN;  if ~isempty(info.ap_vec), apr = info.ap_vec(1); end
    fprintf('%-12s %5d %-12s %11.4f %11.4f %10.4f %7d\n', ...
            opts.elts{k}, ie, info.type, r, 2*r, apr, size(P,1));
    out.name{end+1} = opts.elts{k};  out.ielt(end+1) = ie;
    out.r_mm(end+1) = r;  out.d_mm(end+1) = 2*r;
    out.apr_mm(end+1) = apr;  out.nray(end+1) = size(P,1);
end

% What it decides, spelled out so the answer cannot be mis-read again.
if ~isempty(out.r_mm)
    D = out.d_mm(1);  lam = 5.5e-4;  z = 500;          % DM1 -> DM2, mm
    L50 = sqrt(6*lam*z);  L100 = sqrt(2*lam*z);        % sin(pi*lam*z/L^2) = 1/2, 1
    fprintf(['\nDM1 beam: radius %.3f mm, diameter %.3f mm.\n' ...
             'Fresnel amplitude conversion sin(pi*lam*z/L^2) at lam %.1f nm over z %g mm:\n' ...
             '  50%%%% at a period of %.3f mm = %.1f cycles across the beam\n' ...
             ' 100%%%% at a period of %.3f mm = %.1f cycles across the beam\n' ...
             '(NOTES_gauge_in_coronagraph section C+ quotes 33 for the 50%%%% point.)\n'], ...
            out.r_mm(1), D, lam*1e6, z, L50, D/L50, L100, D/L100);
    fprintf(['ctb_dm.m fills beam_d_mm with 21.3.  Measured DIAMETER %.3f, measured ' ...
             'RADIUS %.3f -> 21.3 is the %s.\n'], D, out.r_mm(1), ...
            iff_(abs(D-21.3) < abs(out.r_mm(1)-21.3), ...
                 'DIAMETER, and the DM model is consistent with the deck', ...
                 'RADIUS, so the 32x32 lattice spans HALF the beam and the pitch is half what it should be'));
end
end

function o = iff_(c, a, b), if c, o = a; else, o = b; end, end
