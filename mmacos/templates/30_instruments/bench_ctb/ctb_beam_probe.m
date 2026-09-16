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
n = macos.n_elt();
nm = cell(1, n);
for i = 1:n, nm{i} = strtrim(macos.get_elt_name(i)); end

fprintf('\n=== CTB measured beam footprint (%s, model %d) ===\n', opts.rx, opts.model);
fprintf('%-12s %6s %12s %12s %12s %10s\n', ...
        'element', 'elt', 'fp radius', 'fp diameter', 'clear r', 'rays');
out = struct('name',{{}}, 'ielt',[], 'r_mm',[], 'd_mm',[], 'apr_mm',[], 'nray',[]);
for k = 1:numel(opts.elts)
    ie = find(strcmp(nm, opts.elts{k}), 1);
    if isempty(ie)
        fprintf('%-12s   NOT IN THIS DECK\n', opts.elts{k});  continue
    end
    st = macos.trace(ie);
    ri = macos.get_ray_info(st.nRays);
    ok = ri.ok_trace ~= 0 & ri.ok_pass ~= 0;
    % footprint in the element's own plane: distance from the chief's hit
    P = [ri.x(ok), ri.y(ok), ri.z(ok)];
    c = P(1,:);                       % the chief is ray 1
    r = max(sqrt(sum((P - c).^2, 2)));
    apr = NaN;
    try, a = macos.get_elt_ap_vec(ie);  apr = a(1);  catch, end
    fprintf('%-12s %6d %12.4f %12.4f %12.4f %10d\n', ...
            opts.elts{k}, ie, r, 2*r, apr, nnz(ok));
    out.name{end+1} = opts.elts{k};  out.ielt(end+1) = ie;
    out.r_mm(end+1) = r;  out.d_mm(end+1) = 2*r;
    out.apr_mm(end+1) = apr;  out.nray(end+1) = nnz(ok);
end

% What it decides, spelled out so the answer cannot be mis-read again.
if ~isempty(out.r_mm)
    D = out.d_mm(1);  lam = 5.5e-4;  z = 500;          % DM1 -> DM2, mm
    % Fresnel amplitude conversion sin(pi*lam*z/L^2); 50% at L^2 = 6*lam*z
    L50 = sqrt(6*lam*z);  L100 = sqrt(2*lam*z);
    fprintf(['\nDM1 beam: radius %.3f mm, diameter %.3f mm.  With lambda %.1f nm and\n' ...
             'DM1->DM2 %g mm, the Fresnel amplitude conversion sin(pi*lam*z/L^2) reaches\n' ...
             '50%%%% at %.3f mm of period = %.1f cycles across the beam, and 100%%%% at\n' ...
             '%.3f mm = %.1f cycles.  (NOTES_gauge_in_coronagraph section C+ says 33.)\n'], ...
            out.r_mm(1), D, lam*1e6, z, L50, D/L50, L100, D/L100);
    fprintf(['ctb_dm.m default beam_d_mm = 21.3 is %s this measurement.\n'], ...
            iff_(abs(D-21.3) < abs(out.r_mm(1)-21.3), 'CONSISTENT with (a diameter)', ...
                 'INCONSISTENT with: 21.3 matches the RADIUS, so the lattice spans half the pupil'));
end
end

function o = iff_(c, a, b), if c, o = a; else, o = b; end, end
