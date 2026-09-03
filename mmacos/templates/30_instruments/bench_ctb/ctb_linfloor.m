function la = ctb_linfloor(J, stroke_bound_nm)
%CTB_LINFLOOR  Linear-achievable dark-zone floor of a measured Jacobian.
%   la = CTB_LINFLOOR(J, STROKE_BOUND_NM) returns the linear-achievable
%   (la) diagnostic of an engine-measured Jacobian J (ctb_dm_jacobian
%   output or its loaded .mat): the top-rank real-stacked least-squares
%   residual of the STORED static field J.E0_dz as a function of rank,
%   with a stroke bound applied.  Closed form per rank from one SVD:
%     ||res_r||^2 = ||e0||^2 - sum_{i<=r} (u_i' e0)^2      (residual)
%     ||a_r||^2   = sum_{i<=r} ((u_i' e0)/s_i)^2           (command)
%   The floor is the contrast at the largest rank whose command stroke
%   rms is within the bound -- the bench_ctb "linear-achievable floor
%   4.5e-9 at 11 nm rms" diagnostic (Session 10), and the per-round la
%   the DST-defects lanes record.  This is a PROPERTY OF G at its
%   linearization point (fixed per Jacobian); relinearization re-measures
%   G at the dug state, so call this on each Jacobian in a relin ladder.
%
%   DM commands are in mm (ctb_dm convention); stroke reported in nm.
%   stroke_bound_nm default 50 (the ctb_efc stroke_warn class).
%
%   See also: ctb_dm_jacobian, ctb_efc, cf_efc_lib (the e2e6m twin).
    if nargin < 2 || isempty(stroke_bound_nm), stroke_bound_nm = 50; end
    G   = double(J.G);
    Gr  = [real(G); imag(G)];
    e0  = double(J.E0_dz(:));
    e0r = [real(e0); imag(e0)];
    [U, S, ~] = svd(Gr, 'econ');
    s   = diag(S);
    Ue  = U' * e0r;
    npix = numel(e0);
    pk   = J.peak_bare;
    res2 = max(sum(e0r.^2) - cumsum(Ue.^2), 0);
    con  = res2 / npix / pk;
    a2   = cumsum((Ue ./ s).^2);           % mm^2 (command magnitude^2)
    stroke_nm = 1e6 * sqrt(a2 / size(Gr, 2));
    ok = stroke_nm <= stroke_bound_nm;
    if ~any(ok), rbest = 1; else, rbest = find(ok, 1, 'last'); end
    la = struct('c_static', sum(e0r.^2)/npix/pk, ...
                'floor', con(rbest), 'rank', rbest, ...
                'stroke_nm', stroke_nm(rbest), 'bound_nm', stroke_bound_nm, ...
                'curve_con', con(:).', 'curve_stroke_nm', stroke_nm(:).');
end
