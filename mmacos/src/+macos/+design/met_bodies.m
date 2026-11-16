function bodies = met_bodies(elts)
%MET_BODIES  Rigid-body frames for dldx_analytic, from the ENGINE state.
%   bodies = macos.design.met_bodies(ELTS) queries the loaded Rx for
%   each element's perturbation frame -- RptElt (the rotation pivot
%   CPERTURB_PROG rotates points about) and TElt (the 6x6 local->global
%   perturbation map macos.perturb applies in 'local' frame) -- and
%   returns the 1xN bodies struct macos.design.dldx_analytic takes
%   (.rpt 3x1 BaseUnits, .T 3x3 triad columns).
%
%   This is the engine-truth replacement for hand-built triads: for
%   segments from segment_rx it reproduces the face triads
%   (xMon/yMon/zMon), and it is the ONLY correct source for non-segment
%   bodies (hub / aft elements), whose TElt is not derivable from the
%   frames struct.  Requires TElt to be block-diagonal blkdiag(R,R) --
%   a general 6x6 (e.g. hexapod actuator coordinates) mixes rotation
%   and translation DOFs and needs the full map; error out rather than
%   silently truncate.
%
%   See also: macos.design.dldx_analytic, macos.design.dmet_dx.
arguments
    elts (1,:) double {mustBeInteger, mustBePositive}
end
n = numel(elts);
csys = mmacos('elt_csys_get', double(elts(:)), double(n));
rpt = mmacos('elt_rpt', double(elts(:)'), zeros(3, n), 0, double(n));
bodies = repmat(struct('rpt', zeros(3,1), 'T', eye(3)), 1, n);
for k = 1:n
    M = csys(:, :, k);
    R = M(1:3, 1:3);
    offblk = max(abs([M(1:3,4:6), M(4:6,1:3)]), [], 'all');
    if offblk > 1e-9 || max(abs(M(4:6,4:6) - R), [], 'all') > 1e-9
        error('macos:design:met_bodies:telt', ...
            ['element %d TElt is not blkdiag(R,R) -- its local DOFs mix ' ...
             'rotation and translation; dldx_analytic''s 3x3-triad ' ...
             'model does not apply'], elts(k));
    end
    bodies(k) = struct('rpt', rpt(:, k), 'T', R);
end
end
