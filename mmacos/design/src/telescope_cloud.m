function cloud = telescope_cloud(GD, P, opts)
%TELESCOPE_CLOUD  The spectrometer's bodies as a point cloud for the telescope's clearance wall.
%   cloud = telescope_cloud(GD, P) samples, from the Dyson chain GD and the
%   runner parameters P (mount, slit mask, detector package), every body the
%   telescope's legs must clear: each optical surface's aperture disc
%   (footprint + mount margin, on the rim and at the centre, lifted onto the
%   sphere for the block's convex face), the slit mask's corners and the
%   FPA package's corners -- the same bodies spectrometer_clearance scores at
%   2 mm, coarsely.  Built once per solve.
    arguments
        GD struct
        P struct
        opts.nring (1,1) double = 2
    end
    mount = fld_(P, 'mount_margin_m', 5e-3);
    F = GD.footprints('nx', 3, 'nlam', 3, 'nring', opts.nring);
    pts = [];  ph = linspace(0, 2*pi, 33);  ph(end) = [];
    for k = 1:numel(GD.surf)
        S = GD.surf(k);
        if strcmp(S.act, 'stop'), continue; end
        R = F(k).radius + mount;
        u = [0, R*cos(ph), 0.5*R*cos(ph)] + F(k).xc;  v = [0, R*sin(ph), 0.5*R*sin(ph)] + F(k).yc;
        q = S.vpt(:) + F(k).xap(:)*u + F(k).yap(:)*v;
        if any(strcmp(S.kind, {'sphere', 'asph'}))
            psi = S.psi(:)/norm(S.psi);
            for i = 1:size(q, 2)
                w = q(:, i) - S.C(:);  wt = w - (w'*psi)*psi;  s2 = S.R^2 - wt'*wt;
                if s2 > 0, q(:, i) = S.C(:) + wt + sign((S.vpt(:) - S.C(:))'*psi)*sqrt(s2)*psi; end
            end
        end
        pts = [pts, q];   %#ok<AGROW>
    end
    % the slit mask and the FPA package as boxes (corners + face centres)
    sm = fld_(P, 'slit_mask_m', [0.064 0.004 0.001]);
    pts = [pts, box_(GD.slit(:), [1;0;0], [0;1;0], [0;0;1], sm(1), sm(2), [-sm(3) 0])];
    pm = fld_(P, 'pkg_margin_m', 5e-3);  pd = fld_(P, 'pkg_depth_m', 10e-3);  ps = fld_(P, 'pkg_shield_m', 0);
    pts = [pts, box_(GD.fpa.center(:), GD.fpa.xhat(:), GD.fpa.yhat(:), GD.fpa.normal(:), GD.fpa.W + 2*pm, GD.fpa.H + 2*pm, [-pd ps])];
    cloud = pts;
end

function q = box_(c, ex, ey, ez, Lx, Ly, zr)
    [u, v, w] = ndgrid([-Lx/2 0 Lx/2], [-Ly/2 0 Ly/2], zr);
    q = c + ex*u(:)' + ey*v(:)' + ez*w(:)';
end

function v = fld_(P, f, d)
    if isfield(P, f) && ~isempty(P.(f)), v = P.(f); else, v = d; end
end
