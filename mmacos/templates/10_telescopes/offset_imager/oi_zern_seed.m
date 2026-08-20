function X = oi_zern_seed(X, P, opts)
%OI_ZERN_SEED  Prepare a design struct for the S5 Zernike stage.
%
%   X = OI_ZERN_SEED(X, P) replaces the aspheres with Zernike surfaces:
%   per mirror, lMon is FROZEN at 1.05x the traced box-centre footprint
%   (the solve doctrine: lMon fixed), the mode set is P.zern_modes, and
%   the coefficients are seeded by LEAST-SQUARES REFIT of the current
%   asphere sag onto the basis's rotationally symmetric members (13, 25,
%   41 = the BornWolf r^4/r^6/r^8 families).  The r^2 remainder of the
%   fit is dropped deliberately -- power belongs to the radii and the
%   FPA refit, both of which the S5 solve keeps open.
%
%   OI_ZERN_SEED(X, P, 'modes', M) overrides the mode set (counter-
%   design probes).  A design with zero aspheres seeds zero coefficients
%   (the sphere+Zernike start).
%
%   See also OFFSET_IMAGER, OI_SOLVE, OFFSET_IMAGER_PARAMS.

    arguments
        X struct
        P struct
        opts.modes (1,:) double = []
    end
    modes = opts.modes;
    if isempty(modes), modes = P.zern_modes; end

    h = footprints_(X, P);
    rho = linspace(0, 1, 201).';
    Rsym = containers.Map('KeyType','double','ValueType','any');
    Rsym(13) = 6*rho.^4 - 6*rho.^2 + 1;
    Rsym(25) = 20*rho.^6 - 30*rho.^4 + 12*rho.^2 - 1;
    Rsym(41) = 70*rho.^8 - 140*rho.^6 + 90*rho.^4 - 20*rho.^2 + 1;
    ksym = intersect([13 25 41], modes);

    for m = 1:3
        lMon = 1.05*h(m);
        A = X.asph(m,:);
        coef = zeros(1, numel(modes));
        if any(A ~= 0)
            r = rho*lMon;
            sag = A(1)*r.^4 + A(2)*r.^6 + A(3)*r.^8;
            B = [ones(size(rho)), rho.^2];
            for k = ksym, B = [B, Rsym(k)]; end %#ok<AGROW>
            c = B\sag;
            for i = 1:numel(ksym)
                coef(modes == ksym(i)) = c(2+i);
            end
        end
        X.zern{m} = struct('modes', modes, 'coef', coef, 'lMon', lMon);
        X.asph(m,:) = 0;
    end
end

% =========================================================================
function h = footprints_(X, P)
    [Xc, G] = oi_close(X, P);
    D = Xc;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
    txt = oi_deck(D);
    sc = oi_score(txt, G, [0 P.offset_deg], 'anchor','center', 'rays', true);
    E = sc.rays{1};
    ie = [1 3 4];
    h = nan(1,3);
    for m = 1:3
        e = E{ie(m)};  ok = e.ok;  ok(1) = false;
        Q = e.pos(1:2,ok);
        h(m) = max(vecnorm(Q - mean(Q,2), 2, 1));
    end
end
