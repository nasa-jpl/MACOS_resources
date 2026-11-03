function B = chain_bundle(G, opts)
%CHAIN_BUNDLE  The multi-field, multi-lambda COLLIMATED bundle through a chain.
%   B = chain_bundle(G, 'nx', nf, 'nlam', nl, 'nring', nr) launches, for each
%   of nf field angles across G.src.fov (centre + ends at nf = 3) and nl
%   wavelengths over G.P.band_m, a disc of rays of diameter G.src.D_src
%   parallel to the field direction, centred on the CHIEF that chain_aim
%   puts through the stop surface's vertex (G.iStop), and traces every ray
%   with chain_trace.  Same output schema as spectrometer_geom's bundle_, so
%   spectrometer_clearance and the footprint tool read it unchanged:
%   B.P (3, nRay, nSurf+1): the launch point then every surface hit;
%   B.D the directions after each station; B.ok; B.meta = [field, lambda, k].
%   Station 0 is the launch plane G.launch (a plane ahead of the frontmost
%   optic), so the first leg is the INCOMING beam -- the leg the unobscuring
%   has to clear.
    arguments
        G struct
        opts.nx (1,1) double = 3
        opts.nlam (1,1) double = 3
        opts.nring (1,1) double = 2
        opts.fields (1,:) double = []
        opts.lams (1,:) double = []
    end
    fields = opts.fields;
    if isempty(fields), fields = linspace(-G.src.fov/2, G.src.fov/2, opts.nx); end
    lams = opts.lams;
    if isempty(lams)
        if isfield(G, 'P') && isfield(G.P, 'band_m') && opts.nlam > 1
            lams = linspace(G.P.band_m(1), G.P.band_m(2), opts.nlam);
        else
            lams = G.src.lambda_c;
        end
    end
    r = G.src.D_src/2;  pts_uv = disc_(r, opts.nring);
    nS = numel(G.surf);  P = [];  D = [];  ok = logical([]);  meta = [];
    for fi = 1:numel(fields)
        d = G.field_dir(fields(fi));
        ex = cross([0;1;0], d);  ex = ex/norm(ex);  ey = cross(d, ex);
        for lam = lams
            if isfield(G, 'launch_field')             % an end-to-end chain seeds its grating aim from the telescope's
                [p0, ~, okA] = G.launch_field(fields(fi), lam);
            else
                [p0, okA] = chain_aim(G.surf, G.launch, d, G.iStop, lam, G);
            end
            if ~okA, continue; end
            for k = 1:size(pts_uv, 2)
                p = p0 + ex*pts_uv(1,k) + ey*pts_uv(2,k);
                [pts, dr, okk] = chain_trace(G.surf, p, d, lam, G);
                if ~okk, continue; end
                P(:, end+1, :) = reshape([p, pts], 3, 1, nS+1);           %#ok<AGROW>
                D(:, end+1, :) = reshape([d, dr], 3, 1, nS+1);            %#ok<AGROW>
                ok(end+1, 1) = true;  meta(end+1, :) = [fields(fi), lam, k];   %#ok<AGROW>
            end
        end
    end
    B.P = P;  B.D = D;  B.ok = ok;  B.meta = meta;
    B.nx = numel(fields);  B.nlam = numel(lams);  B.nring = opts.nring;
    B.fields = fields;  B.lams = lams;
end

function uv = disc_(r, nring)
%DISC_  Centre + nring rings of 8i points out to radius r (the Dyson cone_ pattern).
    uv = [0; 0];
    for i = 1:nring
        a = r*i/nring;  m = 8*i;  ph = 2*pi*(0:m-1)/m;
        uv = [uv, [a*cos(ph); a*sin(ph)]];   %#ok<AGROW>
    end
end
