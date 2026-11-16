function out = coating(srf, opts)
%MACOS.COATING  Set or query the thin-film coating stack on an element.
%   macos.coating(SRF, 'index', N, 'extinc', K, 'thickness', T) sets a
%   multilayer coating on element SRF (the polarization-path coating,
%   active only when macos.polarization('on') is set).  Layers are
%   ordered OUTERMOST -> INNERMOST (matching the 'Coating=' Rx keyword).
%   N, K, T are equal-length vectors (one entry per layer):
%     'index'     [1xL]  real refractive index n of each layer.
%     'extinc'    [1xL]  extinction coefficient kappa (index = n - i*kappa).
%     'thickness' [1xL]  PHYSICAL layer thickness in element BaseUnits
%                        (NOT waves -- the API is wavelength-agnostic on set).
%   Setting a coating invalidates the cached trace (MODIFY), so re-trace.
%
%   S = macos.coating(SRF) with no options QUERIES the stack and returns
%   a struct: .n_layer, .index [1xL], .extinc [1xL], .thickness [1xL]
%   (physical units).  n_layer=0 means no coating on that element.
%
%   Examples:
%     % bare aluminium (single absorbing layer) on a fold at elt 3:
%     macos.coating(3, 'index', 1.2, 'extinc', 7.0, 'thickness', 0.1);
%     % two-layer AR stack:
%     macos.coating(5, 'index',[1.38 2.30], 'extinc',[0 0], ...
%                      'thickness',[1.0e-7 5.0e-8]);
%     s = macos.coating(3);   % query
%
%   Round-trips: macos.coating(macos.coating(s)) reproduces the stack to
%   round-off (coat_get o coat_set == identity).
%
%   See also: macos.polarization, macos.ray_field.
arguments
    srf            (1,1) double {mustBeInteger, mustBePositive}
    opts.index     double = []
    opts.extinc    double = []
    opts.thickness double = []
end

MAXL = 10;   % mCoat in elt_mod.F

% ---- query mode ------------------------------------------------------
if isempty(opts.index) && isempty(opts.extinc) && isempty(opts.thickness)
    [nL, idx, ext, thk] = mmacos('coat_get', srf, MAXL);
    nL = round(nL);
    out.n_layer   = nL;
    out.index     = idx(1:nL).';
    out.extinc    = ext(1:nL).';
    out.thickness = thk(1:nL).';
    return
end

% ---- set mode --------------------------------------------------------
n = opts.index(:).';  k = opts.extinc(:).';  t = opts.thickness(:).';
L = numel(n);
if L < 1 || L > MAXL
    error('macos:coating:badCount', ...
        'need 1..%d layers (got %d)', MAXL, L);
end
if numel(k) ~= L || numel(t) ~= L
    error('macos:coating:lenMismatch', ...
        '''index'', ''extinc'', ''thickness'' must be equal length.');
end
mmacos('coat_set', srf, L, n, k, t);
end
