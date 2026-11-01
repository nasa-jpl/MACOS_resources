classdef Lens
%MACOS.DESIGN.LENS  An N-surface refractive element: the lens primitive.
%   L = macos.design.Lens('R', [61.6 -44.2 -128.9], 't', [6 2.5], ...
%                         'glass', {'N-BK7', 'N-SF6'}, 'D', 25, 'name', 'AC254')
%
%   A lens is its surfaces and the media between them, in the CODE V sign
%   convention: light travels +z (along the chief ray, wherever the lens is
%   placed), surface 1 first, and a radius is POSITIVE when its centre of
%   curvature lies to the +z side of its vertex.  Air surrounds the lens on
%   both sides.  Lengths are in the units of whatever uses the lens (the
%   Bench's BaseUnits; mm everywhere in the templates); wavelengths in nm.
%
%   Properties
%     R          1xN signed vertex radii; Inf = plano
%     K          1xN conic constants (0 = sphere; default all 0)
%     t          1x(N-1) axial thicknesses, vertex k to vertex k+1
%     glass      1x(N-1) cell: the medium AFTER surface k -- a catalog name
%                ('N-BK7', resolved by macos.design.glass from the engine's
%                table), 'air' (an air-spaced element), or a NUMBER (a fixed,
%                non-dispersive index, e.g. the 1.5 of Bench.add_lens)
%     D          clear diameter the face-overlap guard is checked at
%                (NaN = no guard unless semidia is given)
%     semidia    [] or 1xN clear semi-diameters; when given, each surface is
%                emitted with a circular aperture of that radius
%     surf_names {} or 1xN element names (default <name>S1 .. <name>SN)
%     name       label
%     source     provenance (a catalog part's vendor and drawing, a textbook)
%
%   Methods
%     P = L.paraxial(lambda_nm)  first-order properties (y-nu, no engine):
%           .efl .power .bfl (last vertex -> rear focus) .ffl (first vertex
%           -> front focus, measured along -z, so positive = in front)
%           .h1 (front principal plane, z from vertex 1) .h2 (rear principal
%           plane, z from the last vertex) .n (gap indices) .c (curvatures)
%           .z (vertex positions) .y .u (the unit marginal ray, y1 = 1, u0 = 0)
%     n = L.index(lambda_nm)     1x(N-1) gap indices
%     s = L.sag(k, r)            signed sag of surface k at height r
%     e = L.edge_thickness(r)    1x(N-1) axial gap at height r (r scalar or
%                                1xN per surface; default the clear radius)
%     L2 = L.flip()              the same lens used backwards
%     L2 = L.scale(s)            every length times s
%
%   Static constructors
%     macos.design.Lens.plano(f, D, ...)   the plano singlet Bench.add_lens
%                                          builds, as a Lens (same options,
%                                          same thickness rule, same names)
%     macos.design.Lens.from_iodc(Li)      an IODC 2010 green-lens spec
%                                          (challenges/iodc2010/iodc2010_lens)
%
%   Placing a lens on a bench: macos.design.Bench.add_optic(L, dist).  The
%   paraxial solve is the textbook y-nu trace (Welford; Kingslake): refraction
%   n'u' = n u - y c (n' - n), transfer y' = y + t u' -- generalized from
%   challenges/iodc2010/iodc2010_paraxial.
%
%   See also macos.design.glass, macos.design.Bench, macos.design.lens_catalog.

properties
    R          (1,:) double = [Inf Inf]
    K          (1,:) double = []
    t          (1,:) double = 1
    glass      (1,:) cell   = {1.5}
    D          (1,1) double = NaN
    semidia    (1,:) double = []
    surf_names (1,:) cell   = {}
    name       (1,:) char   = 'lens'
    source     (1,:) char   = ''
end

properties (Dependent)
    N           % number of surfaces
    z           % 1xN vertex positions, z(1) = 0
    span        % first to last vertex (sum of t)
end

methods
    function L = Lens(opts)
        arguments
            opts.R          (1,:) double = [Inf Inf]
            opts.K          (1,:) double = []
            opts.t          (1,:) double = 1
            opts.glass      (1,:) cell   = {1.5}
            opts.D          (1,1) double = NaN
            opts.semidia    (1,:) double = []
            opts.surf_names (1,:) cell   = {}
            opts.name       (1,:) char   = 'lens'
            opts.source     (1,:) char   = ''
        end
        f = fieldnames(opts);
        for i = 1:numel(f), L.(f{i}) = opts.(f{i}); end
        if isempty(L.K), L.K = zeros(size(L.R)); end
        L.validate_();
    end

    function v = get.N(L),      v = numel(L.R); end
    function v = get.z(L),      v = [0, cumsum(L.t)]; end
    function v = get.span(L),   v = sum(L.t); end

    % -----------------------------------------------------------------
    function c = curv(L)
        %CURV  1xN signed curvatures (0 = plano).
        c = 1 ./ L.R;
        c(~isfinite(L.R)) = 0;
    end

    function n = index(L, lambda_nm)
        %INDEX  1x(N-1) indices of the gaps at LAMBDA_NM.
        n = zeros(1, L.N - 1);
        for k = 1:L.N - 1, n(k) = macos.design.glass(L.glass{k}, lambda_nm); end
    end

    function P = paraxial(L, lambda_nm)
        %PARAXIAL  First-order properties at LAMBDA_NM (see class help).
        n = L.index(lambda_nm);
        c = L.curv();
        [y, u] = ynu_(L.t, [n 1], c, 1, 0);
        assert(u(end) ~= 0, 'Lens.paraxial: %s is afocal (u'' = 0)', L.name);
        efl = -1 / u(end);
        bfl = -y(end) / u(end);
        % front focus: the same trace through the lens reversed
        Lf = L.flip();
        [yf, uf] = ynu_(Lf.t, [fliplr(n) 1], Lf.curv(), 1, 0);
        ffl = -yf(end) / uf(end);
        P = struct('efl', efl, 'power', 1 / efl, 'bfl', bfl, 'ffl', ffl, ...
            'h1', efl - ffl, 'h2', bfl - efl, 'n', n, 'c', c, 'z', L.z, ...
            'y', y, 'u', u, 'lambda_nm', lambda_nm);
    end

    function s = sag(L, k, r)
        %SAG  Signed sag of surface K at height R (NaN past the surface's extent).
        c = L.curv();  c = c(k);
        if c == 0, s = zeros(size(r)); return; end
        a = 1 - (1 + L.K(k)) * c^2 * r.^2;
        s = c * r.^2 ./ (1 + sqrt(max(a, 0)));
        s(a < 0) = NaN;
    end

    function e = edge_thickness(L, r)
        %EDGE_THICKNESS  1x(N-1) axial gap between faces k and k+1 at height R.
        %   R scalar (every face) or 1xN (per face, the gap taking the larger
        %   of its two faces' heights); default: L.semidia, else L.D/2.
        if nargin < 2
            if ~isempty(L.semidia), r = L.semidia; else, r = L.D / 2; end
        end
        if isscalar(r), r = r * ones(1, L.N); end
        e = zeros(1, L.N - 1);
        for k = 1:L.N - 1
            rk = max(r(k), r(k + 1));
            e(k) = L.t(k) - L.sag(k, rk) + L.sag(k + 1, rk);
        end
    end

    function L2 = flip(L)
        %FLIP  The lens used backwards (surface N first).
        L2 = L;
        L2.R = -fliplr(L.R);
        L2.R(isinf(L2.R)) = Inf;          % a plano face stays +Inf
        L2.K = fliplr(L.K);
        L2.t = fliplr(L.t);
        L2.glass = fliplr(L.glass);
        if ~isempty(L.semidia), L2.semidia = fliplr(L.semidia); end
        if ~isempty(L.surf_names), L2.surf_names = fliplr(L.surf_names); end
    end

    function L2 = scale(L, s)
        %SCALE  Every length times S (radii, thicknesses, diameters).
        assert(s > 0, 'Lens.scale: the factor must be positive');
        L2 = L;
        L2.R = L.R * s;  L2.t = L.t * s;  L2.D = L.D * s;
        if ~isempty(L.semidia), L2.semidia = L.semidia * s; end
    end

    function names = names(L)
        %NAMES  The element name of every surface.
        if ~isempty(L.surf_names), names = L.surf_names; return; end
        names = arrayfun(@(k) sprintf('%sS%d', L.name, k), 1:L.N, 'UniformOutput', false);
    end

    function k = powered(L)
        %POWERED  Indices of the surfaces with curvature.
        k = find(isfinite(L.R));
    end
end

methods (Access = private)
    function validate_(L)
        N = numel(L.R);
        assert(N >= 2, 'Lens: at least two surfaces');
        assert(all(L.R ~= 0), 'Lens: a radius of 0 is not a surface (Inf = plano)');
        assert(numel(L.K) == N, 'Lens: K needs one conic per surface');
        assert(numel(L.t) == N - 1 && all(L.t > 0), 'Lens: t needs N-1 positive thicknesses');
        assert(numel(L.glass) == N - 1, 'Lens: glass needs one medium per gap (N-1)');
        for k = 1:N - 1
            g = L.glass{k};
            assert((ischar(g) || isstring(g)) || (isnumeric(g) && isscalar(g) && g >= 1), ...
                'Lens: glass{%d} must be a catalog name, ''air'' or an index >= 1', k);
        end
        if ~isempty(L.semidia)
            assert(numel(L.semidia) == N && all(L.semidia > 0), 'Lens: semidia must be 1xN positive');
        end
        if ~isempty(L.surf_names)
            assert(numel(L.surf_names) == N, 'Lens: surf_names needs one name per surface');
        end
    end
end

methods (Static)
    function L = plano(f, D, opts)
        %PLANO  The plano singlet macos.design.Bench.add_lens builds, as a Lens.
        %   L = macos.design.Lens.plano(F, D, 'mode', 'focus', 'n', 1.5, ...)
        %   takes add_lens's arguments and options ('mode', 'n', 'Kc',
        %   'thickness', 'edge_margin', 'name') and applies its rules exactly:
        %   R = (n-1) f on the powered face, the conic seed (-n^2 focus, 0
        %   collimate), the centre thickness sag(D/2) + edge_margin when none
        %   is given, the face names <name>pow / <name>flat.  'focus' = POWERED
        %   front (convex to the incoming light), 'collimate' = FLAT front,
        %   powered back.  Bench.add_optic(L, dist) of this lens emits the
        %   same elements add_lens(dist, f, D, ...) does (tLens pins it on the
        %   ZWFS test arm, byte for byte).
        arguments
            f (1,1) double {mustBePositive}
            D (1,1) double {mustBePositive}
            opts.mode (1,:) char {mustBeMember(opts.mode, {'collimate', 'focus'})} = 'focus'
            opts.n    (1,1) double {mustBePositive} = 1.5
            opts.Kc   (1,1) double = NaN
            opts.thickness (1,1) double {mustBeNonnegative} = 0
            opts.edge_margin (1,1) double {mustBeNonnegative} = 2.0
            opts.name (1,:) char = 'L'
        end
        n = opts.n;  Rm = (n - 1) * f;
        Kc = opts.Kc;
        if isnan(Kc)
            if strcmp(opts.mode, 'focus'), Kc = -n^2; else, Kc = 0; end
        end
        sag = macos.design.Bench.conic_sag(Rm, Kc, D/2);
        t = opts.thickness;  if t == 0, t = sag + opts.edge_margin; end
        assert(t > sag, ['Lens.plano: thickness %.3g < powered-surface ' ...
            'sag %.3g at r=%.3g -> faces overlap in the beam.'], t, sag, D/2);
        nm = opts.name;
        switch opts.mode
            case 'focus'
                L = macos.design.Lens('R', [Rm Inf], 'K', [Kc 0], 't', t, 'glass', {n}, ...
                    'D', D, 'surf_names', {[nm 'pow'], [nm 'flat']}, 'name', nm, ...
                    'source', 'Bench.add_lens plano singlet (focus)');
            case 'collimate'
                L = macos.design.Lens('R', [Inf -Rm], 'K', [0 Kc], 't', t, 'glass', {n}, ...
                    'D', D, 'surf_names', {[nm 'flat'], [nm 'pow']}, 'name', nm, ...
                    'source', 'Bench.add_lens plano singlet (collimate)');
        end
    end

    function L = from_iodc(Li)
        %FROM_IODC  An IODC 2010 green-lens spec (iodc2010_lens) as a Lens.
        %   The radii come from Li.Rk when present (a relaxed lens), else
        %   Li.sgn * Li.R (0 = plano); the gaps are Li.t(1:N-1) with
        %   Li.glass(1:N-1); the image distance Li.t(N) and the stop are the
        %   evaluator's, not the lens's, and are not carried.
        N = Li.N;
        if isfield(Li, 'Rk') && ~isempty(Li.Rk)
            R = Li.Rk;
        else
            R = Li.sgn * Li.R;  R(Li.sgn == 0) = Inf;
        end
        sd = [];
        if isfield(Li, 'semidia') && ~isempty(Li.semidia) && all(isfinite(Li.semidia)), sd = Li.semidia; end
        L = macos.design.Lens('R', R, 't', Li.t(1:N-1), 'glass', Li.glass(1:N-1), ...
            'semidia', sd, 'name', Li.name, 'source', 'challenges/iodc2010 green-lens spec');
    end
end
end

% -------------------------------------------------------------------------
function [y, u] = ynu_(t, n_after, c, y1, u0)
% y(k): height AT surface k; u(k): angle AFTER surface k (paraxial, tan).
% n_after(k) is the index after surface k (the last is the image space).
N = numel(c);  y = zeros(1, N);  u = zeros(1, N);
yk = y1;  uk = u0;  nk = 1;
for k = 1:N
    y(k) = yk;
    np = n_after(k);
    uk = (nk * uk - yk * c(k) * (np - nk)) / np;
    u(k) = uk;  nk = np;
    if k < N, yk = yk + t(k) * uk; end
end
end
