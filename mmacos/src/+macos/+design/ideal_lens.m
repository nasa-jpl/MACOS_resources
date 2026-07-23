function L = ideal_lens(f, D, opts)
%MACOS.DESIGN.IDEAL_LENS  Ideal focusing-lens primitive (2-surface singlet).
%   L = macos.design.ideal_lens(F, D) returns a spec for a singlet of
%   effective focal length F and clear diameter D that images the
%   INFINITE conjugate.  MACOS has no paraxial/ideal-lens engine
%   primitive, so an ideal lens is realized as a refractive singlet
%   (two surfaces) with the powered surface's conic constant chosen to
%   be exactly stigmatic at the infinite conjugate.
%
%   All lengths in the SAME units you emit the Rx in (VSG2 uses mm).
%
%   Name-value options:
%     'type'    'conic'    (default) — flat + CONIC surface, conic
%                          constant K=-n^2 nulls on-axis spherical
%                          aberration: EXACTLY stigmatic at infinite
%                          conjugate (verified: rmsWFE ~ 1e-14).
%               'spherical'          — flat + SPHERICAL surface
%                          (lensmaker radius); carries real spherical
%                          aberration (~0.1 wave at F/3.6).  Use for a
%                          more physical Stage-A lens or as an SA source.
%     'mode'    'focus'     (default) — collimated in -> focus out
%                          (imager, e.g. VSG2 L2).  Flat face toward the
%                          collimated side, powered face toward focus.
%               'collimate'          — point source at front focus ->
%                          collimated out (collimator, e.g. VSG2 L1).
%                          Powered face toward the source, flat toward
%                          the collimated side (the reverse of 'focus').
%     'n'       glass index (default 1.5).
%     'thickness' center thickness (default max(D/20, 2)).
%     'name'    base element name (default 'lens').
%
%   Returned struct L:
%     .f .D .n .type .mode .thickness      — echoed inputs
%     .Rmag                                — |R| of the powered surface
%     .surf(1), .surf(2)                   — the two surfaces IN LIGHT
%       ORDER along +axis, each a struct:
%         .name .kind('Refractor') .surface('Flat'|'Conic') .Kr .Kc
%         .indref(index the ray ENTERS past this surface) .ap(=D/2)
%         .dz(axial offset from the lens front vertex)
%     .vertex_span = .thickness            — front-to-back axial length
%
%   Emit the two element blocks with macos.design.ideal_lens_emit.
%
%   CONVENTION (verified empirically 2026-07-23 against the engine, on
%   Apple-silicon gfortran build; see vsg PLAN §4):
%     collimated in +axis -> FLAT front (air->glass) -> powered back
%     (glass->air) focuses to a point at F past the back vertex, needs
%       Kr = +(n-1)*f   (POSITIVE radius magnitude),
%       Kc = -n^2       (conic; 0 for spherical).
%     'collimate' mode mirrors the surface order + reuses the same conic
%     by reversibility of the infinite conjugate.
%
%   See also: macos.design.ideal_lens_emit, macos.design.Telescope.

arguments
    f   (1,1) double {mustBeReal, mustBeNonzero}
    D   (1,1) double {mustBePositive}
    opts.type (1,:) char {mustBeMember(opts.type,{'conic','spherical'})} = 'conic'
    opts.mode (1,:) char {mustBeMember(opts.mode,{'focus','collimate'})} = 'focus'
    opts.n    (1,1) double {mustBePositive} = 1.5
    opts.thickness (1,1) double {mustBeNonnegative} = 0
    opts.name (1,:) char = 'lens'
end

n = opts.n;
t = opts.thickness;  if t == 0, t = max(D/20, 2.0); end

L.f = f;  L.D = D;  L.n = n;  L.type = opts.type;
L.mode = opts.mode;  L.thickness = t;  L.name = opts.name;

% Powered surface: |R| = (n-1)*f (lensmaker, one powered surface).
% Verified engine sign: Kr = +|R| (POSITIVE) focuses collimated -> point.
L.Rmag = (n-1)*abs(f);
Kr_pow = +L.Rmag;                         % positive radius magnitude
switch opts.type
    case 'conic',     Kc_pow = -n^2;      % stigmatic hyperbola
    case 'spherical', Kc_pow = 0.0;       % sphere (has SA)
end
KR_FLAT = -1.0e22;                        % engine flat sentinel

flat = struct('name',[ opts.name '_flat'], 'kind','Refractor', ...
    'surface','Flat',  'Kr',KR_FLAT, 'Kc',0.0,  'ap',D/2);
pow  = struct('name',[ opts.name '_pow'],  'kind','Refractor', ...
    'surface',ternary(strcmp(opts.type,'conic'),'Conic','Conic'), ...
    'Kr',Kr_pow, 'Kc',Kc_pow, 'ap',D/2);
% (spherical still emits Surface=Conic with Kc=0 -- a sphere.)

switch opts.mode
    case 'focus'        % flat front (collimated) -> powered back (focus)
        s1 = flat;  s1.indref = n;    s1.dz = 0.0;    % air->glass
        s2 = pow;   s2.indref = 1.0;  s2.dz = t;      % glass->air
    case 'collimate'    % powered front (source) -> flat back (collimated)
        s1 = pow;   s1.indref = n;    s1.dz = 0.0;    % air->glass
        s2 = flat;  s2.indref = 1.0;  s2.dz = t;      % glass->air
end
L.surf = [s1, s2];
L.vertex_span = t;
end

% ---------------------------------------------------------------------
function y = ternary(c, a, b)
    if c, y = a; else, y = b; end
end
