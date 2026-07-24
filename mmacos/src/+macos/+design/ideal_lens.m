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
%                          (imager, e.g. VSG2 L2).
%               'collimate'          — point source at front focus ->
%                          collimated out (collimator, e.g. VSG2 L1).
%               BOTH modes use the SAME surface order: FLAT front (air->
%                          glass) then POWERED back (glass->air).  The
%                          powered face must face the glass->air step to
%                          work at the infinite conjugate; reversing the
%                          order (powered front) does NOT collimate
%                          (~7000 waves of defocus -- verified).  'mode'
%                          currently only documents intent + picks the
%                          default conjugate for future placement helpers.
%     'n'       glass index (default 1.5).
%     'thickness' center thickness.  Default is SAG-AWARE: the exact
%                 conic sag of the powered surface at the clear
%                 semi-diameter D/2, plus an edge margin, so the two
%                 surfaces never overlap inside the beam footprint (an
%                 overlap silently corrupts the trace -- this was the
%                 VSG2 L1 collimation bug: a 5mm-thick f=722 lens whose
%                 front sag is 6.85mm at r=70).  Override for a specific
%                 mechanical thickness.
%     'edge_margin' minimum glass at the rim beyond the sag (default 2).
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
%   CONVENTION (verified empirically against the engine, Apple-silicon
%   gfortran build; see vsg PLAN §4):
%     FLAT front (air->glass) -> POWERED back (glass->air), for BOTH
%     focus and collimate, needs
%       Kr = +(n-1)*f   (POSITIVE radius magnitude).
%     Kc: for a FOCUS lens (collimated in) Kc=-n^2 is stigmatic to
%     ~1e-14.  For a COLLIMATE lens (point source in) the stigmatic Kc
%     is NOT -n^2 (empirically ~ -1.5 for n=1.5, F/5); Kc=0 (SPHERICAL)
%     is the recommended FIRST CUT -- it collimates cleanly with ~20
%     waves of pure spherical aberration and NO defocus, a sane seed for
%     CALIB to refine the conic from.  Reversing the surface order does
%     NOT collimate (defocus, ~7000 waves).
%
%   See also: macos.design.ideal_lens_emit, macos.design.Telescope.

arguments
    f   (1,1) double {mustBeReal, mustBeNonzero}
    D   (1,1) double {mustBePositive}
    opts.type (1,:) char {mustBeMember(opts.type,{'conic','spherical'})} = 'conic'
    opts.mode (1,:) char {mustBeMember(opts.mode,{'focus','collimate'})} = 'focus'
    opts.n    (1,1) double {mustBePositive} = 1.5
    opts.thickness (1,1) double {mustBeNonnegative} = 0
    opts.edge_margin (1,1) double {mustBeNonnegative} = 2.0
    opts.name (1,:) char = 'lens'
end

n = opts.n;
% Sag-aware default thickness: the powered surface's exact conic sag at
% the clear semi-diameter must fit inside the glass, else the flat and
% powered faces cross within the beam and the trace is garbage.
Rmag = (n-1)*abs(f);  a = D/2;
Kc_pow_ = 0;  if strcmp(opts.type,'conic'), Kc_pow_ = -n^2; end
sag = conic_sag(Rmag, Kc_pow_, a);
t = opts.thickness;  if t == 0, t = sag + opts.edge_margin; end
assert(t > sag, ['ideal_lens: thickness %.3g < powered-surface sag %.3g ' ...
    'at r=%.3g -> surfaces overlap in the beam. Increase thickness.'], t, sag, a);

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

% FLAT FRONT (air->glass) -> POWERED BACK (glass->air) for BOTH modes.
% Verified empirically (2026-07-24): a point source through flat-front/
% powered-back collimates (rmsWFE responds to Kc: ~21 waves spherical,
% ~few waves conic).  The REVERSED order (powered front) does NOT
% collimate -- it leaves ~7000 waves of pure DEFOCUS regardless of Kc.
% The infinite conjugate is reversible, but the powered surface must
% face the SAME index step (glass->air) either way, so the surface order
% is identical for focus and collimate.
s1 = flat;  s1.indref = n;    s1.dz = 0.0;    % air->glass
s2 = pow;   s2.indref = 1.0;  s2.dz = t;      % glass->air
L.surf = [s1, s2];
L.vertex_span = t;
end

% ---------------------------------------------------------------------
function y = ternary(c, a, b)
    if c, y = a; else, y = b; end
end

% ---------------------------------------------------------------------
function z = conic_sag(Rmag, Kc, r)
%CONIC_SAG  Exact sag of a conic surface of vertex radius Rmag, conic Kc,
%   at radial height r.  z = r^2/(R(1+sqrt(1-(1+K)r^2/R^2))).
%   Falls back to the parabolic approx if the radicand goes negative
%   (r beyond the conic's real extent).
    c = 1/Rmag;
    rad = 1 - (1+Kc)*c^2*r^2;
    if rad <= 0
        z = 0.5*c*r^2;              % parabolic fallback
    else
        z = c*r^2 / (1 + sqrt(rad));
    end
end
