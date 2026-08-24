classdef Bench < handle
%MACOS.DESIGN.BENCH  Sequential optical-bench builder (add-optic utilities).
%   B = macos.design.Bench(NAME, ...) starts a bench layout at a point
%   source and lets you append optics ONE AT A TIME, each placed a given
%   DISTANCE along the current chief ray.  The builder tracks the chief
%   ray analytically as it goes -- reflecting it at mirrors and Snell-
%   refracting it through flat interfaces (so a tilted beam-splitter
%   plate gets its real lateral walk-off) -- and every element is
%   emitted already CENTERED on the chief ray (VptElt = RptElt = the
%   chief crossing), which is exactly the "recenter on the beam" step a
%   hand-laid folded bench otherwise needs after the fact.
%
%   The result is a folded 3-D bench prescription with real angles of
%   incidence (no unfolded paraxial fiction) that loads and traces with
%   zero vignetting, ready for staged optimization (collimate / focus /
%   conjugate placement).  See templates/40_benches/bench_layout.
%
%   Constructor options (name-value):
%     'pos'       source point (3x1, BaseUnits=mm).  Default [0;0;0].
%     'dir'       initial chief direction (3x1).      Default [1;0;0].
%     'wavelen'   wavelength in mm.                   Default 6.328e-4 (HeNe).
%     'aperture'  point-source cone: FULL cone angle in RADIANS (the
%                 MACOS point-source convention -- N.A. 0.1 means
%                 2*asin(0.1)=0.20033, NOT a diameter).  Default 0.1.
%     'ngridpts'  source ray-grid sampling.           Default 63.
%     'zsource'   Rx zSource value.                   Default 25.
%
%   Add-optic methods -- DIST is measured from the current chief-ray
%   point ALONG the chief ray; each returns the element index (indices
%   are assigned sequentially, 1-based, in light order):
%     add_baffle(dist, radius)          Obscuration w/ circular aperture
%                                       (the pupil stop of the bench).
%     add_lens(dist, f, D, ...)         plano singlet, 2 refractor
%                                       surfaces.  'mode':
%             'collimate' -- FLAT front / POWERED back, Kr=+(n-1)f.
%                            DIST locates the POWERED (back) surface;
%                            put it f from the source point to collimate.
%             'focus'     -- POWERED front / flat back, Kr=-(n-1)f
%                            (converging, psi=+chief pairing).  DIST
%                            locates the POWERED (front) surface; focus
%                            lands ~f beyond it.
%     add_bs_reflect(dist, out, ...)    beam-splitter REFLECT encounter:
%                                       flat mirror turning the chief to
%                                       direction OUT.  Second output is
%                                       a BS token recording the plate
%                                       (normal, thickness, index) for
%                                       the later transmit pass.
%     add_mirror(dist, ...)             flat mirror.  'out',[3x1] turns
%                                       the chief; 'out','retro'
%                                       (default) is normal-incidence
%                                       retro-reflection (e.g. a DM
%                                       being probed on-axis).
%     add_bs_transmit(bs)               TRANSMIT pass through the plate
%                                       in token BS (from either side --
%                                       side-agnostic): two flat
%                                       refractor surfaces centered on
%                                       the Snell-refracted chief (real
%                                       walk-off).  No DIST: the plate
%                                       is already placed.
%     add_bs_reflect_return(bs)         recombining reflection off the
%                                       BS with its REAL glass path:
%                                       back-face in -> internal reflect
%                                       at the coating -> back-face out
%                                       (two glass transits -- what a
%                                       compensator plate balances).
%     plate(dist, psi, ...)             define a plane-parallel plate
%                                       token (e.g. a COMPENSATOR) with
%                                       no element added; pass it to
%                                       add_bs_transmit per pass.
%     add_fold(dist, out)               fold mirror (OUT required).
%     add_oap(dist, out, ...)           off-axis parabola section
%                                       (RptElt=pole, VptElt=parent
%                                       vertex); 'mode' 'collimate' or
%                                       'focus', 'focus_dist' sets the
%                                       conjugate.  See method help.
%     add_relay(dist, ...)              Offner-type concentric 3-mirror
%                                       1:1 ring-field relay ('R',
%                                       'ring_offset', 'focus_dist',
%                                       'side').  See method help.
%     add_reference(dist, name)         passive Reference plane marker
%                                       (e.g. a focal-mask site).
%     add_polarizer(dist, axis)         ideal linear polarizer (TrPolarizer);
%                                       AXIS = transmission axis (3-vector).
%                                       Transmissive + geometrically inert;
%                                       use in a collimated normal-incidence
%                                       leg.  Requires ifPol.
%     add_waveplate(dist, axis, R)      linear retarder (WavePlate); AXIS =
%                                       fast axis, R = retardance in WAVES at
%                                       the bench Wavelen (0.25 QWP / 0.5 HWP).
%                                       A double-passed plate is TWO WavePlate
%                                       elements sharing one global fast axis.
%     add_detector(dist, name)          FocalPlane (last element).
%
%   Output / inspection:
%     emit(file)      write the .in prescription (header + elements +
%                     the nOutCord/Tout terminator block MACOS requires).
%     print_chain     one line per element: index, name, kind, path
%                     length from the source, chief crossing.
%     sketch(...)     annotated XY-plane layout schematic: chief-ray
%                     polyline, element footprints + names, and every
%                     leg labeled with its length -- pass 'labels'
%                     (one per element) to name each leg after YOUR
%                     input parameter, so the figure documents exactly
%                     what a user must specify.  Returns the figure.
%
%   CONVENTIONS BAKED IN (engine-verified on the VSG2-class bench,
%   2026-07-24 -- these cost real debug time, do not "fix" them):
%     * A collimating singlet must be FLAT front / POWERED back with
%       Kr=+(n-1)f; the reversed order does NOT collimate (~7000 waves
%       of pure defocus).
%     * A converging surface pairs (Kr<0, psi=+chief) -- same psi sense
%       as the rest of the transmissive train.  (Kr>0, psi=-chief) is
%       the only other converging pair; mixing them diverges.
%     * Mirror normal: psi = normalize(d_out - d_in) -- faces the
%       incoming beam.  Retro: psi = -d_in.
%     * CURVED MIRRORS (the MACOS curvature convention, Dave): Kr is
%       ALWAYS NEGATIVE and psi ALWAYS points in the CONCAVE direction
%       (from the surface toward its center of curvature / open side).
%       A convex working side (e.g. an Offner secondary) is GEOMETRY --
%       the beam simply hits the other side -- never a sign flip.
%       add_oap and add_relay emit this pairing.
%     * Lens thickness must exceed the powered surface's edge sag or
%       the faces cross inside the beam (silent garbage trace); the
%       default thickness is sag-aware.
%     * Every Rx ends with nOutCord=5 + a Tout block -- the parser's
%       element-list terminator.  emit() writes it.
%
%   See also: macos.design.ideal_lens, macos.design.Telescope.

properties
    name     (1,:) char   = 'bench'
    baseunits (1,:) char  = 'mm'        % Rx BaseUnits/WaveUnits.  Every
                                        % LENGTH handed to this builder --
                                        % distances, radii, thicknesses,
                                        % wavelen -- is in these units; the
                                        % builder never converts.  Default
                                        % 'mm' keeps every existing bench
                                        % bit-identical.  Set 'm' to build a
                                        % bench that splices onto a
                                        % metre-based telescope deck.
    wavelen  (1,1) double = 6.328e-4    % baseunits (HeNe in mm)
    aperture (1,1) double = 0.1         % FULL cone angle, radians (point src)
    zsource  (1,1) double = 25
    ngridpts (1,1) double = 63
    src_pos  (3,1) double = [0;0;0]
    src_dir  (3,1) double = [1;0;0]
    pos      (3,1) double               % current chief-ray point
    dir      (3,1) double               % current chief-ray direction (unit)
    path_len (1,1) double = 0           % chief path length from source
    E        (1,:) struct = struct([])  % element list (light order)
end

methods
    function b = Bench(name, opts)
        arguments
            name (1,:) char
            opts.pos      (3,1) double = [0;0;0]
            opts.dir      (3,1) double = [1;0;0]
            opts.wavelen  (1,1) double {mustBePositive} = 6.328e-4
            opts.aperture (1,1) double {mustBePositive} = 0.1
            opts.ngridpts (1,1) double {mustBeInteger, mustBePositive} = 63
            opts.zsource  (1,1) double = 25
            opts.baseunits (1,:) char {mustBeMember(opts.baseunits, ...
                              {'m','cm','mm','um','nm','in','ft'})} = 'mm'
        end
        b.name     = name;
        b.baseunits = opts.baseunits;
        b.src_pos  = opts.pos;
        b.src_dir  = macos.design.Bench.unit(opts.dir);
        b.pos      = b.src_pos;
        b.dir      = b.src_dir;
        b.wavelen  = opts.wavelen;
        b.aperture = opts.aperture;
        b.ngridpts = opts.ngridpts;
        b.zsource  = opts.zsource;
    end

    % -----------------------------------------------------------------
    function i = add_baffle(b, dist, radius, opts)
        %ADD_BAFFLE  Circular pupil stop (Obscuration element).
        arguments
            b
            dist   (1,1) double {mustBePositive}
            radius (1,1) double {mustBePositive}
            opts.name (1,:) char = 'Baffle'
        end
        P = b.step(dist);
        e = b.blank(opts.name, 'Obscuration');
        e.psi = b.dir;  e.vpt = P;
        e.aptype = 'Circular';  e.aprad = radius;
        i = b.push(e);
    end

    % -----------------------------------------------------------------
    function L = add_lens(b, dist, f, D, opts)
        %ADD_LENS  Plano singlet (two refractor surfaces) on the chief ray.
        %   DIST locates the POWERED surface (see class help).  Returns
        %   struct L: .i_flat .i_pow .Kr .Kc .thickness .mode.
        arguments
            b
            dist (1,1) double {mustBePositive}
            f    (1,1) double {mustBePositive}
            D    (1,1) double {mustBePositive}
            opts.mode (1,:) char {mustBeMember(opts.mode,{'collimate','focus'})} = 'focus'
            opts.n    (1,1) double {mustBePositive} = 1.5
            opts.Kc   (1,1) double = NaN
            opts.thickness (1,1) double {mustBeNonnegative} = 0
            opts.edge_margin (1,1) double {mustBeNonnegative} = 2.0
            opts.name (1,:) char = 'L'
        end
        n = opts.n;  R = (n-1)*f;
        % Kc seed: focus lens -> -n^2 (stigmatic at infinite conjugate);
        % collimate lens -> 0 (spherical: collimates with pure SA, no
        % defocus -- the sane optimizer seed; -n^2 is NOT stigmatic here).
        Kc = opts.Kc;
        if isnan(Kc)
            if strcmp(opts.mode,'focus'), Kc = -n^2; else, Kc = 0; end
        end
        sag = macos.design.Bench.conic_sag(R, Kc, D/2);
        t = opts.thickness;  if t == 0, t = sag + opts.edge_margin; end
        assert(t > sag, ['Bench.add_lens: thickness %.3g < powered-surface ' ...
            'sag %.3g at r=%.3g -> faces overlap in the beam.'], t, sag, D/2);

        flt = b.blank([opts.name 'flat'], 'Refractor');  flt.psi = b.dir;
        pow = b.blank([opts.name 'pow'],  'Refractor');  pow.psi = b.dir;
        pow.surface = 'Conic';  pow.Kc = Kc;
        switch opts.mode
            case 'collimate'   % FLAT front (air->glass), POWERED back, Kr=+R
                assert(dist > t, 'Bench.add_lens: dist %.3g <= thickness %.3g.', dist, t);
                flt.vpt = b.step(dist - t);  flt.indref = n;
                L.i_flat = b.push(flt);
                pow.vpt = b.step(t);         pow.indref = 1.0;  pow.Kr = +R;
                L.i_pow = b.push(pow);
            case 'focus'       % POWERED front (Kr<0 converging), flat back
                pow.vpt = b.step(dist);    pow.indref = n;    pow.Kr = -R;
                L.i_pow = b.push(pow);
                flt.vpt = b.step(t);       flt.indref = 1.0;
                L.i_flat = b.push(flt);
        end
        L.Kr = pow.Kr;  L.Kc = Kc;  L.thickness = t;  L.mode = opts.mode;
    end

    % -----------------------------------------------------------------
    function [i, bs] = add_bs_reflect(b, dist, out, opts)
        %ADD_BS_REFLECT  Beam-splitter reflect encounter; returns BS token.
        arguments
            b
            dist (1,1) double {mustBePositive}
            out  (3,1) double
            opts.thickness (1,1) double {mustBePositive} = 10
            opts.n         (1,1) double {mustBePositive} = 1.5
            opts.name      (1,:) char = 'BS'
        end
        out = macos.design.Bench.unit(out);
        P = b.step(dist);
        psi = macos.design.Bench.unit(out - b.dir);   % faces the incoming beam
        e = b.blank([opts.name 'refl'], 'Reflector');
        e.psi = psi;  e.vpt = P;  e.extinc = 1e22;
        i = b.push(e);
        bs = struct('vpt', P, 'psi', psi, 'thickness', opts.thickness, ...
                    'n', opts.n, 'name', opts.name);
        b.dir = out;
    end

    % -----------------------------------------------------------------
    function i = add_mirror(b, dist, opts)
        %ADD_MIRROR  Flat mirror; 'out','retro' (default) or a 3-vector.
        arguments
            b
            dist (1,1) double {mustBePositive}
            opts.out = 'retro'
            opts.name (1,:) char = 'M'
            opts.aprad (1,1) double = 0     % 0 = no aperture (ApType=None)
            opts.Kr   (1,1) double = 0     % nonzero: Surface=Conic, vertex
                                           % radius Kr (Kr<0 = concave toward
                                           % the beam for a retro mirror --
                                           % e.g. a weak test-optic figure)
            opts.grid_file (1,:) char = '' % nonempty: Surface=GridData with
                                           % this GridFile (surface figure map,
                                           % e.g. a DM); requires grid_n/grid_dx
            opts.grid_n  (1,1) double = 0  % nGridMat (grid is grid_n x grid_n)
            opts.grid_dx (1,1) double = 0  % GridSrfdx (BaseUnits per sample)
        end
        P = b.step(dist);
        if ischar(opts.out) || isstring(opts.out)
            assert(strcmp(char(opts.out), 'retro'), 'Bench.add_mirror: out must be ''retro'' or a 3-vector.');
            out = -b.dir;
        else
            out = macos.design.Bench.unit(opts.out(:));
            assert(norm(out - b.dir) > 1e-9, ...
                'Bench.add_mirror: out == incoming direction (no mirror can do that).');
        end
        psi = macos.design.Bench.unit(out - b.dir);
        e = b.blank(opts.name, 'Reflector');
        e.psi = psi;  e.vpt = P;  e.extinc = 1e22;
        if opts.Kr ~= 0
            e.surface = 'Conic';  e.Kr = opts.Kr;  e.Kc = 0.0;
        end
        if ~isempty(opts.grid_file)
            % GridData figure map (SrfType 9): grid frame = the element's
            % own local frame (pData=vertex, zData=psi) so pokes localize
            assert(opts.grid_n > 1 && opts.grid_dx > 0, ...
                'Bench.add_mirror: grid_file needs grid_n and grid_dx.');
            e.surface = 'GridData';
            e.gridfile = opts.grid_file;
            e.gridn = opts.grid_n;  e.griddx = opts.grid_dx;
        end
        if opts.aprad > 0, e.aptype = 'Circular';  e.aprad = opts.aprad; end
        i = b.push(e);
        b.dir = out;
    end

    % -----------------------------------------------------------------
    function idx = add_bs_reflect_return(b, bs, opts)
        %ADD_BS_REFLECT_RETURN  Recombining reflection off the plate in
        %   token BS, modeled with its REAL glass path: the returning
        %   beam (e.g. a reference arm coming back from its retro
        %   mirror) enters through the BACK face, reflects INTERNALLY
        %   off the coated psi-side face, and exits through the back
        %   face into the common output port -- three elements
        %   [back-in, coating-reflect, back-out], two glass transits.
        %   (This is what a compensator plate in the other arm
        %   balances.)  Returns the three element indices.
        arguments
            b
            bs (1,1) struct
            opts.tag (1,:) char = 'r'
        end
        psi = bs.psi;  t = bs.thickness;  n = bs.n;
        denom = dot(b.dir, psi);
        assert(abs(denom) > 1e-12, 'Bench.add_bs_reflect_return: chief parallel to plate.');
        % the returning beam must approach from the glass (-psi) side
        vptB = bs.vpt - t*psi;                       % back face point
        sB = dot(vptB - b.pos, psi) / denom;
        assert(sB > 0, 'Bench.add_bs_reflect_return: plate is behind the beam.');
        Pin = b.pos + sB*b.dir;
        dg  = macos.design.Bench.refract(b.dir, psi, 1.0, n);
        sf  = dot(bs.vpt - Pin, psi) / dot(dg, psi); % to the coating
        assert(sf > 0, 'Bench.add_bs_reflect_return: degenerate glass path.');
        Pr  = Pin + sf*dg;
        dr  = macos.design.Bench.reflect(dg, psi);
        sb2 = dot(vptB - Pr, psi) / dot(dr, psi);    % back out
        assert(sb2 > 0, 'Bench.add_bs_reflect_return: degenerate return path.');
        Pout = Pr + sb2*dr;
        dout = macos.design.Bench.refract(dr, psi, n, 1.0);

        e1 = b.blank([bs.name 'bin' opts.tag], 'Refractor');
        e1.psi = psi;  e1.vpt = Pin;   e1.indref = n;    e1.extinc = 0;
        b.path_len = b.path_len + sB;   i1 = b.push(e1);
        e2 = b.blank([bs.name 'cref' opts.tag], 'Reflector');
        e2.psi = psi;  e2.vpt = Pr;    e2.indref = n;    e2.extinc = 1e22;
        b.path_len = b.path_len + sf;   i2 = b.push(e2);
        e3 = b.blank([bs.name 'bout' opts.tag], 'Refractor');
        e3.psi = psi;  e3.vpt = Pout;  e3.indref = 1.0;  e3.extinc = 0;
        b.path_len = b.path_len + sb2;  i3 = b.push(e3);
        idx = [i1, i2, i3];
        b.pos = Pout;  b.dir = dout;
    end

    % -----------------------------------------------------------------
    function O = add_oap(b, dist, out, opts)
        %ADD_OAP  Off-axis parabola section turning the chief to OUT.
        %   An off-axis section in MACOS is the SAME parent conic with
        %   RptElt (the section POLE, where the beam hits) different
        %   from VptElt (the parent VERTEX); psiElt is the parent AXIS.
        %   The pole is placed DIST along the chief.  Specify the parent
        %   EITHER by its focal length ('f', the usual optical spec) OR by
        %   the conjugate distance ('focus_dist'):
        %     'mode','collimate' -- the incoming chief diverges from a
        %        focus 'focus_dist' BACK along the incoming chief (for
        %        a source-fed OAP that is the source distance); the
        %        reflected beam is collimated along OUT.
        %     'mode','focus'     -- incoming collimated; the reflected
        %        chief focuses 'focus_dist' ahead along OUT.
        %
        %   Parent focal length from the polar equation of the parabola
        %   with the focus at the origin, r = 2f/(1 - cos(theta_polar)),
        %   where theta_polar is measured from the focus->vertex direction
        %   (-axis) to the focus->pole direction; since axis is +/-d_in and
        %   the turn is cth = d_in.OUT, this gives  f_parent = r*(1-cth)/2
        %   and, inverting, the conjugate that realizes a desired f is
        %   focus_dist = 2f/(1-cth) = f/cos^2(AOI)  (AOI = angle of
        %   incidence; turn theta = 180 - 2*AOI).  ARBITRARY fold angles
        %   are supported -- near-normal (small AOI, e.g. 5 deg) gives a
        %   nearly on-axis section (pole ~= vertex), 90-deg folds throw the
        %   vertex fully lateral.
        %
        %   HISTORY: through 2026-07 this used (1+cth) -- correct ONLY at
        %   theta=90 deg (cth=0), the sole regime the OAP tests exercised;
        %   wrong for every other fold.  Fixed to (1-cth); 90-deg results
        %   are bit-identical so back-compat holds.
        %
        %   'aprad' is kept as metadata (sketch footprint + a builder-side
        %   vignetting note) but is NOT emitted as a hard aperture: a
        %   Circular ApVec is applied about VptElt (the parent vertex),
        %   which for an off-axis section sits far from the beam at the
        %   pole, so it would block the whole bundle.  Put functional stops
        %   on flat marker planes (add_reference/add_baffle), whose
        %   vertex == pole.
        %
        %   Returns struct O: .i .f_parent .pole .vertex .focus.
        arguments
            b
            dist (1,1) double {mustBePositive}
            out  (3,1) double
            opts.mode (1,:) char {mustBeMember(opts.mode,{'collimate','focus'})} = 'collimate'
            opts.focus_dist (1,1) double = NaN   % conjugate distance (mm)
            opts.f          (1,1) double = NaN   % parent focal length (mm)
            opts.name (1,:) char = 'OAP'
            opts.aprad (1,1) double = 0
        end
        out = macos.design.Bench.unit(out);
        d_in = b.dir;
        P = b.step(dist);
        cth = dot(d_in, out);
        assert(cth > -1 + 1e-9, 'Bench.add_oap: retro OAP is degenerate.');
        % conjugate r: from a pinned focal length 'f', else explicit focus_dist
        if ~isnan(opts.f)
            assert(opts.f > 0, 'Bench.add_oap: f must be positive.');
            r = 2*opts.f/(1 - cth);          % = f/cos^2(AOI)
        elseif ~isnan(opts.focus_dist)
            assert(opts.focus_dist > 0, 'Bench.add_oap: focus_dist must be positive.');
            r = opts.focus_dist;
        else
            error('Bench.add_oap: provide ''f'' (parent focal length) or ''focus_dist''.');
        end
        f_par = r*(1 - cth)/2;               % parabola polar eqn (see help)
        switch opts.mode
            case 'collimate'    % axis = collimated output direction
                a = out;   Fpt = P - r*d_in;
            case 'focus'        % axis = anti-parallel to collimated input
                a = -d_in; Fpt = P + r*out;
        end
        V = Fpt - f_par*a;
        e = b.blank(opts.name, 'Reflector');
        e.surface = 'Conic';  e.Kr = -2*f_par;  e.Kc = -1.0;
        e.psi = a;            % engine convention: psi = parent axis toward the
                              % open/focus side, paired with KrElt=-|R|
        e.vpt = V;  e.rpt = P;  e.extinc = 1e22;
        e.aprad = opts.aprad; % metadata only (sketch/vignetting); aptype stays
                              % 'None' -- a vertex-framed Circle would block the
                              % off-axis beam (see help).
        O.i = b.push(e);
        O.f_parent = f_par;  O.pole = P;  O.vertex = V;  O.focus = Fpt;
        b.dir = out;
    end

    % -----------------------------------------------------------------
    function O = add_relay(b, dist, opts)
        %ADD_RELAY  Concentric 3-mirror relay (Offner-type), 1:1 ring field.
        %   Relays the focus 'focus_dist' BACK along the incoming chief
        %   to a symmetric image (magnification -1).  All three mirrors
        %   belong to one concentric family about a center C: M1 concave
        %   (Kr=-R), M2 the stop (Kr=-R/2; the beam works its convex
        %   side, but the emission keeps the SAME Kr<0 / psi-toward-
        %   center convention -- convex is geometry, not a sign flip),
        %   M3 a second patch on the M1 sphere.  Object and image lie in
        %   the plane through C.
        %
        %   DIST places the M1 chief-hit along the incoming chief.
        %   'R' is the M1/M3 radius; 'ring_offset' = |object - C| is the
        %   working ring height h (needs |R-h| < focus_dist < R+h);
        %   'side' (+1/-1) picks the fold side.  The chief is traced by
        %   exact sphere reflections, so the emitted poles sit on the
        %   real beam.
        %
        %   Returns O: .i (3 element indices), .C, .R, .h, .image (the
        %   image point), .image_dist (from M3 along the outgoing chief
        %   -- e.g. b.add_reference(O.image_dist, 'Image')).
        arguments
            b
            dist (1,1) double {mustBePositive}
            opts.type (1,:) char {mustBeMember(opts.type,{'offner'})} = 'offner'
            opts.focus_dist (1,1) double {mustBePositive}
            opts.R (1,1) double {mustBePositive}
            opts.ring_offset (1,1) double {mustBePositive}
            opts.side (1,1) double {mustBeMember(opts.side,[-1 1])} = 1
            opts.name (1,:) char = 'Relay'
        end
        d_in = b.dir;
        P1 = b.step(dist);
        Obj = P1 - opts.focus_dist*d_in;
        R = opts.R;  h = opts.ring_offset;  Lm = opts.focus_dist;
        assert(Lm > abs(R - h) + 1e-9 && Lm < R + h - 1e-9, ...
            ['Bench.add_relay: no concentric solution -- need ' ...
             '|R-h|=%.4g < focus_dist=%.4g < R+h=%.4g.'], abs(R-h), Lm, R+h);
        % center C: |C-Obj| = h and |C-P1| = R (two-circle intersection
        % in the plane spanned by the chief and the fold side)
        x  = (Lm^2 + h^2 - R^2)/(2*Lm);
        y2 = h^2 - x^2;
        assert(y2 > 1e-12, 'Bench.add_relay: degenerate ring geometry.');
        v = macos.design.Bench.perp(d_in);
        C = Obj + x*d_in + opts.side*sqrt(y2)*v;

        % chief by exact sphere reflections: M1 (R) -> M2 (R/2) -> M3 (R)
        n1 = macos.design.Bench.unit(P1 - C);
        d2 = macos.design.Bench.reflect(d_in, n1);
        [P2, s2] = macos.design.Bench.sphere_hit(P1, d2, C, R/2);
        n2 = macos.design.Bench.unit(P2 - C);
        d3 = macos.design.Bench.reflect(d2, n2);
        [P3, s3] = macos.design.Bench.sphere_hit(P2, d3, C, R);
        n3 = macos.design.Bench.unit(P3 - C);
        d4 = macos.design.Bench.reflect(d3, n3);

        Ps = {P1, P2, P3};  ns = {n1, n2, n3};
        Rs = [R, R/2, R];   ss = [0, s2, s3];
        tags = {'M1','M2','M3'};
        O.i = zeros(1,3);
        for k = 1:3
            b.path_len = b.path_len + ss(k);
            e = b.blank([opts.name tags{k}], 'Reflector');
            e.surface = 'Conic';  e.Kr = -Rs(k);  e.Kc = 0.0;
            e.psi = -ns{k};        % toward C: the concave direction
            e.vpt = Ps{k};  e.extinc = 1e22;
            O.i(k) = b.push(e);
        end
        b.pos = P3;  b.dir = d4;

        % image: outgoing chief meets the object/image plane through C
        % (plane normal = the relay axis, the chief component perp to the
        % ring direction)
        t = macos.design.Bench.unit(Obj - C);
        a = macos.design.Bench.unit(d_in - dot(d_in, t)*t);
        si = dot(C - P3, a)/dot(d4, a);
        O.image = P3 + si*d4;
        O.image_dist = si;
        O.C = C;  O.R = R;  O.h = h;
    end

    % -----------------------------------------------------------------
    function i = add_fold(b, dist, out, opts)
        %ADD_FOLD  Fold mirror (turn direction required).
        arguments
            b
            dist (1,1) double {mustBePositive}
            out  (3,1) double
            opts.name (1,:) char = 'Fold'
        end
        i = b.add_mirror(dist, 'out', out, 'name', opts.name);
    end

    % -----------------------------------------------------------------
    function tok = plate(b, dist, psi, opts)
        %PLATE  Define a tilted plane-parallel plate token WITHOUT adding
        %   any element or touching the chief -- e.g. a COMPENSATOR
        %   plate, or a substrate this arm only transmits.  DIST places
        %   a reference point on the plate's psi-side face along the
        %   current chief; PSI is the plate normal (glass extends away
        %   from psi).  Feed the token to add_bs_transmit (any number
        %   of passes, from either side) or add_bs_reflect_return.
        arguments
            b
            dist (1,1) double {mustBePositive}
            psi  (3,1) double
            opts.thickness (1,1) double {mustBePositive} = 10
            opts.n         (1,1) double {mustBePositive} = 1.5
            opts.name      (1,:) char = 'Plate'
        end
        tok = struct('vpt', b.pos + dist*b.dir, ...
                     'psi', macos.design.Bench.unit(psi), ...
                     'thickness', opts.thickness, 'n', opts.n, ...
                     'name', opts.name);
    end

    % -----------------------------------------------------------------
    function idx = add_bs_transmit(b, bs, opts)
        %ADD_BS_TRANSMIT  Transmit pass through the plate in token BS.
        %   SIDE-AGNOSTIC: the chief may approach from either face (a
        %   double-passed compensator returns through its back face).
        %   Snell-refracts the chief through the tilted plate; both
        %   surfaces are emitted centered on the refracted chief (real
        %   walk-off).  Returns [i_in i_out].  'tag' suffixes the
        %   element names (distinguish repeated passes).
        arguments
            b
            bs (1,1) struct
            opts.tag (1,:) char = ''
        end
        psi = bs.psi;  t = bs.thickness;
        denom = dot(b.dir, psi);
        assert(abs(denom) > 1e-12, 'Bench.add_bs_transmit: chief parallel to plate.');
        sF = dot(bs.vpt - b.pos, psi) / denom;              % psi-side face
        sB = dot(bs.vpt - t*psi - b.pos, psi) / denom;      % far face
        ss = sort([sF, sB]);
        assert(ss(1) > 0, 'Bench.add_bs_transmit: plate is behind the beam.');
        P1 = b.pos + ss(1)*b.dir;
        d1 = macos.design.Bench.refract(b.dir, psi, 1.0, bs.n);
        if sF <= sB, q2 = bs.vpt - t*psi; else, q2 = bs.vpt; end
        s2 = dot(q2 - P1, psi) / dot(d1, psi);
        assert(s2 > 0, 'Bench.add_bs_transmit: degenerate plate crossing.');
        P2 = P1 + s2*d1;
        d2 = macos.design.Bench.refract(d1, psi, bs.n, 1.0);

        e1 = b.blank([bs.name 'txf' opts.tag], 'Refractor');
        e1.psi = psi;  e1.vpt = P1;  e1.indref = bs.n;  e1.extinc = 0;
        b.path_len = b.path_len + ss(1);       % geometric path bookkeeping
        i1 = b.push(e1);
        e2 = b.blank([bs.name 'txb' opts.tag], 'Refractor');
        e2.psi = psi;  e2.vpt = P2;  e2.indref = 1.0;   e2.extinc = 0;
        b.path_len = b.path_len + s2;
        i2 = b.push(e2);
        idx = [i1, i2];
        b.pos = P2;  b.dir = d2;
    end

    % -----------------------------------------------------------------
    function i = add_reference(b, dist, name)
        %ADD_REFERENCE  Passive Reference plane (e.g. focal-mask site).
        arguments
            b
            dist (1,1) double {mustBePositive}
            name (1,:) char = 'Ref'
        end
        P = b.step(dist);
        e = b.blank(name, 'Reference');
        e.psi = b.dir;  e.vpt = P;  e.zelt = 0;
        i = b.push(e);
    end

    % -----------------------------------------------------------------
    function i = add_polarizer(b, dist, axis, opts)
        %ADD_POLARIZER  Ideal linear polarizer (TrPolarizer element).
        %   add_polarizer(DIST, AXIS) places a TrPolarizer a distance DIST
        %   along the current chief ray.  AXIS is the TRANSMISSION axis as a
        %   3-vector in global coordinates (need not be unit; the engine
        %   projects it into each ray's transverse plane).  Transmissive and
        %   geometrically inert (RefSrf geometry) -- the chief passes
        %   straight through, so it belongs in a COLLIMATED, NORMAL-INCIDENCE
        %   leg (psi = the current chief direction), where the off-normal
        %   material-axis question is identically absent (packet
        %   REVIEW_POL_ELEMENTS_2026-07-27.md).  Requires ifPol; the .in
        %   default axis is what emit() writes (the harness may override at
        %   runtime with macos.polarizer).
        arguments
            b
            dist (1,1) double {mustBePositive}
            axis (:,1) double
            opts.name (1,:) char = 'Polarizer'
        end
        assert(numel(axis) == 3 && norm(axis) > 0, ...
            'Bench.add_polarizer: axis must be a non-zero 3-vector.');
        P = b.step(dist);
        e = b.blank(opts.name, 'TrPolarizer');
        e.psi = b.dir;  e.vpt = P;  e.zelt = 0;
        e.polaxis = axis(:) / norm(axis);
        i = b.push(e);
    end

    % -----------------------------------------------------------------
    function i = add_waveplate(b, dist, axis, retardance, opts)
        %ADD_WAVEPLATE  Linear retarder (WavePlate element).
        %   add_waveplate(DIST, AXIS, R) places a WavePlate a distance DIST
        %   along the chief.  AXIS is the FAST axis (3-vector, global).  R is
        %   the retardance in WAVES at the bench Wavelen (0.25 = quarter-wave,
        %   0.5 = half-wave) -- emit() writes Retardance= in waves and the
        %   parser scales by Wavelen on load, so the plate is fixed glass and
        %   a wavelength sweep is chromatic (same treatment as Coating=).
        %   Transmissive and geometrically inert; belongs in a collimated
        %   normal-incidence leg.  A double-passed physical plate is TWO
        %   WavePlate elements (one each side of the retro) sharing this axis,
        %   the same way the compensator is add_bs_transmit'd twice.
        arguments
            b
            dist (1,1) double {mustBePositive}
            axis (:,1) double
            retardance (1,1) double
            opts.name (1,:) char = 'WavePlate'
        end
        assert(numel(axis) == 3 && norm(axis) > 0, ...
            'Bench.add_waveplate: axis must be a non-zero 3-vector.');
        P = b.step(dist);
        e = b.blank(opts.name, 'WavePlate');
        e.psi = b.dir;  e.vpt = P;  e.zelt = 0;
        e.polaxis = axis(:) / norm(axis);
        e.retard = retardance;
        i = b.push(e);
    end

    % -----------------------------------------------------------------
    function i = add_detector(b, dist, name)
        %ADD_DETECTOR  FocalPlane (terminal element).
        arguments
            b
            dist (1,1) double {mustBePositive}
            name (1,:) char = 'Detector'
        end
        P = b.step(dist);
        e = b.blank(name, 'FocalPlane');
        e.psi = b.dir;  e.vpt = P;  e.zelt = 0;
        i = b.push(e);
    end

    % -----------------------------------------------------------------
    function emit(b, file)
        %EMIT  Write the bench as a MACOS .in prescription.
        assert(~isempty(b.E), 'Bench.emit: no elements.');
        d0 = b.src_dir;
        xg = macos.design.Bench.perp(d0);
        yg = cross(d0, xg);
        F = @(v) sprintf('%.15G  %.15G  %.15G', v(1), v(2), v(3));
        ln = {};
        ln{end+1} = sprintf('%% %s -- generated by macos.design.Bench', b.name);
        ln{end+1} = sprintf('        ChfRayDir=  %s', F(d0));
        ln{end+1} = sprintf('        ChfRayPos=  %s', F(b.src_pos));
        ln{end+1} = sprintf('          zSource=  %.10G', b.zsource);
        ln{end+1} = sprintf('        BaseUnits=  %s', b.baseunits);
        ln{end+1} = sprintf('        WaveUnits=  %s', b.baseunits);
        ln{end+1} =         '           IndRef=  1.0D+00';
        ln{end+1} =         '           Extinc=  0.0D+00';
        ln{end+1} = sprintf('          Wavelen=  %.9E', b.wavelen);
        ln{end+1} =         '             Flux=  1.0D+00';
        ln{end+1} = sprintf('         Aperture=  %.9E', b.aperture);
        ln{end+1} =         '         Obscratn=  0.0D+00';
        ln{end+1} =         '         GridType=  Circular';
        ln{end+1} = sprintf('         nGridpts=  %d', b.ngridpts);
        ln{end+1} = sprintf('            xGrid=  %s', F(xg));
        ln{end+1} = sprintf('            yGrid=  %s', F(yg));
        ln{end+1} = sprintf('             nElt=  %d', numel(b.E));
        for k = 1:numel(b.E)
            e = b.E(k);
            ln{end+1} = '';                                          %#ok<*AGROW>
            ln{end+1} = sprintf('             iElt=  %d', k);
            ln{end+1} = sprintf('          EltName=  %s', e.name);
            ln{end+1} = sprintf('          Element=  %s', e.element);
            ln{end+1} = sprintf('          Surface=  %s', e.surface);
            ln{end+1} = sprintf('            KrElt=  %.10E', e.Kr);
            ln{end+1} = sprintf('            KcElt=  %.10E', e.Kc);
            ln{end+1} = sprintf('           psiElt=  %s', F(e.psi));
            ln{end+1} = sprintf('           VptElt=  %s', F(e.vpt));
            ln{end+1} = sprintf('           RptElt=  %s', F(e.rpt));
            % pol-element keywords (ChkDf2 REQUIRES PolAxis= on both types and
            % Retardance= on WavePlate, or the load is rejected) -- written in
            % waves at the bench Wavelen, matching the Rx_PolElt fixture order
            if strcmp(e.element, 'TrPolarizer') || strcmp(e.element, 'WavePlate')
                ln{end+1} = sprintf('          PolAxis=  %s', F(e.polaxis));
            end
            if strcmp(e.element, 'WavePlate')
                ln{end+1} = sprintf('       Retardance=  %.10E', e.retard);
            end
            ln{end+1} = sprintf('           IndRef=  %.6E', e.indref);
            ln{end+1} = sprintf('           Extinc=  %.6E', e.extinc);
            ln{end+1} =         '            nCoat=  0';
            ln{end+1} = sprintf('             xObs=  %s', F(macos.design.Bench.perp(e.psi)));
            ln{end+1} =         '             nObs=  0';
            if ~isempty(e.gridfile)
                w = macos.design.Bench.perp(e.psi);
                ln{end+1} = sprintf('         nGridMat=  %d', e.gridn);
                ln{end+1} = sprintf('         GridFile=  %s', e.gridfile);
                ln{end+1} = sprintf('        GridSrfdx=  %.10E', e.griddx);
                ln{end+1} = sprintf('            pData=  %s', F(e.vpt));
                ln{end+1} = sprintf('            xData=  %s', F(w));
                ln{end+1} = sprintf('            yData=  %s', F(cross(e.psi, w)));
                ln{end+1} = sprintf('            zData=  %s', F(e.psi));
            end
            ln{end+1} = sprintf('           ApType=  %s', e.aptype);
            if strcmp(e.aptype, 'Circular')
                ln{end+1} = sprintf('            ApVec=  %.10E  0.0D+00  0.0D+00', e.aprad);
            end
            ln{end+1} =         '         PropType=  Geometric';
            ln{end+1} = sprintf('             zElt=  %.6G', e.zelt);
            ln{end+1} =         '          nECoord=  -6';
        end
        ln{end+1} = '';
        ln{end+1} = '         nOutCord=  5';
        ln{end+1} = '             Tout=  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00';
        ln{end+1} = '                    0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00';
        ln{end+1} = '                    0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00';
        ln{end+1} = '                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00';
        ln{end+1} = '                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00';
        fid = fopen(file, 'w');  assert(fid > 0, 'Bench.emit: cannot write %s', file);
        fprintf(fid, '%s\n', ln{:});  fclose(fid);
    end

    % -----------------------------------------------------------------
    function f = sketch(b, opts)
        %SKETCH  Annotated XY-plane layout schematic of the bench.
        %   f = b.sketch('labels', L, 'title', T) draws the chief-ray
        %   polyline through every element, a footprint bar for each
        %   optic (sized by its aperture when set), the element names,
        %   and a label on every leg.  L is a cell array, one entry per
        %   element: the label for the leg ENDING at that element
        %   (empty '' = just the length).  Use it to name legs after
        %   the input parameters of your build script, so the figure
        %   teaches what the add_* DIST arguments mean.
        arguments
            b
            opts.labels (1,:) cell = {}
            opts.title  (1,:) char = ''
            opts.seg_len (1,1) double {mustBePositive} = 24
        end
        n = numel(b.E);  assert(n > 0, 'Bench.sketch: no elements.');
        pts = [b.src_pos, cat(2, b.E.rpt)];
        f = figure('Color','w');
        hold on; axis equal; grid on;
        % greedy label de-clash: keep every label at least rmin from the
        % ones already placed by pushing it further along its own offset
        % direction (clustered optics -- e.g. the two BS passes -- would
        % otherwise print on top of each other)
        ext  = max(max(pts(1:2,:),[],2) - min(pts(1:2,:),[],2));
        rmin = 0.022*ext;
        placed = zeros(2,0);
        % collision pushes go VERTICALLY (labels are horizontal text, so
        % only vertical stacking separates them regardless of width)
        dstk = [0; -rmin];
        plot(pts(1,:), pts(2,:), '-', 'Color',[0.85 0.33 0.1], 'LineWidth',1.2);
        plot(pts(1,1), pts(2,1), 'p', 'MarkerSize',13, ...
            'MarkerFaceColor',[0.85 0.33 0.1], 'MarkerEdgeColor','k');
        text(pts(1,1), pts(2,1) - 0.5*rmin, ...
            sprintf('Source: Aperture=%.4g rad (full cone), nGrid=%d', ...
                    b.aperture, b.ngridpts), ...
            'FontSize',8, 'Interpreter','none');
        placed(:,end+1) = [pts(1,1); pts(2,1) - 0.5*rmin];
        prev_s = 0;
        for k = 1:n
            e = b.E(k);
            p0 = pts(:,k);  p1 = pts(:,k+1);
            % optic footprint: bar perpendicular to psi, sized by aperture
            w = macos.design.Bench.perp(e.psi);
            hl = opts.seg_len/2;  if e.aprad > 0, hl = e.aprad; end
            plot([e.rpt(1)-hl*w(1), e.rpt(1)+hl*w(1)], ...
                 [e.rpt(2)-hl*w(2), e.rpt(2)+hl*w(2)], 'k-', 'LineWidth',2);
            nm = sprintf('%d:%s', k, e.name);
            if e.aprad > 0, nm = sprintf('%s (R=%.4g)', nm, e.aprad); end
            tx = e.rpt(1:2) + 7*e.psi(1:2);
            while ~isempty(placed) && any(vecnorm(placed - tx) < rmin)
                tx = tx + dstk;
            end
            placed(:,end+1) = tx;                                %#ok<AGROW>
            if norm(tx - e.rpt(1:2)) > 12
                plot([e.rpt(1) tx(1)], [e.rpt(2) tx(2)], ':', ...
                     'Color', [0.65 0.65 0.65]);
            end
            text(tx(1), tx(2), nm, ...
                'FontSize',8, 'FontWeight','bold', 'Interpreter','none');
            % leg annotation: parameter name (if given) + length
            leg = e.s - prev_s;  prev_s = e.s;
            useg = p1 - p0;
            if leg > 1e-6 && norm(useg(1:2)) > 1e-9
                lbl = sprintf('%.4g', leg);
                if k <= numel(opts.labels) && ~isempty(opts.labels{k})
                    lbl = sprintf('%s = %.4g', opts.labels{k}, leg);
                end
                u = useg/norm(useg);
                vperp = macos.design.Bench.perp(u);
                mid = (p0(1:2) + p1(1:2))/2 + 6*vperp(1:2);
                while ~isempty(placed) && any(vecnorm(placed - mid) < rmin)
                    mid = mid + dstk;
                end
                placed(:,end+1) = mid;                           %#ok<AGROW>
                text(mid(1), mid(2), lbl, 'FontSize',7.5, ...
                    'Color',[0.1 0.3 0.7], 'Interpreter','none', ...
                    'HorizontalAlignment','center');
            end
        end
        xlabel('X (mm)');  ylabel('Y (mm)');
        if isempty(opts.title)
            opts.title = sprintf('%s -- bench layout (XY plane)', b.name);
        end
        title(opts.title, 'Interpreter','none');
    end

    % -----------------------------------------------------------------
    function print_chain(b)
        %PRINT_CHAIN  One line per element: index, name, kind, chief crossing.
        fprintf('%s: %d elements, chief path %.3f mm\n', b.name, numel(b.E), b.path_len);
        for k = 1:numel(b.E)
            e = b.E(k);
            fprintf('  %2d  %-12s %-11s s=%9.3f  vpt=[%9.3f %9.3f %9.3f]\n', ...
                k, e.name, e.element, e.s, e.vpt(1), e.vpt(2), e.vpt(3));
        end
    end
end

% =====================================================================
methods (Access = private)
    function P = step(b, dist)
        b.pos = b.pos + dist*b.dir;
        b.path_len = b.path_len + dist;
        P = b.pos;
    end

    function e = blank(b, name, element)
        e = struct('name', name, 'element', element, 'surface', 'Flat', ...
            'Kr', -1e22, 'Kc', 0.0, 'psi', b.dir, 'vpt', b.pos, ...
            'rpt', [NaN;NaN;NaN], ...   % NaN = same as vpt (resolved in push)
            'indref', 1.0, 'extinc', 0.0, 'aptype', 'None', 'aprad', 0, ...
            'gridfile', '', 'gridn', 0, 'griddx', 0, ...
            'polaxis', [1;0;0], 'retard', 0.0, ...   % pol-element fields (TrPolarizer/WavePlate)
            'zelt', 1e22, 's', b.path_len);
    end

    function i = push(b, e)
        e.s = b.path_len;
        if any(isnan(e.rpt)), e.rpt = e.vpt; end
        if isempty(b.E), b.E = e; else, b.E(end+1) = e; end
        i = numel(b.E);
    end
end

% =====================================================================
methods (Static)
    function u = unit(v)
        n = norm(v);  assert(n > 0, 'Bench: zero vector.');
        u = v(:)/n;
    end

    function x = perp(d)
        %PERP  A unit vector perpendicular to D (z-cross convention).
        d = macos.design.Bench.unit(d);
        x = cross([0;0;1], d);
        if norm(x) < 1e-9, x = cross([0;1;0], d); end
        x = macos.design.Bench.unit(x);
    end

    function r = reflect(d, nh)
        %REFLECT  Mirror reflection of direction D about unit normal NH.
        d = macos.design.Bench.unit(d);  nh = macos.design.Bench.unit(nh);
        r = d - 2*dot(d, nh)*nh;
    end

    function [P, s] = sphere_hit(p0, d, C, rho)
        %SPHERE_HIT  Nearest forward intersection of ray (P0,D) with the
        %   sphere of radius RHO about C.
        d = macos.design.Bench.unit(d);
        oc = p0 - C;
        bq = dot(d, oc);
        cq = dot(oc, oc) - rho^2;
        disc = bq^2 - cq;
        assert(disc > 0, 'Bench.sphere_hit: ray misses the sphere.');
        roots = [-bq - sqrt(disc), -bq + sqrt(disc)];
        roots = roots(roots > 1e-9);
        assert(~isempty(roots), 'Bench.sphere_hit: sphere is behind the ray.');
        s = min(roots);
        P = p0 + s*d;
    end

    function t = refract(d, nh, n1, n2)
        %REFRACT  Vector Snell refraction of unit direction D at a flat
        %   interface with unit normal NH, from index N1 into N2.
        d = macos.design.Bench.unit(d);  nh = macos.design.Bench.unit(nh);
        cosi = -dot(d, nh);
        if cosi < 0, nh = -nh;  cosi = -cosi; end   % orient normal against d
        eta = n1/n2;
        k = 1 - eta^2*(1 - cosi^2);
        assert(k > 0, 'Bench.refract: total internal reflection.');
        t = eta*d + (eta*cosi - sqrt(k))*nh;
        t = macos.design.Bench.unit(t);
    end

    function z = conic_sag(Rmag, Kc, r)
        %CONIC_SAG  Exact conic sag; parabolic fallback past the real extent.
        c = 1/Rmag;
        rad = 1 - (1+Kc)*c^2*r^2;
        if rad <= 0, z = 0.5*c*r^2; else, z = c*r^2/(1 + sqrt(rad)); end
    end
end
end
