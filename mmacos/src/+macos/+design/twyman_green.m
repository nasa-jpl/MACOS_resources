function G = twyman_green(opts)
%MACOS.DESIGN.TWYMAN_GREEN  Build a compensated Twyman-Green IFO rig.
%   G = macos.design.twyman_green(...) builds BOTH arms of the generic
%   Twyman-Green interferometer (the templates/40_benches/bench_ifo layout)
%   with the Bench add-optic utilities and returns them ready to emit:
%
%     TEST ARM  source -> baffle -> L1 -> BS reflect (45 deg,
%               front-coated plate) -> compensator (double-passed) ->
%               TEST OPTIC (retro) -> BS transmit -> Recomb -> L2 ->
%               focal mask -> detector at the test-optic pupil image.
%     REF ARM   same front end -> BS transmit -> PZT (retro) -> BS
%               internal-reflect return -> same Recomb plane -> same
%               output train to the same detector plane.
%
%   Glass paths balance exactly (identical plate + compensator, real
%   internal V on the reference return); the flat-test-optic null is
%   ~2.5e-9 rad.  See example_bench_ifo.m for the fully-annotated
%   walk-through and the PSI processing that runs on this rig.
%
%   The TEST OPTIC can carry the "unknown":
%     'to_Kr'         weak-sphere figure (Surface=Conic vertex radius)
%     'to_grid_file'/'to_grid_n'/'to_grid_dx'
%                     a GridData figure map (e.g. a DM surface built
%                     with PROPER) in the optic's own local frame.
%
%   TWO SPLITTER FLAVOURS ('pbs'):
%     'plate' (default)  the layout above -- a front-coated perfect-conductor
%                        plate plus a compensator.  With 'polarizing' the
%                        polarization split is carried CONCEPTUALLY, by an
%                        ideal TrPolarizer in each arm.
%     'cube'             a CEMENTED MacNeille polarizing cube: one coated
%                        interface at 45 deg between two prisms of the same
%                        glass.  The split is real coating physics -- the test
%                        arm transmits out and reflects back, the reference arm
%                        reflects out and transmits back, each arm's
%                        double-passed QWP swapping its state between the
%                        coating's own s and p eigenaxes so both leave by the
%                        same output port.  Faces are normal to their beams, so
%                        the cube needs NO compensator: every traversal is
%                        a/2 -> diagonal -> a/2 whichever port you enter by, and
%                        the two arms' glass paths are identical by
%                        construction.  Requires 'polarizing',true.  See
%                        macos.design.pbs_macneille for the stack and
%                        templates/90_polarization/tg_psi_dm_v2 for the rig.
%
%   All lengths mm.  Returns struct G:
%     .bt .br     test / reference Bench objects (call .emit yourself)
%     .T  .R      element-index structs (.iTO/.iRC/.iMASK/.iDET; .iPZT;
%                 cube: .iPBSf/.iPBSr = [face diagonal face] per traversal)
%     .bs         the shared BS plate token (cube: the cube token)
%     .pbs        cube only: the macos.design.pbs_macneille design struct
%     .det_leg    detector leg length (shared plane)
%     .P          the resolved parameter struct
%
%   See also: macos.design.Bench, templates/40_benches/bench_ifo.

arguments
    opts.F1 (1,1) double = 500
    opts.F2 (1,1) double = 250
    % ---- beam-splitter angle of incidence (slice-2 AOI/clearance trade) --
    % BS_AOI is the chief AOI on the beam-splitter, degrees.  The fold turn
    % angle is 180 - 2*BS_AOI, so BS_AOI=45 is the canonical right-angle
    % fold.  Every downstream face (compensator, BS substrate transits, the
    % internal-reflect return) tracks the BS normal automatically through
    % the shared bs token and the recomputed chief -- only the reflect
    % turn direction is set here.  DEFAULT 45 emits BIT-IDENTICALLY to the
    % pre-slice-2 rig (the exact [0;-1;0] literal, no cosd(90) round-off).
    opts.BS_AOI (1,1) double {mustBePositive} = 45
    opts.D_LENS (1,1) double = 60
    opts.N_GLASS (1,1) double = 1.5
    opts.R_BAFFLE (1,1) double = 12.5
    opts.D_SB (1,1) double = 250
    opts.FILL (1,1) double = 0.95
    opts.BS_T (1,1) double = 1.5
    opts.PLATE_SUB (1,:) double = []   % [n t]: SUBSTRATE for every thin
                                       %  polarizing element -- the input
                                       %  polarizer, both arm quarter-wave
                                       %  plates, the output plate and the
                                       %  analyzer.  Two refracting faces of
                                       %  index n, t thick, around the ideal
                                       %  element, which keeps ITS OWN
                                       %  station; the chief comes out t/2
                                       %  further along per plate, so stations
                                       %  downstream shift by t/2 per upstream
                                       %  plate (measured: FocalMask +5.0 mm
                                       %  for five 2 mm plates).  That is the
                                       %  glass being real, and the tail
                                       %  retune absorbs it.  [] = the ideal
                                       %  zero-thickness element (the record).
    opts.EDGE_MARGIN (1,1) double {mustBeNonnegative} = 2.0
    opts.MASK_SUB (1,:) double = []    % [n t]: the MASK's own plate -- the
                                       %  etched dimple, the pinhole plate or
                                       %  the metasurface is a pattern on a
                                       %  slab, and that slab sits in the
                                       %  CONVERGING beam.  The two faces go
                                       %  BEFORE the sandwich's entrance
                                       %  sphere, with the mask plane in air
                                       %  behind them, and they are inserted
                                       %  INSIDE the existing L2->sphere gap,
                                       %  so the mask does not move.  What
                                       %  does move is the focus: t*(1-1/n)
                                       %  of shift, which the tail retune
                                       %  absorbs, plus the spherical
                                       %  aberration t*(n^2-1)*NA^4/(8n^3)
                                       %  that is the reason to model it at
                                       %  all.  [] = no plate (the record).
                                       %  every singlet's edge thickness (mm):
                                       %  add_lens makes the centre thickness
                                       %  sag + this.  2.0 is the record; a
                                       %  103 mm singlet wants 3-5.
    opts.D_L1_BS (1,1) double = 150
    opts.D_BS_TO (1,1) double = 250
    opts.D_BS_CMP (1,1) double = 100
    opts.D_RECOMB (1,1) double = 5
    opts.D_RC_L2 (1,1) double = NaN   % NaN = 200 (plate); the cube subtracts
                                      %  its half-side so the Recomb->L2->mask
                                      %  ->detector conjugate is the SAME
                                      %  geometry as the plate rig and the
                                      %  l2_trade tail trims transfer verbatim
    opts.R_TO_AP (1,1) double = 30
    % THESE DEFAULTS ARE THE MISFED BENCH'S (2026-09-17).  L1_Kr 236.866 is
    % (n-1)*473.7 = (n-1)*(F1 - zsource): l2_trade matched the radius to the
    % conjugate the source really sat at, so the default lens collimates only
    % when SRC_AT_FOCUS is FALSE.  With SRC_AT_FOCUS the caller must pass the
    % re-solved pair -- L1_Kr 249.246312, L1_Kc -0.583016, L2_Kc -0.581843 for
    % the 96 mm rig's scale, or run tg96_collimate for its own.  Left as they
    % are so every bench built on the defaults (psri_bench, the 90_polarization
    % rigs, bench_ifo_dm) keeps its record.
    opts.L1_Kr (1,1) double = 236.866
    opts.L1_Kc (1,1) double = -0.5829
    opts.L2_Kr (1,1) double = -124.076
    opts.L2_Kc (1,1) double = -0.5826
    % ---- powered-optic type: 'lens' (default; the record) | 'oap' ---------
    % 'oap' replaces the two lenses L1 (collimator) and L2 (focuser) with
    % off-axis parabola sections (macos.design.Bench.add_oap), all-reflective.
    % The folds are kept ENTIRELY in the source->OAP1 and OAP2->detector legs
    % and OAP1 re-emits along the lens rig's post-L1 direction (+x), so the BS,
    % both arms and the recomb plane are geometrically UNCHANGED -- the Stage-A
    % clearance, the sampling budget and the tail conjugate bookkeeping all
    % carry over.  Both OAPs fold IN THE BS PLANE (x-y).  OAP conics are set by
    % add_oap (Kr=-2*f_par, Kc=-1); the L1_Kr/L1_Kc/L2_Kr/L2_Kc lens seeds are
    % ignored in 'oap' mode.  'lens' emits BIT-IDENTICALLY to the pre-oap rig.
    % emit an ExitPupil Reference just before the detector (for
    % macos.pupil_quality / fex, which need a Return/Reference at nElt-1).
    % OFF by default -> byte-identical to the pre-oap rig.
    opts.ep_ref (1,1) logical = false
    opts.optics (1,:) char {mustBeMember(opts.optics,{'lens','oap'})} = 'lens'
    % see POL_IN below: an OAP collimator's conjugate leg comes back along
    % the collimated axis, so the input polarizer's station is not free.
    opts.OAP1_AOI (1,1) double {mustBePositive} = 15   % collimator fold AOI, deg
    opts.OAP2_AOI (1,1) double {mustBePositive} = 15   % focuser  fold AOI, deg
    opts.OAP1_SIDE (1,1) double {mustBeMember(opts.OAP1_SIDE,[-1 1])} = 1
    opts.OAP2_SIDE (1,1) double {mustBeMember(opts.OAP2_SIDE,[-1 1])} = 1
    opts.ngridpts (1,1) double = 63
    opts.to_Kr (1,1) double = 0
    opts.to_grid_file (1,:) char = ''
    opts.to_grid_n (1,1) double = 0
    opts.to_grid_dx (1,1) double = 0
    % ---- detector-leg (tail) architecture (l2_trade work; default is the
    % original singlet, bit-identical when these are omitted) -------------
    opts.tail_arch (1,:) char {mustBeMember(opts.tail_arch, ...
        {'singlet','fieldlens','doublet'})} = 'singlet'
    opts.mask_prop (1,:) char {mustBeMember(opts.mask_prop, ...
        {'geometric','nf','nf_legacy'})} = 'geometric'
                                         % 'nf': bracket the FocalMask with
                                         %  reference SPHERES carrying NF1/
                                         %  NF2 legs (the ctb_dcr.in FPM
                                         %  idiom) so the wavefront lands on
                                         %  a focal-scale grid there -- for
                                         %  lambda/D-class focal masks (the
                                         %  ZWFS dimple).  The exit sphere
                                         %  carries the ENTRANCE sphere's
                                         %  zElt/Kr (SYMMETRIC sandwich, as
                                         %  ctb_dcr.in) so the round trip is
                                         %  the exact identity and the tail
                                         %  sees the DM-CONJUGATE pupil.
                                         % 'nf_legacy': the 2026-09-04..08
                                         %  emission (exit sphere zElt/Kr =
                                         %  0.6*D_MASK_FL): ASYMMETRIC, so
                                         %  the engine's sphere-to-plane leg
                                         %  applies a focal quadratic phase
                                         %  ~ (Z2-Z1)*Z1/Z2 = a Fresnel
                                         %  DEFOCUS of the reimaged pupil
                                         %  (z_eff 4.86 m on the zwfs_dm96
                                         %  rig: 16% flat-pupil change, 29%
                                         %  rms amplitude modulation under a
                                         %  30 nm DM state, ringed poke
                                         %  kernel).  Kept ONLY to reproduce
                                         %  the S1-S6 zwfs_dm96 record.
                                         % 'geometric' (default) =
                                         %  bit-identical legacy emission.
                                         %  fieldlens arch only.
    opts.DET_TRIM (1,1) double = 0       % additive trim on det_leg (all arches)
    opts.MASK_TRIM (1,1) double = 0      % additive trim on the FocalMask
                                         %  position (thin-lens seed -> true
                                         %  focus; det_leg re-chains from the
                                         %  moved mask, pupil conjugate kept)
    % 'fieldlens': pupil-relay field lens just behind the FocalMask
    opts.FL_F (1,1) double = 150         % field-lens focal length
    opts.FL_Kc (1,1) double = NaN        % powered-face conic (NaN = seed -n^2)
    opts.FL_D (1,1) double = 12          % field-lens diameter (beam at the
                                         %  focus is sub-mm; a small FL keeps
                                         %  the sag/thickness physical at
                                         %  short focal lengths)
    opts.D_MASK_FL (1,1) double = 5      % mask -> field lens distance
    % 'doublet': L2 split into two air-spaced plano singlets
    opts.L2A_F (1,1) double = 500        % front element focal length
    opts.L2B_F (1,1) double = 500        % back element focal length
    opts.L2_SEP (1,1) double = 25        % powered-surface separation
    opts.L2A_Kc (1,1) double = NaN       % conics (NaN = add_lens seed -n^2)
    opts.L2B_Kc (1,1) double = NaN
    % ---- polarizing phase-shifting variant (slice 3) --------------------
    % When 'polarizing' is true the builder inserts real TrPolarizer /
    % WavePlate elements to make a ROTATING-ANALYZER polarization PSI:
    %   input polarizer (both arms) -> [BS] -> a double-passed QWP in each
    %   arm (net half-wave, rotating that arm's linear state) -> [recomb] ->
    %   output QWP -> rotating analyzer -> detector.
    % The two arms leave orthogonally polarized; the output QWP maps them to
    % orthogonal circular, so an analyzer at angle t imposes a fringe phase
    % 2t -- stepping t = 0/45/90/135 deg is a four-step PSI with NO moving
    % PZT.  Every pol element sits in a COLLIMATED, NORMAL-INCIDENCE leg
    % (psi = chief), where the material-axis convention is identically
    % absent.  Axes are given as ANGLES (deg) in each leg's LOCAL transverse
    % plane (perp(dir), cross(dir,perp)) so "45 deg" is fold-correct.  The
    % analyzer axis is a default; the harness steps it at runtime.  DEFAULT
    % false emits BIT-IDENTICALLY to the non-polarizing rig (all insertions
    % gated, and each steals its standoff from the following leg so the BS,
    % test-optic, PZT and pupil conjugates are unmoved).
    opts.polarizing (1,1) logical = false
    opts.pol_in_deg   (1,1) double = 45     % input polarizer, both arms
    opts.qwp_test_deg (1,1) double = NaN    % test-arm QWP fast axis
    opts.qwp_ref_deg  (1,1) double = NaN    % ref-arm QWP fast axis
    opts.out_qwp_deg  (1,1) double = NaN    % output QWP fast axis (shared leg)
    opts.analyzer_deg (1,1) double = 0      % analyzer default (stepped at run)
    opts.qwp_ret      (1,1) double = 0.25   % nominal QWP retardance (waves)
    opts.D_QWP        (1,1) double = 25     % arm-QWP standoff from the retro
    opts.D_POL        (1,1) double = 10     % input-polarizer / output-leg standoff
    % ---- where the input polarizer lives (reflective rigs) --------------
    % 'collimated' (default, the record): D_POL past the collimator, in the
    %   collimated pre-splitter leg.  Fine for a LENS collimator, whose
    %   conjugate leg is on-axis, so nothing travels the other way there.
    % 'source': in the DIVERGING source leg, D_POL past the baffle.  An OAP
    %   collimator's conjugate leg comes BACK along the collimated axis, and
    %   at 10 mm past the pole the two are 10*tan(2*AOI) apart -- 1.7 mm at
    %   5 deg -- so a polarizer there sits inside the incoming cone at ANY
    %   fold angle (measured: -102 mm of clearance at 5 deg, still -80 mm at
    %   30 deg, oap_fold_solve/fold1).  Polarizing the SOURCE is what a real
    %   reflective bench does anyway.  Costs: the leg is f/8.3, so the ray
    %   normal-incidence assumption in add_polarizer's help is broken at the
    %   sin^2(3.4 deg) = 0.4 % level, and OAP1's own diattenuation now acts
    %   on an already-polarized beam.  'oap' optics only.
    opts.POL_IN (1,:) char {mustBeMember(opts.POL_IN,{'collimated','source'})} = 'collimated'
    % ---- feed the collimator at its TRUE focus (reflective rigs) --------
    % Bench emits zSource and the engine puts the real point source at
    % ChfRayPos + zSource*ChfRayDir (sourcsub.F:38), so the source sits
    % 'zsource' mm DOWNSTREAM of the point front_end computes -- while
    % add_oap builds the parabola for a focus AT that point.  The collimator
    % is therefore fed 25 mm inside its focus, and the collimated beam
    % carries a 926 urad rms residual (a 28.8 m focus).
    % A LENS rig hides that in its TUNED figures (L1_Kr / L1_Kc from
    % l2_trade); a parabola has no such freedom, so the same error surfaces
    % as the reflective rig's "fold coma" -- measured 0.13 / 0.24 / 0.37
    % lambda F/D of best-focus blur at 1 / 3 / 5 deg, and EXACTLY ZERO at
    % every angle once the source is moved back (oap_conj_probe, runs/conj).
    % SRC_AT_FOCUS true adds zsource to the source distance so the effective
    % point source lands on the parabola's focus.  Default false = the
    % record, so no existing number moves silently.
    %
    % EXTENDED TO THE LENS RIG 2026-09-17 (BRIEF_to_tg_redo package A).  The
    % lens rig has the SAME error -- its collimator is fed zsource inside its
    % conjugate -- and its TUNED L1 conic absorbed it, which is why it never
    % surfaced as blur: measured on the deck of record, the "collimated"
    % space carries 5.8e-4 rad rms of angular spread = 41 waves of curvature
    % over the beam, and the rays walk off the grid between the
    % physical-optics chain's near-field legs (tg96_pupil_s2s).  With the
    % source at the conjugate and the powered face's conic re-solved there
    % (tg96_collimate) the spread is 1e-5-class.  The bench ORIGIN moves back
    % rather than the lens forward, so L1's powered face and every station
    % downstream of it stay exactly where the record put them.
    opts.SRC_AT_FOCUS (1,1) logical = false
    opts.zsource      (1,1) double  = 25   % the Rx zSource both arms emit
    % Additive trim on the source -> collimator conjugate (mm), applied with
    % SRC_AT_FOCUS.  A plano singlet's conjugate is not F1 from its powered
    % vertex (the rear principal plane sits t/n inside the glass) and the
    % exact surface for these plano orientations is a Cartesian oval, not a
    % conic, so the residual is a SOLVE: tg96_collimate minimizes the exit
    % rays' angular spread over (SRC_TRIM, L1_Kc) and the sheet carries the
    % winner.  A parabola is exact, so the oap rig's answer is 0.
    opts.SRC_TRIM     (1,1) double  = 0
    % ---- v2: a REAL polarizing beamsplitter (cemented MacNeille cube) ----
    % 'pbs','plate' (default) is the v1 rig: the splitter is a front-coated
    % PERFECT-CONDUCTOR plate plus a compensator, and the polarization split
    % is carried CONCEPTUALLY by an ideal TrPolarizer in each arm.
    %
    % 'pbs','cube' replaces that concept with the COMPONENT: one cemented
    % coated interface at 45 deg INSIDE the glass, and the arms are routed by
    % real coating physics.  Test arm = TRANSMIT out / REFLECT back; reference
    % arm = REFLECT out / TRANSMIT back.  Each arm's double-passed QWP is a
    % net half-wave that swaps its state between the coating's own s and p
    % eigenaxes, so both arms leave by the SAME output port -- the physical
    % "all light to the output port" routing, still one sequential deck per
    % arm because the engine does not split rays.  Requires 'polarizing'.
    %
    % Faces are normal to their beams (no deviation, no walk-off, no face
    % diattenuation) and every traversal is a/2 -> diagonal -> a/2, so the two
    % arms balance their glass EXACTLY with no compensator plate.
    opts.pbs (1,:) char {mustBeMember(opts.pbs, {'plate','cube'})} = 'plate'
    opts.CUBE_SIDE (1,1) double {mustBePositive} = 60   % cube edge, mm
    opts.CUBE_N    (1,1) double = NaN    % prism index; NaN = the MacNeille
                                         %  index of the coating pair
    opts.pbs_coat  (:,3) double = NaN(1,3)  % diagonal stack [n k thk_waves];
                                         %  NaN = macos.design.pbs_macneille.
                                         %  Pass zeros(0,3) for an explicitly
                                         %  BARE cemented interface -- which,
                                         %  with the same glass either side,
                                         %  is optically nothing and reflects
                                         %  no light at all (the tTgPol2
                                         %  structural gate).  An empty matrix
                                         %  cannot mean "default" AND "none".
    opts.pbs_nperiod (1,1) double {mustBeInteger, mustBePositive} = 4
    opts.ar_faces  (1,1) logical = true   % single-layer MgF2 AR on the faces
    opts.AR_N      (1,1) double {mustBePositive} = 1.38   % MgF2
end
P = opts;

% ---- azimuth defaults, resolved per PBS flavour ----------------------
% The plate rig leaves the arms at -45/+45 (a half-wave at azimuth a maps a
% 45-deg input to 2a-45), so its arm plates sit at 0 and 45 and the output
% QWP at 0.  The CUBE rig leaves each arm on a coating EIGENAXIS -- the test
% arm on p (local 0), the reference arm on s (local 90) -- so every plate
% wants to be at 45 deg to its own arm state: 45/45/45.  NaN resolves to the
% flavour's design value; the plate defaults are unchanged, so a plate build
% still emits bit-identically.
cube = strcmp(P.pbs, 'cube');
if isnan(P.qwp_test_deg), P.qwp_test_deg = 45*cube;      end
if isnan(P.qwp_ref_deg),  P.qwp_ref_deg  = 45;           end
if isnan(P.out_qwp_deg),  P.out_qwp_deg  = 45*cube;      end
% The cube's exit face sits a half-side beyond its centre, where the plate's
% coating sat, so the whole output leg would otherwise ride 30 mm further
% from the test optic and the l2_trade tail trims would no longer apply.
if isnan(P.D_RC_L2),      P.D_RC_L2 = 200 - cube*P.CUBE_SIDE/2; end

% ---- BS fold direction from the AOI ---------------------------------
% turn = 180 - 2*AOI about +z, applied to the +x chief toward -y.  Pin the
% 45-deg case to the exact literal so the default rig stays bit-identical
% (cosd(90) is 6.1e-17, which would perturb every emitted coordinate).
if abs(P.BS_AOI - 45) < 1e-12
    bs_out = [0; -1; 0];
else
    turn = 180 - 2*P.BS_AOI;
    bs_out = [cosd(turn); -sind(turn); 0];
end

% =====================================================================
%  v2: cemented MacNeille cube.  Built first because BOTH arms share one
%  cube token (absolute geometry), which is what makes their glass paths
%  identical to the last bit rather than merely equal by arithmetic.
% =====================================================================
if cube
    assert(P.polarizing, ...
        ['twyman_green: ''pbs'',''cube'' IS the polarization split -- it ' ...
         'needs ''polarizing'',true (the arm waveplates route the light).']);
    PBS = macos.design.pbs_macneille('nperiod', P.pbs_nperiod, ...
                                     'lambda', 6.328e-4, 'aoi', 45);
    if ~isnan(P.CUBE_N)
        % Deliberate detune: a real catalogue glass instead of the design
        % index.  Brewster is then violated at the H/L interfaces, r_p stops
        % being zero, and the cube starts to rotate the arm states -- the v2
        % tolerance knob.  Re-solve the stack at the requested index so the
        % quarter-wave-AT-ANGLE thicknesses stay self-consistent.
        PBS = macos.design.pbs_macneille('nperiod', P.pbs_nperiod, ...
                    'lambda', 6.328e-4, 'aoi', 45, 'n_glass', P.CUBE_N);
    end
    coat_d = P.pbs_coat;
    if ~isempty(coat_d) && all(isnan(coat_d(:))), coat_d = PBS.layers; end
    % Single-layer MgF2 quarter-wave AR on the four faces (Macleod ch. 3;
    % n = 1.38 is the standard published visible value).  Normal incidence,
    % so the quarter wave is 0.25 exactly with no angle factor.  Bare glass
    % at n = 1.6554 loses 6.2% a face; this takes it to 0.49%, and the four
    % face crossings per arm are what set the output-port efficiency.
    if P.ar_faces, coat_ar = [P.AR_N, 0, 0.25]; else, coat_ar = zeros(0,3); end
    n_prism = PBS.n_glass;
end

% ---- test arm -------------------------------------------------------
if cube
    bt = front_end(P, 'ifo_test');
    bt.add_polarizer(P.D_POL, ax_local(bt.dir, P.pol_in_deg), 'name','PolIn', 'substrate',P.PLATE_SUB);
    cubetok = bt.pbs_cube(P.D_L1_BS - P.D_POL, bs_out, 'side',P.CUBE_SIDE, ...
        'n',n_prism, 'coat',coat_d, 'ar',coat_ar, 'name','PBS');
    T.iPBSf = bt.add_pbs_pass(cubetok, 'mode','transmit', 'tag','f');
    leg_to  = P.D_BS_TO - P.CUBE_SIDE/2 - P.D_QWP;
    assert(leg_to > 0, 'twyman_green: cube too large for D_BS_TO.');
    qa_t = ax_local(bt.dir, P.qwp_test_deg);
    bt.add_waveplate(leg_to, qa_t, P.qwp_ret, 'name','QWPtestIn', 'substrate',P.PLATE_SUB);   % one plate, D_QWP before the retro (both passes)
    T.iTO = bt.add_mirror(P.D_QWP, 'name','TestOptic', ...
        'aprad',P.R_TO_AP, 'Kr',P.to_Kr, 'grid_file',P.to_grid_file, ...
        'grid_n',P.to_grid_n, 'grid_dx',P.to_grid_dx);
    bt.add_waveplate(P.D_QWP, qa_t, P.qwp_ret, 'name','QWPtestOut', 'substrate',P.PLATE_SUB);
    T.iPBSr = bt.add_pbs_pass(cubetok, 'mode','reflect', 'tag','r');
    T.iRC = bt.add_reference(P.D_RECOMB, 'Recomb');
    [T, det_leg] = tail(bt, P, T, T.iTO, []);

    % ---- reference arm ----------------------------------------------
    br = front_end(P, 'ifo_ref');
    br.add_polarizer(P.D_POL, ax_local(br.dir, P.pol_in_deg), 'name','PolIn', 'substrate',P.PLATE_SUB);
    R.iPBSf = br.add_pbs_pass(cubetok, 'mode','reflect', 'tag','f');
    qa_r = ax_local(br.dir, P.qwp_ref_deg);
    br.add_waveplate(leg_to, qa_r, P.qwp_ret, 'name','QWPrefIn', 'substrate',P.PLATE_SUB);    % one plate, D_QWP before the flat
    R.iPZT = br.add_mirror(P.D_QWP, 'name','PZT');
    br.add_waveplate(P.D_QWP, qa_r, P.qwp_ret, 'name','QWPrefOut', 'substrate',P.PLATE_SUB);
    R.iPBSr = br.add_pbs_pass(cubetok, 'mode','transmit', 'tag','r');
    d_rc = dot(bt.E(T.iRC).vpt - br.pos, br.dir);
    assert(d_rc > 0, 'twyman_green: recomb plane behind the reference return');
    R.iRC = br.add_reference(d_rc, 'Recomb');
    [R, ~] = tail(br, P, R, [], det_leg);

    G = struct('bt',bt, 'br',br, 'T',T, 'R',R, 'bs',cubetok, ...
               'det_leg',det_leg, 'P',P, 'pbs',PBS);
    return
end

bt = front_end(P, 'ifo_test');
% input polarizer in the collimated pre-BS leg (slice-3 variant); it steals
% its standoff from the L1->BS leg so the BS stays put (bit-identical off)
if P.polarizing && ~pol_at_source_(P)
    bt.add_polarizer(P.D_POL, ax_local(bt.dir, P.pol_in_deg), 'name','PolIn', 'substrate',P.PLATE_SUB);
    d_l1_bs = P.D_L1_BS - P.D_POL;
else
    d_l1_bs = P.D_L1_BS;   % POL_IN 'source': front_end already placed it
end
[~, bs] = bt.add_bs_reflect(d_l1_bs, bs_out, 'thickness',P.BS_T, 'n',P.N_GLASS);
cmp = bt.plate(P.D_BS_CMP, bs.psi, 'thickness',P.BS_T, 'n',P.N_GLASS, 'name','Comp');
bt.add_bs_transmit(cmp, 'tag','d');
leg_to = P.D_BS_TO - P.D_BS_CMP - P.BS_T;
if P.polarizing
    % double-passed QWP: SAME global fast axis both passes -> net half-wave,
    % rotating this arm's linear state.  ONE physical plate, D_QWP before
    % the retro: the forward pass ('In') is placed there too, so the
    % emitted deck shows the plate where it is (2026-09-15: the 'In'
    % record used to sit D_QWP after the compensator, inside the node,
    % where it read as a part in another beam); the return pass ('Out')
    % rides the geometry-absolute comp transit.
    qa_t = ax_local(bt.dir, P.qwp_test_deg);
    bt.add_waveplate(leg_to - P.D_QWP, qa_t, P.qwp_ret, 'name','QWPtestIn', 'substrate',P.PLATE_SUB);
    leg_to = P.D_QWP;
end
T.iTO = bt.add_mirror(leg_to, 'name','TestOptic', ...
    'aprad',P.R_TO_AP, 'Kr',P.to_Kr, 'grid_file',P.to_grid_file, ...
    'grid_n',P.to_grid_n, 'grid_dx',P.to_grid_dx);
if P.polarizing
    bt.add_waveplate(P.D_QWP, qa_t, P.qwp_ret, 'name','QWPtestOut', 'substrate',P.PLATE_SUB);
end
bt.add_bs_transmit(cmp, 'tag','u');
bt.add_bs_transmit(bs, 'tag','o');
T.iRC = bt.add_reference(P.D_RECOMB, 'Recomb');
[T, det_leg] = tail(bt, P, T, T.iTO, []);

% ---- reference arm --------------------------------------------------
br = front_end(P, 'ifo_ref');
if P.polarizing && ~pol_at_source_(P)
    br.add_polarizer(P.D_POL, ax_local(br.dir, P.pol_in_deg), 'name','PolIn', 'substrate',P.PLATE_SUB);
end
br.add_bs_transmit(bs, 'tag','f');
leg_pzt = P.D_BS_TO;
if P.polarizing
    qa_r = ax_local(br.dir, P.qwp_ref_deg);
    br.add_waveplate(leg_pzt - P.D_QWP, qa_r, P.qwp_ret, 'name','QWPrefIn', 'substrate',P.PLATE_SUB);   % one plate, at the flat's end
    leg_pzt = P.D_QWP;
end
R.iPZT = br.add_mirror(leg_pzt, 'name','PZT');
if P.polarizing
    br.add_waveplate(P.D_QWP, qa_r, P.qwp_ret, 'name','QWPrefOut', 'substrate',P.PLATE_SUB);
end
br.add_bs_reflect_return(bs);
d_rc = dot(bt.E(T.iRC).vpt - br.pos, br.dir);
assert(d_rc > 0, 'twyman_green: recomb plane behind the reference return');
R.iRC = br.add_reference(d_rc, 'Recomb');
[R, ~] = tail(br, P, R, [], det_leg);

G = struct('bt',bt, 'br',br, 'T',T, 'R',R, 'bs',bs, 'det_leg',det_leg, 'P',P);
end

% ---------------------------------------------------------------------
function tf = pol_at_source_(P)
%POL_AT_SOURCE_  true when the input polarizer belongs in the diverging leg.
%   Only meaningful on a reflective front end; a lens collimator has no
%   conjugate leg coming back along the collimated axis, so the option is
%   ignored (and 'lens' stays bit-identical to the record).
tf = strcmp(P.optics, 'oap') && strcmp(P.POL_IN, 'source');
end

function b = front_end(P, name)
    AP = 2*atan(P.R_BAFFLE/P.D_SB)*P.FILL;
    if strcmp(P.optics, 'oap')
        % Reflective collimator.  Keep the fold ENTIRELY in the source->OAP1
        % leg: place the source off-axis so its diverging chief, after OAP1's
        % in-BS-plane fold at OAP1_AOI, emerges collimated along +x -- the
        % exact post-L1 direction of the lens rig -- with OAP1's POLE at the
        % same point [F1;0;0] the lens L1 surface occupied.  Then the BS leg
        % begins at the identical (pos,dir) and everything downstream of the
        % collimator is geometrically unchanged.
        d_out = [1; 0; 0];
        dev   = deg2rad(180 - 2*P.OAP1_AOI);          % chief turn = 180 - 2*AOI
        a     = P.OAP1_SIDE * dev;                    % rotate d_out by -turn to
        c = cos(-a); s = sin(-a);                     %  recover the incoming dir
        d_in  = [c*d_out(1) - s*d_out(2); s*d_out(1) + c*d_out(2); 0];
        pole  = [P.F1; 0; 0];                         % == lens-rig L1 pole
        % one conjugate back -- plus zSource when SRC_AT_FOCUS, so the
        % EFFECTIVE point source (ChfRayPos + zSource*ChfRayDir) lands on the
        % parabola's focus rather than zSource mm inside it.
        d_src = P.F1;
        if P.SRC_AT_FOCUS, d_src = P.F1 + P.zsource + P.SRC_TRIM; end
        src   = pole - d_src*d_in;
        b = macos.design.Bench(name, 'aperture', AP, 'ngridpts', P.ngridpts, ...
                               'pos', src, 'dir', d_in, 'zsource', P.zsource);
        b.add_baffle(P.D_SB, P.R_BAFFLE);
        d_pole = d_src - P.D_SB;
        if P.polarizing && pol_at_source_(P)
            % The input polarizer in the DIVERGING leg, D_POL past the baffle:
            % it is then (d_pole - D_POL) before the pole, where the outgoing
            % collimated beam is that distance x tan(2*AOI1) away -- a real
            % separation, unlike the 10 mm station in collimated space.
            %
            % Its axis is REFLECTED back through OAP1 so the state arriving at
            % the splitter is the record's.  A plane mirror maps a transverse
            % vector by a = a - 2(a.n)n about its normal, and that map is an
            % involution, so the incoming axis that becomes ax_local(d_out,
            % pol_in_deg) after the fold is that same expression applied to it.
            % Without this, "45 deg" in the source leg is 45 deg about a
            % DIFFERENT local x (ax_local seeds from perp(dir)) and the fold
            % flips the in-plane component -- a real change of input state, not
            % a labelling one.  A COATED OAP1 (coat_oap) then acts on an
            % already-polarized beam: its diattenuation and retardance are a
            % genuine cost of this arrangement, not an artifact.
            a_out = ax_local(d_out, P.pol_in_deg);
            nh = d_out - d_in;  nh = nh/norm(nh);        % pole normal (bisector)
            a_in = a_out - 2*dot(a_out, nh)*nh;
            b.add_polarizer(P.D_POL, a_in, 'name','PolIn', 'substrate',P.PLATE_SUB);
            d_pole = d_pole - P.D_POL;
        end
        b.add_oap(d_pole, d_out, 'mode','collimate', ...
                  'focus_dist', P.F1, 'name','L1', 'aprad', P.D_LENS/2);
    else
        % the lens rig.  SRC_AT_FOCUS false is the record: the source sits
        % F1 from the powered face on paper while the engine puts the real
        % point source zsource mm downstream of ChfRayPos, so the collimator
        % is fed that far inside its conjugate and the tuned L1 figures
        % absorbed it.  True moves the bench ORIGIN back by zsource (+
        % SRC_TRIM), which lands the effective point source on the conjugate
        % and leaves the powered face -- and therefore every station
        % downstream of it -- at the coordinates the record used.
        d_src = P.F1;
        if P.SRC_AT_FOCUS, d_src = P.F1 + P.zsource + P.SRC_TRIM; end
        b = macos.design.Bench(name, 'aperture', AP, 'ngridpts', P.ngridpts, ...
                               'pos', [P.F1 - d_src; 0; 0], 'zsource', P.zsource);
        b.add_baffle(P.D_SB, P.R_BAFFLE);
        L1 = b.add_lens(d_src - P.D_SB, P.F1, P.D_LENS, 'mode','collimate', 'edge_margin',P.EDGE_MARGIN, ...
                        'n',P.N_GLASS, 'name','L1');
        b.E(L1.i_pow).Kr = P.L1_Kr;  b.E(L1.i_pow).Kc = P.L1_Kc;
    end
end

function L = add_focuser(b, dist, P)
%ADD_FOCUSER  L2 as a lens (default) or an OAP (P.optics=='oap').  Returns a
%   uniform descriptor L: .idx (powered-surface element index), .s (its path
%   station), .thk (glass thickness; 0 for an OAP), .F (design focal = F2), so
%   the tail conjugate math is identical for both.  The OAP folds the
%   recomb->detector leg in the BS plane at OAP2_AOI; mask/FL/detector follow.
    if strcmp(P.optics, 'oap')
        d_rc = b.dir;
        dev  = deg2rad(180 - 2*P.OAP2_AOI);
        a    = P.OAP2_SIDE * dev;  c = cos(a); s = sin(a);
        out  = [c*d_rc(1) - s*d_rc(2); s*d_rc(1) + c*d_rc(2); 0];
        O = b.add_oap(dist, out, 'mode','focus', 'focus_dist', P.F2, ...
                      'name','L2', 'aprad', P.D_LENS/2);
        L = struct('idx',O.i, 's',b.E(O.i).s, 'thk',0, 'F',P.F2);
    else
        L2 = b.add_lens(dist, P.F2, P.D_LENS, 'mode','focus', 'edge_margin',P.EDGE_MARGIN, ...
                        'n',P.N_GLASS, 'name','L2');
        b.E(L2.i_pow).Kr = P.L2_Kr;  b.E(L2.i_pow).Kc = P.L2_Kc;
        L = struct('idx',L2.i_pow, 's',b.E(L2.i_pow).s, 'thk',L2.thickness, 'F',P.F2);
    end
end

function [ix, det_leg] = tail(b, P, ix, conj_elt, det_leg)
%TAIL  Detector leg: Recomb -> (L2 architecture) -> FocalMask -> Detector at
%   the TEST-OPTIC pupil conjugate.  Shared by both arms (same det_leg), so
%   any architecture stays common-path.  The invariant every arch keeps:
%   the detector sits at the thin-lens pupil image of CONJ_ELT (plus
%   P.DET_TRIM, a knob for nulling the DM-tilt lever the thin-lens seed
%   leaves -- ~4.6 mm on the baseline singlet).
% Slice-3 output leg (both arms, before L2, collimated): output QWP then
% rotating analyzer.  They steal 2*D_POL from the Recomb->L2 leg so L2 and
% the pupil conjugate stay put.  ix.iOutQWP / ix.iAnalyzer are exposed.
d_rc_l2 = P.D_RC_L2;
assert(strcmp(P.mask_prop, 'geometric') || strcmp(P.tail_arch, 'fieldlens'), ...
    'twyman_green: mask_prop=''nf''/''nf_legacy'' is implemented for tail_arch=''fieldlens'' only.');
if P.polarizing
    ix.iOutQWP   = b.add_waveplate(P.D_POL, ax_local(b.dir, P.out_qwp_deg), ...
                                   P.qwp_ret, 'name','OutQWP', 'substrate',P.PLATE_SUB);
    ix.iAnalyzer = b.add_polarizer(P.D_POL, ax_local(b.dir, P.analyzer_deg), ...
                                   'name','Analyzer', 'substrate',P.PLATE_SUB);
    d_rc_l2 = P.D_RC_L2 - 2*P.D_POL;
end
switch P.tail_arch
case 'singlet'                     % original architecture (default)
    L2 = add_focuser(b, d_rc_l2, P);
    ix.iMASK = b.add_reference(mask_plate_(b, P.MASK_SUB, L2.F - L2.thk + P.MASK_TRIM), 'FocalMask');
    if ~isempty(conj_elt)
        s_o = L2.s - b.E(conj_elt).s;
        s_i = 1/(1/L2.F - 1/s_o);
        det_leg = s_i - (b.E(ix.iMASK).s - L2.s) + P.DET_TRIM;
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');

case 'fieldlens'                   % C1: field lens just behind the mask
    L2 = add_focuser(b, d_rc_l2, P);
    dmask = L2.F - L2.thk + P.MASK_TRIM;
    if strcmp(P.mask_prop, 'nf') || strcmp(P.mask_prop, 'nf_legacy')
        % NF1/NF2 sandwich (ctb_dcr.in FPM idiom): a reference SPHERE
        % concentric with the focus carries the sphere->plane leg onto
        % the FocalMask -- the wavefront lands there on a focal-scale
        % grid, so a lambda/D-class complex mask (ZWFS dimple) is
        % representable -- and the mask's own leg goes plane->sphere
        % onto a matching sphere behind it; geometric from there.
        d_in = 0.85 * dmask;
        d_sph = mask_plate_(b, P.MASK_SUB, dmask - d_in);
        b.add_reference(d_sph, 'MaskSphereIn', 'surface','Conic', ...
            'kr',-d_in, 'proptype','NF1', 'zelt',d_in);
        ix.iMASK = b.add_reference(d_in, 'FocalMask', ...
            'proptype','NF2', 'zelt',1e22);
        d_out = 0.6 * P.D_MASK_FL;               % the exit sphere's STATION
        if strcmp(P.mask_prop, 'nf')
            % SYMMETRIC sandwich: the exit sphere carries the ENTRANCE
            % sphere's zElt/Kr.  The engine's SPH2PL (NF1) multiplies the
            % focal field by exp(i*S*(m^2+n^2)) with S ~ (Z2-Z1)*Z1/Z2
            % (Z1 = zElt of the entrance sphere, Z2 = zElt of the element
            % after the mask) and PL2SPH (NF2) is a plain shifted FFT, so
            % Z2 == Z1 makes the unmasked round trip the exact identity
            % (measured 1.8e-15) and the geometric tail (identity on the
            % grid, per index) hands the detector the DM-conjugate pupil
            % with only the mask's action on it -- the textbook ZWFS.  The
            % sphere's physical sag over the beam at d_out is um-class; the
            % ray bookkeeping (dx labels, registration affine) is unchanged
            % (measured: dx_at(iDET) and the ray magnification identical to
            % the legacy emission).  ctb_dcr.in's FPM sandwich is symmetric
            % the same way (both EPreturn spheres at the same zElt).
            z_out = d_in;
        else
            % LEGACY (nf_legacy): Z2 = d_out ~= Z1 -> the round trip is a
            % Fresnel defocus of the pupil by z_eff = Z1*(Z1-Z2)/Z2 (4.86 m
            % on the zwfs_dm96 rig).  Reproduces the S1-S6 record only.
            z_out = d_out;
        end
        b.add_reference(d_out, 'MaskSphereOut', 'surface','Conic', ...
            'kr',-z_out, 'zelt',z_out);
        d_fl = P.D_MASK_FL - d_out;
    else
        d_m = mask_plate_(b, P.MASK_SUB, dmask);
        ix.iMASK = b.add_reference(d_m, 'FocalMask');
        d_fl = P.D_MASK_FL;
    end
    fl_args = {'mode','focus', 'n',P.N_GLASS, 'name','FL', 'edge_margin',P.EDGE_MARGIN};
    if ~isnan(P.FL_Kc), fl_args = [fl_args {'Kc', P.FL_Kc}]; end
    FL = b.add_lens(d_fl, P.FL_F, P.FL_D, fl_args{:});
    if ~isempty(conj_elt)
        s_o  = L2.s - b.E(conj_elt).s;
        s_i1 = 1/(1/L2.F - 1/s_o);                 % DM image via L2
        d12  = b.E(FL.i_pow).s - L2.s;
        s_o2 = d12 - s_i1;                         % <0 = virtual object
        s_i2 = 1/(1/P.FL_F - 1/s_o2);
        det_leg = s_i2 - FL.thickness + P.DET_TRIM;
    end
    if P.ep_ref
        ix.iEP  = b.add_reference(det_leg, 'ExitPupil');   % nElt-1 for pupil_quality
        ix.iDET = b.add_detector(1e-6, 'Detector');
    else
        ix.iDET = b.add_detector(det_leg, 'Detector');
    end

case 'doublet'                     % C2: L2 as two air-spaced singlets
    assert(~strcmp(P.optics,'oap'), ...
        'twyman_green: optics=''oap'' supports tail_arch singlet/fieldlens only.');
    aA = {'mode','focus', 'n',P.N_GLASS, 'name','L2A', 'edge_margin',P.EDGE_MARGIN};
    if ~isnan(P.L2A_Kc), aA = [aA {'Kc', P.L2A_Kc}]; end
    A = b.add_lens(d_rc_l2, P.L2A_F, P.D_LENS, aA{:});
    aB = {'mode','focus', 'n',P.N_GLASS, 'name','L2B', 'edge_margin',P.EDGE_MARGIN};
    if ~isnan(P.L2B_Kc), aB = [aB {'Kc', P.L2B_Kc}]; end
    gap = P.L2_SEP - A.thickness;
    assert(gap > 0, 'twyman_green: L2_SEP %.3g <= L2A thickness %.3g', ...
        P.L2_SEP, A.thickness);
    B = b.add_lens(gap, P.L2B_F, P.D_LENS, aB{:});
    % focus of the pair for collimated input (thin-lens seed; the runner
    % trims conics against the mask spot)
    s_iB = 1/(1/P.L2B_F - 1/(P.L2_SEP - P.L2A_F));
    assert(s_iB > B.thickness, 'twyman_green: doublet focus inside L2B');
    ix.iMASK = b.add_reference(mask_plate_(b, P.MASK_SUB, s_iB - B.thickness + P.MASK_TRIM), 'FocalMask');
    if ~isempty(conj_elt)
        s_o  = b.E(A.i_pow).s - b.E(conj_elt).s;
        s_i1 = 1/(1/P.L2A_F - 1/s_o);
        d12  = b.E(B.i_pow).s - b.E(A.i_pow).s;
        s_o2 = d12 - s_i1;
        s_i2 = 1/(1/P.L2B_F - 1/s_o2);
        det_leg = (b.E(B.i_pow).s + s_i2) - b.E(ix.iMASK).s + P.DET_TRIM;
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');
end
end

% ---------------------------------------------------------------------
function a = ax_local(dir, deg)
%AX_LOCAL  A polarization axis at DEG (from the local x) in the transverse
%   plane of a beam travelling along DIR.  The local basis is
%   (u1, u2) = (perp(dir), cross(dir, perp(dir))), the SAME right-handed
%   transverse frame the Bench emitter uses for xObs, so "45 deg" means the
%   same physical direction in every folded leg.  Returned as a global
%   3-vector already in the transverse plane (exactly normal-incidence).
    u1 = macos.design.Bench.perp(dir(:));
    u2 = cross(dir(:), u1);
    a  = cosd(deg)*u1 + sind(deg)*u2;
end

function d_rest = mask_plate_(b, sub, d_total)
%MASK_PLATE_  The mask's substrate, inserted INSIDE the gap ahead of it.
%   The two faces are placed so that the distance still to run after them is
%   D_REST and the TOTAL is unchanged at D_TOTAL -- so the mask (or the
%   sandwich's entrance sphere) does not move a micron when the plate is
%   switched on, and the only thing that changes is what the beam went
%   through on the way.  That is what makes the plate's cost measurable: the
%   focus shift t*(1-1/n) and the spherical aberration it carries show up as
%   a CHANGE in the tail's null and in the retuned DET_TRIM, not mixed with a
%   geometry move.
%
%   The faces sit in the CONVERGING beam, which is the whole point: in a
%   collimated leg a plane-parallel plate is pure path.
if isempty(sub), d_rest = d_total; return; end
assert(numel(sub) == 2 && sub(1) > 1 && sub(2) > 0, ...
    'twyman_green: MASK_SUB must be [n t] with n > 1 and t > 0.');
gap = 1.0;                                  % mm of air behind the plate, so
                                            % no two elements are coincident
assert(d_total > sub(2) + 2*gap, ...
    'twyman_green: MASK_SUB plate (%g mm) does not fit in the %g mm run to the mask.', ...
    sub(2), d_total);
b.add_substrate(d_total - sub(2) - gap, sub(1), sub(2), 'name','MaskSub');
d_rest = gap;
end
