function P = tma_longslit_params(over)
%TMA_LONGSLIT_PARAMS  Single source of truth for the tma_longslit template.
%
%   P = TMA_LONGSLIT_PARAMS() returns the default parameter set: the dyson5
%   3k front end to Joe's spec (f 330 mm, D 183 mm, F/1.8, a 9.4-deg strip
%   onto a 54 mm slit, telecentric), seeded from the SBG VSWIR freeform TMA
%   (Bradley et al., ICSO 2024, Proc. SPIE 13699, 1369945, Fig. 4b, digitised
%   by CC 2026-10-07: legs 270/257/313 mm at f 345, chief AOI 30/37/15 deg).
%
%   P = TMA_LONGSLIT_PARAMS(OVER) applies the fields of struct OVER on top of
%   the defaults -- how a second instance is run without editing this file:
%
%       P = tma_longslit_params(struct('f_m',0.250,'D_m',0.120, ...
%             'strip_half_deg',3.0,'slit_m',0.026));
%
%   Every stage of TMA_LONGSLIT_RUN reads ONLY this struct.  Fields:
%
%   Spec (the requirement; Joe's sheet by default)
%     f_m            system focal length, m (plate scale along the slit)
%     D_m            entrance beam diameter, m (F/# = f/D)
%     strip_half_deg strip half-angle, deg (the field is OUT of the fold
%                    plane: along global x, the sagittal direction)
%     slit_m         slit length, m (2*f*tan(strip_half) should cover it)
%     pixel_m        detector pixel, m (spot / smile / keystone units)
%     telecentric_deg  chief rays at the slit within this of the slit
%                    normal across the strip
%     cone_fnum      [F#min F#max] the marginal-ray cone at the slit, BOTH
%                    directions (the paper's F/# anamorphicity constraint;
%                    Jim: "a little faster than the spectrometer")
%     work_dist_m    minimum working distance M3 -> slit, m (the Dyson
%                    hangs in it)
%     spec_joe / spec_paper   score thresholds (two columns of every table)
%
%   Seed (the form): the zig-zag TMA -- M1 concave, M2 small at the stop,
%   M3 carrying the power, a long working distance
%     seed_f_m       focal length the seed legs were digitised at, m
%     seed_legs_m    [M1->M2 M2->M3 M3->slit] at seed_f_m, m (scaled by
%                    f_m/seed_f_m at build)
%     aoi_deg        chief-ray angle of incidence at M1..M3, deg
%     turn           [+1 -1 +1]: the turn sense of the chief at each mirror
%                    about global x (M1 and M3 the same way, M2 back: the
%                    zig-zag, not a Korsch ring)
%     stop           'M2' (default) | 'M1' -- the element the chief is
%                    aimed through; first_order also reports the other
%     axis_side      [+1 +1 +1]: parent axis of each off-axis conic leans
%                    toward the INCOMING chief (+1) or the OUTGOING one (-1)
%     axis_theta_deg [] = the AOI (each section an off-axis PARABOLOID, the
%                    exact first-order section); else the angle between the
%                    local normal and the parent axis per mirror (the conic
%                    follows from the local radii)
%
%   Figure ladder (stage 'figure', TLS_FIGURE)
%     ladder         struct array: .name, .dofs (cell of TLS_FIGURE groups),
%                    .from (the rung index it warm-starts from, 0 = seed)
%     solve_fields_deg  [] = 5 fields over 0..strip half (the section is
%                    mirror-symmetric about the fold plane)
%     w_plate, w_bow row weights on the chief's slit-axis position (vs f tan)
%                    and across-slit position (vs the centre field), per um
%     w_tel          telecentric rows, um per rad of chief angle at the slit
%     w_cone         cone hinge rows, um per unit F/# outside cone_fnum
%     cone_tol       F/# slack on both ends of the cone hinge
%     w_wd           working-distance hinge, um per m below work_dist_m
%     maxfev         lsqnonlin evaluation budget per rung
%     aoi_span_deg, leg_span  bounds on the layout DOFs about the seed (the
%                    unbounded layout collapses to a coaxial, self-obscuring
%                    train)
%     resume_upto    figure stage: reuse R0..R<n> from the saved record
%     score_lambda_m wavelength of the Airy term in the spot FWHM (ARF) rows
%
%   Numerics
%     model          MACOS model size
%     ngridpts       source ray grid (circular)
%     nfield         strip fields across +-strip_half_deg (odd: includes 0)
%     lambda_m       wavelength for the decks, m
%
%   Output
%     tag            filename prefix for decks / records
%     outdir         artifact directory ('' = the template directory)
%
%   See also TMA_LONGSLIT_RUN, TMA_LONGSLIT, TLS_FIRST_ORDER, TLS_SECTION.

P = struct();
% ---- spec (Joe's sheet, deck_dyson spec slide)
P.f_m            = 0.330;
P.D_m            = 0.183;
P.strip_half_deg = 4.7;
P.slit_m         = 0.054;
P.pixel_m        = 18e-6;
P.telecentric_deg = 0.5;
P.cone_fnum      = [1.7 1.8];
P.work_dist_m    = 0.250;
P.spec_joe   = struct('smile_px', 0.1, 'keystone_px', 0.1, 'crf_px', 1.5, 'srf_px', [1.5 2.0], 'eip', 0.75);
P.spec_paper = struct('smile_px', 0.05, 'keystone_px', 0.10, 'srf_px', 1.8, 'crf_px', 2.8, 'arf_px', 2.8);   % Bradley 2024 Table 1 (smile 5 % / keystone 10 % of a pixel)
% ---- seed: SBG VSWIR Fig. 4b (f 345), digitised +-5 mm / +-2 deg
P.seed_f_m       = 0.345;
P.seed_legs_m    = [0.270 0.257 0.313];
P.aoi_deg        = [30 37 15];
P.turn           = [+1 -1 +1];
P.stop           = 'M2';
P.axis_side      = [+1 +1 +1];
P.axis_theta_deg = [];
% ---- figure ladder
% R_t, R_s are re-derived at every iterate (X.closure); .from = the rung to warm-start from (0 = the seed).  The
% asphere rung is kept as the RECORD of why the even aspheres fail on these sections (their parent axes are 0.9-1.5 m
% from the poles: an h^4 term is almost all pole tilt + curvature, which the closure absorbs) -- the layout and
% freeform rungs branch from the conic rung, not from it.
P.ladder = struct('name', {'conic', 'asph', 'geom', 'ff34', 'ff', 'ff2', 'ffw', 'fft', 'ffo'}, ...
                  'dofs', {{'theta', 'slit_dz'}, ...
                           {'theta', 'slit_dz', 'asph'}, ...
                           {'theta', 'slit_dz', 'aoi', 'legs'}, ...
                           {'theta', 'slit_dz', 'mon34'}, ...
                           {'theta', 'slit_dz', 'mon'}, ...
                           {'theta', 'slit_dz', 'mon'}, ...
                           {'theta', 'slit_dz', 'mon'}, ...
                           {'theta', 'slit_dz', 'mon'}, ...
                           {'theta', 'slit_dz', 'mon'}}, ...
                  'from', {0, 1, 1, 1, 4, 5, 4, 7, 7}, ...
                  'maxfev', {[], [], [], [], [], 3000, 3000, 3000, 3000}, ...
                  'w_spot_x', {[], [], [], [], [], [], 3, 3, 3}, ...
                  'field_wt', {[], [], [], [], [], [], [1 1 2 3 3], [1 1 2 3 3], [1 1 2 3 3]}, ...
                  'w_tel', {[], [], [], [], [], [], [], 3e5, []}, ...
                  'w_off', {[], [], [], [], [], [], [], [], 1000});   % ffo: 1000, so R7's 2.4 um edge offset costs ~8e6 = the merit's scale (x30 was 7e3: R8's lesson)
% ff2: ff continued, with the centroid-bow rows (CBOW).  ffw (CC 2026-10-07): from R4 (the CRF-best rung), the
% ALONG-slit spot rows x3 (the CRF direction: the Dyson cannot fix it; the slit truncates the across-slit width) and
% the outer solve fields x2/x3, the CBOW rows kept.  fft (CC 2026-10-07): from R7, the TELE rows x30 (3e5 um/rad,
% target <= 0.005 deg; inconclusive: still 0.4 % of the merit).  ffo (CC 2026-10-07): from R7, + the OFF rows x30 (the
% point-source across-slit shift; < 1.8 um at the edge for 0.1 px).  The e2e smile (0.14 px) was tested against the telescope's along-track chief angle (-0.51 mrad at the
% strip edge, even in field -- smile's shape)
% (the layout rung is the RECORD of why the layout stays at the seed: bounded to +-5 deg it runs the AOIs to their
%  lower bounds, buys 153-341 -> 141-254 um and costs the clearance, -4.9 mm M3->slit vs M2; the freeform rungs branch
%  from the conic rung, layout frozen)
P.solve_fields_deg = [];
P.w_spot_x       = 1;            % SPOT rows along the slit (CRF direction); a rung's .w_spot_x overrides
P.w_spot_y       = 1;            % SPOT rows across the slit (ARF direction)
P.w_off          = 0;            % OFF rows (chief - centroid across the slit); a rung's .w_off overrides
P.solve_field_wt = [];           % per-solve-field weights ([] = ones); a rung's .field_wt overrides
P.w_plate        = 30;
P.w_bow          = 30;
P.w_tel          = 1e4;
P.w_cone         = 1e4;
P.cone_tol       = 0.005;      % F/# slack on the cone hinge (Joe's D 183 / f 330 is itself F/1.803)
P.w_wd           = 1e6;
P.maxfev         = 1500;
P.aoi_span_deg   = 5;            % layout bounds: AOI within +-5 deg of the seed (the form is kept)
P.leg_span       = 0.15;         % legs within +-15 % of the seed
P.resume_upto    = -1;
% ---- e2e (stage 'e2e', TLS_E2E): the spectrometer the telescope feeds, by the dyson5 join
P.e2e            = struct('tel_dyson', 'size:F:240', 'template', 'dyson5_t5e_tA_EP_3k_m30_B1_e2e.in');
P.e2e_roll_deg   = [0 180];
P.e2e_rung       = '';           % '' = the best clear rung; or a rung name           % stage 'figure': reuse rungs R0..R<n> from <tag>_figure.mat (-1 = none)
P.score_lambda_m = 2.5e-6;
P.mount_m        = 5e-3;
% ---- numerics
P.model          = 256;
P.ngridpts       = 41;
P.nfield         = 9;
P.lambda_m       = 633e-9;
% ---- output
P.tag            = 'tls';
P.outdir         = '';

if nargin > 0 && ~isempty(over)
    fn = fieldnames(over);
    for k = 1:numel(fn)
        assert(isfield(P, fn{k}), 'tma_longslit_params: unknown field ''%s''', fn{k});
        P.(fn{k}) = over.(fn{k});
    end
end
if isempty(P.outdir), P.outdir = fileparts(mfilename('fullpath')); end
P.legs_m = P.seed_legs_m*P.f_m/P.seed_f_m;     % the seed scaled to the spec focal length
end
