function P = offset_imager_params(over)
%OFFSET_IMAGER_PARAMS  Single source of truth for the offset_imager template.
%
%   P = OFFSET_IMAGER_PARAMS() returns the default parameter set -- the
%   rodgers3 challenge instance (Mike Rodgers' 22-deg offset-field
%   imager, 260802-WFOVimager_Offsetfield-jmr): EPD 75 mm, F/4, 1 um,
%   20x20-deg field box offset 22 deg, his packaging envelope.
%
%   P = OFFSET_IMAGER_PARAMS(OVER) applies the fields of struct OVER on
%   top of the defaults (the e2e2_params pattern) -- this is how a second
%   instance is run without touching this file:
%
%       P = offset_imager_params(struct('EPD_m',0.200,'Fno',2.5, ...
%             'box_deg',[10 10],'offset_deg',12, ...));
%
%   Every stage of OFFSET_IMAGER reads ONLY this struct.  Fields:
%
%   Identity / optics
%     EPD_m        entrance pupil (= input beam) diameter, m
%     Fno          working focal ratio: EFL = EPD_m * Fno is enforced
%                  EXACTLY by the first-order closure at every solve
%                  iterate (afocal4 doctrine: identities are re-derived,
%                  never penalized)
%     lambda_m     scoring wavelength, m
%
%   Field
%     box_deg      [XAN_full  YAN_full] field box FULL widths, deg
%     offset_deg   YAN of the box centre (0 = on-axis box)
%     nsolve       solve set is an nsolve x nsolve grid over the box
%                  (Mike used 3x3 = 9 points; solve set != scoring set)
%     map_n        dense scoring/report map is map_n x map_n
%
%   Packaging (the designer's envelope -- inputs, not solved)
%     z_m1_m       global z of the M1 vertex (beam enters +z), m
%     spacings_m   SIGNED thickness chain [m1->stop, stop->m2, m2->m3], m
%                  (CODE V sign convention: negative = beam travels -z).
%                  The rodgers3 defaults put the stop ON the m2 plane
%                  (stop->m2 = 0), 58 mm upstream of m1.
%     bfd_m        [] = back focus free (FP posed at the traced focus);
%                  a value pins the recenter thickness instead
%
%   Constraint set (evaluated as report gates; S4+ walls)
%     exit_dir     [] = report-only; a 1x3 unit vector pins the exit
%                  chief direction (Mike r2+: exit beam horizontal =
%                  [0 0 -1] after an odd mirror count ... stated per
%                  instance)
%     exit_tol_deg tolerance on the exit-chief direction gate, deg
%     clear_m      list of clearance requirements, m (each is checked as
%                  min distance of every traced beam leg to every mirror
%                  edge not on that leg; Mike r4+: >0.050 and >0.035)
%
%   Surface budget (the ladder; stages can be disabled)
%     stages       which of S1..S5 to run (default 1:5)
%     zern_modes   S5 Zernike term set (BornWolf engine mode numbers);
%                  see the default's comment below for the doctrine
%
%   Numerics
%     model        MACOS model size
%     sampling     deck nGridpts (41 -> ~1184 rays) for every REPORTED map
%     solve_sampling  nGridpts inside the solve loop only (default 21)
%     seed_R1_m    M1 radius seed for the first-order seed solver (the
%                  third first-order condition alongside EFL and
%                  Petzval = 0; see OI_PARAXIAL/oi_seed)
%     gn_iters     damped Gauss-Newton iteration cap per stage
%
%   Output
%     tag          filename prefix for decks/figures/reports
%     outdir       artifact directory ('' = the template directory)
%
%   See also OFFSET_IMAGER, OI_DECK, OI_PARAXIAL, OI_SCORE.

    arguments
        over struct = struct()
    end

    % ---- the rodgers3 challenge instance (defaults) ---------------------
    P.name       = 'rodgers3';
    P.tag        = 'oi';
    P.EPD_m      = 0.075;
    P.Fno        = 4.0;
    P.lambda_m   = 1.0e-6;

    P.box_deg    = [20 20];
    P.offset_deg = 22;
    P.nsolve     = 3;
    P.map_n      = 11;

    P.z_m1_m     = 0.6649568;
    P.spacings_m = [-0.0579400  0  0.6828880];
    P.bfd_m      = [];

    P.exit_dir     = [];        % report-only by default; T3 sets Mike's
    P.exit_tol_deg = 1.0;
    P.exit_wt      = 1e4;       % exit-direction residual weight, nm/deg
                                % (an EQUALITY constraint solved in the
                                % least-squares sense from S3 on -- a
                                % boolean wall would freeze any stage
                                % that STARTS outside tolerance)
    P.clear_m      = [0.050 0.035];

    P.stages     = 1:5;
    % S5 Zernike term set, BornWolf ENGINE mode numbers.  Default = the
    % x-symmetric set to 8th radial order with PISTON CARRIED (their
    % frozen-thickness despace surrogate) and POWER (mode 5) + TILTS
    % (modes 2,3) HELD OUT -- power is pinned to the radii and tilt to
    % the pointing/ADE, per the Zernike solve doctrine.  This is exactly
    % Mike's varied C-set mapped by mode = C_idx - 1 (challenge Stage 0).
    P.zern_modes = [1 4 9 10 11 12 13 19 20 21 22 23 24 25 ...
                    33 34 35 36 37 38 39 40 41];

    P.model      = 256;
    P.sampling   = 41;
    P.solve_sampling = 21;      % coarser ray grid inside the solve loop
                                % (reported numbers always use sampling)
    P.seed_R1_m  = 8.8;         % M1 radius scale for the first-order seed
    P.gn_iters   = 12;

    P.outdir     = '';

    % ---- overrides -------------------------------------------------------
    fn = fieldnames(over);
    for i = 1:numel(fn)
        if ~isfield(P, fn{i})
            error('offset_imager_params:unknown', ...
                  'unknown parameter "%s"', fn{i});
        end
        P.(fn{i}) = over.(fn{i});
    end

    % ---- derived (never set directly) ------------------------------------
    P.EFL_m = P.EPD_m * P.Fno;
end
