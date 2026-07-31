function R = xp_optimize(map_n, max_rounds, joint, variant)
%XP_OPTIMIZE  Re-solve stages 3 and 4 against the EXIT-PUPIL-REFERENCED merit.
%
%   R = XP_OPTIMIZE()                          map_n = 9, max_rounds = 4,
%                                              joint = true, EPD 4060
%   R = XP_OPTIMIZE(9, 4, true, 'seq')         the same solve at the CODE V
%                                              .seq truth: EPD 5000, the M1
%                                              hole, and HIS 15-point
%                                              half-box optimisation field
%                                              set (CALIB caps at 12 FoV, so
%                                              the 15 are decimated to 12 --
%                                              reported, not silent).
%                                              Artifacts suffixed _seq.
%
%   Step 3 of the Rodgers arc.  The committed stages 3/4 were solved against
%   `OPD` at the terminal FocalPlane -- std(OPL) to each ray's OWN intercept
%   on a detector plane tilted 14.3 deg, which carries
%   (transverse ray aberration) x tan(tilt), ~22x the wavefront error
%   (PACKET Addendum 3 §A.1, Addendum 4 §A).  Here the merit is a per-field
%   chief-ray-tied exit-pupil sphere -- the strict metric: `add_pupil` puts
%   an ExitPupil Return at nElt-1, `optimize` sets `OptWFElt = nElt-1` and
%   `OptFEX= Yes` and sets the stop, and CALIB re-runs FEX per field.
%   REQUIRES the OptFEX engine fix (macos PR #68); without it the keyword is
%   silently ignored, the reference sphere sticks at the on-axis image, and
%   the solve runs away.  Gate: mmacos/tests/tOptFex.m.
%
%   NAMING.  `_xpopt`, not `_orsopt`: the merit is the FEX-set EXIT-PUPIL
%   sphere tied to the frozen detector, NOT the literal ORS command
%   (`CRSOPTIMIZE`), which optimises the sphere RADIUS and so removes
%   per-field focus -- the opposite of what a fixed FPA imposes, and in any
%   case not reachable from CALIB (`MACOS_OPS` has no ORS branch).
%
%   ALTERNATION.  The merit's sphere is centred on each field's chief-ray
%   intercept on the DETECTOR, so the detector must be right for the merit to
%   be right, and it is refitted from the converged design.  Each round:
%   solve -> re-fit the FPA (align_focal_plane, 'allow_pupil') -> repeat until
%   the plane is stationary.
%
%   Scoring is by `strict_wfe_deck` on the emitted deck -- an INDEPENDENT
%   path from the in-loop merit, so a solve that games its own objective
%   shows up here.

    if nargin < 1, map_n = 9;      end
    if nargin < 2, max_rounds = 4; end
    if nargin < 3, joint = true;   end
    if nargin < 4 || isempty(variant), variant = 'epd4060'; end
    isseq = strcmpi(variant,'seq');
    % JOINT (default) solves the detector WITH the optics -- align_focal_plane
    % once as a seed, then the FPA's tilt + focus enter the CALIB DOF set and
    % there is NO alternation loop.  This mirrors Rodgers' own procedure and
    % removes the two-objective mismatch the alternation showed (Addendum 7:
    % the FPA was still drifting 0.6-13 mm per round after four rounds).
    % FPA DOF mask: [TIP TILT CLOCK DX DY PIST ROC CONIC] -> TIP (rotation
    % about local x = the alpha tilt) and PIST (translation along the local z
    % = the surface normal = focus/Tz).  Confirmed against
    % macos_ops.F:CPERTURB_2, where PV(1:3) is the rotation and PV(4:6) the
    % translation in the element frame.  NOT DOFs 3/4, which are CLOCK (a
    % near-null direction on a detector) and DX (a lateral shift the chief-ray
    % tie absorbs).
    FPA_DOFS = [1 0 0 0 0 1 0 0];
    TOL_FP   = 5e-5;      % 0.05 mm station move
    TOL_TILT = 1e-4;      % ~0.006 deg normal move

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    if isseq
        P = rodgers_common('seq');
    else
        P = rodgers_common();  P.EPD_mm = 4060;
    end
    lam_nm  = P.lambda_m*1e9;
    if isseq
        % HIS field set, decimated to CALIB's 12-FoV cap.  Keep the on-axis
        % (0,0) point implicit (optimize drops it) and drop the three points
        % with the SMALLEST radius from the box centre -- the least
        % informative, and the choice is stated here rather than hidden.
        % CALIB caps the TOTAL FoV count at 12, and the on-axis field is
        % IMPLICIT (optimize adds it), so at most 11 explicit points fit.
        % Keep the 11 largest-radius of his 15 -- the informative ones -- and
        % SAY which were dropped rather than truncating silently.
        Fs = P.seq.Frel;
        Fs = Fs(any(abs(Fs) > 1e-12, 2), :);          % drop (0,0): implicit
        [~,ord] = sort(vecnorm(Fs,2,2), 'descend');
        keep = sort(ord(1:min(11,size(Fs,1))));
        optF = Fs(keep,:);
        drop = setdiff(1:size(Fs,1), keep);
        fprintf(['  optimisation field set: %d explicit + 1 implicit on-axis = %d ' ...
                 'FoV,\n    from his %d (CALIB caps at 12).  Dropped %d ' ...
                 'smallest-radius point(s):\n'], ...
                size(optF,1), size(optF,1)+1, size(P.seq.Frel,1), numel(drop));
        for q = drop
            fprintf('      (XAN %+7.4f, dYAN %+7.4f) deg\n', rad2deg(Fs(q,1)), rad2deg(Fs(q,2)));
        end
        Frel = P.seq.Frel;                             % score on ALL 15
    else
        optF = macos.design.field_grid(P.fov_half_deg*60, 3, 'units','arcmin','origin',false);
        Frel = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');
    end
    R = struct('map_n',map_n,'lambda_nm',lam_nm,'stage',cell(1,2));

    % 'ref' = the committed FP-merit solve, 'his' = HIS design strict-scored.
    % Both are EPD-4060 / 9x9-box numbers, so at the .seq truth they are NOT
    % comparable and are blanked rather than printed misleadingly; the seq
    % 'his' column is filled in by RUN_SEQ from the matching his_designs run.
    if isseq
        ref3 = [NaN NaN]; his3 = [NaN NaN];  ref4 = [NaN NaN]; his4 = [NaN NaN];
    else
        ref3 = [181.234 97.059];  his3 = [115.312 53.652];
        ref4 = [118.591 84.806];  his4 = [ 64.851 35.358];
    end
    specs = { struct('st',3, 'dofs',[0 0 0 0 0 0 0 1], 'gt',P.gt.s3_box, ...
                     'ref',ref3, 'his',his3), ...
              struct('st',4, 'dofs',[0 0 0 0 0 0 0 1; 1 0 0 0 1 0 0 1; 1 0 0 0 1 0 0 1], ...
                     'gt',P.gt.s4_box, 'ref',ref4, 'his',his4) };

    for c = 1:numel(specs)
        S = specs{c};
        banner('STAGE %d -- re-solve against the exit-pupil merit', S.st);
        t = build_tma(P, P.K_nom, P.offset_deg);
        t.align_focal_plane('grid',5,'span_arcmin',6);      % before add_pupil
        t.add_pupil();
        nE = numel(t.spec.elt);
        fprintf('  deck is now %d elements; ExitPupil at %d\n', nE, t.spec.pupil.ep_elt);

        rounds = 0;  hist = [];
        Vseed = t.spec.elt(nE).Vpt(:);  Nseed = t.spec.elt(nE).psi(:);
        Nseed = Nseed/norm(Nseed);
        if joint
            % ---- ONE solve, detector in the DOF set ----------------------
            res = t.optimize('fields', optF, 'dofs', S.dofs, ...
                             'fpa_dofs', FPA_DOFS, 'max_iters', 120);
            rounds = 1;
            R(c).converged = res.converged;
            V = t.spec.elt(nE).Vpt(:);  N = t.spec.elt(nE).psi(:); N = N/norm(N);
            R(c).fpa_move_mm  = norm(V - Vseed)*1e3;
            R(c).fpa_tilt_deg = acosd(min(1,abs(dot(N,Nseed))));
            fprintf(['  JOINT solve: converged=%d;  FPA moved %.4g mm and ' ...
                     '%.4g deg from the align seed\n'], ...
                    res.converged, R(c).fpa_move_mm, R(c).fpa_tilt_deg);
            fprintf('  merit WFE before/after (per FOV, waves): %s -> %s\n', ...
                    mat2str(res.wfe_before,4), mat2str(res.wfe_after,4));
        else
            Vprev = Vseed;  Nprev = Nseed;
            for r = 1:max_rounds
                t.optimize('fields', optF, 'dofs', S.dofs, 'max_iters', 120);
                fp = t.align_focal_plane('grid',5,'span_arcmin',6,'allow_pupil',true);
                rounds = r;
                dV = norm(fp.fp_vpt(:) - Vprev);
                dN = norm(fp.psi(:)/norm(fp.psi) - Nprev/norm(Nprev));
                K  = [t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc];
                fprintf(['  round %d: FPA moved %.4g mm / %.4g rad;  ' ...
                         'K = %.9f %.9f %.9f\n'], r, dV*1e3, dN, K);
                hist(end+1,:) = [r dV dN K]; %#ok<AGROW>
                Vprev = fp.fp_vpt(:);  Nprev = fp.psi(:);
                if dV < TOL_FP && dN < TOL_TILT
                    fprintf('  converged: FPA stationary within tolerance.\n');
                    break;
                end
            end
        end

        R(c).stage = S.st;  R(c).rounds = rounds;  R(c).hist = hist;
        R(c).K = [t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc];
        R(c).rigid = [rigid_of(t.spec.elt(2)); rigid_of(t.spec.elt(3))];

        deck = fullfile(here, sprintf('rodgers1_%s_stage%d_xpopt.in', variant, S.st));
        t.save(deck);
        s = strict_wfe_deck(deck, Frel);
        w = s.wfe_m(isfinite(s.wfe_m))*1e9;
        R(c).scan = s;  R(c).min = min(w);  R(c).max = max(w);  R(c).avg = mean(w);
        R(c).gt = S.gt; R(c).ref = S.ref;   R(c).his = S.his;  R(c).deck = deck;
        R(c).nfin = numel(w);  R(c).ntot = numel(s.wfe_m);

        Kgt = P.K_s3;  if S.st == 4, Kgt = P.K_s4; end
        fprintf('  conics:   MACOS(xp)        Rodgers          |diff|\n');
        for i = 1:3
            fprintf('    K_M%d %16.9f %16.9f   %.2e\n', i, R(c).K(i), Kgt(i), ...
                    abs(R(c).K(i)-Kgt(i)));
        end
        if S.st == 4
            fprintf('  rigid body, IN THE DECODED FRAME (his ADE sign flipped):\n');
            hy = [P.Ydec_M2_mm P.Ydec_M3_mm];  ha = -[P.tilt_M2_deg P.tilt_M3_deg];
            old = [-3.741681 0.233023; -43.838729 1.114205];   % committed FP-merit solve
            for i = 1:2
                fprintf(['    M%d  Ydec %10.4f mm (was %10.4f, his %10.4f)' ...
                         '   alpha %8.4f deg (was %8.4f, his %8.4f)\n'], ...
                        i+1, R(c).rigid(i,1), old(i,1), hy(i), ...
                        R(c).rigid(i,2), old(i,2), ha(i));
            end
        end
        fprintf('  rays/field %d..%d,  %d/%d fields finite,  %d solve(s)\n', ...
                min(s.nrays), max(s.nrays), R(c).nfin, R(c).ntot, rounds);
        fprintf('  STRICT (nm)  min %9.3f  max %9.3f  avg %9.3f\n', R(c).min, R(c).max, R(c).avg);
        fprintf('  was (FP merit)                max %9.3f  avg %9.3f\n', S.ref(1), S.ref(2));
        fprintf('  HIS design                    max %9.3f  avg %9.3f\n', S.his(1), S.his(2));
        fprintf('  Rodgers reported              max %9.3f  avg %9.3f\n', ...
                S.gt(2)*lam_nm, S.gt(3)*lam_nm);
        fprintf('  ratio vs Rodgers: max %.3f x   avg %.3f x   (was %.3f / %.3f)\n', ...
                R(c).max/(S.gt(2)*lam_nm), R(c).avg/(S.gt(3)*lam_nm), ...
                S.ref(1)/(S.gt(2)*lam_nm), S.ref(2)/(S.gt(3)*lam_nm));

        png = fullfile(here, sprintf('rodgers1_%s_stage%d_xpopt_strict.png', variant, S.st));
        scan = struct('fields', s.fields*180/pi*60 + s.bias_deg*60, ...
                      'wfe', s.wfe(:), 'metric','strict');
        fig = t.view_field_map(scan,'kind','contour','save',png,'visible',false);
        close(fig);  R(c).png = png;
    end

    banner('EXIT-PUPIL RE-SOLVE  (%s: EPD %g, nm @ %g nm, %d fields)', variant, P.EPD_mm, lam_nm, size(Frel,1));
    fprintf('  stage | xp-merit max/avg | FP-merit max/avg | HIS design | Rodgers | max x\n');
    for c = 1:numel(R)
        g = R(c).gt;
        fprintf('  S%-4d | %8.1f/%-8.1f | %8.1f/%-8.1f | %6.1f/%-6.1f | %5.1f/%-5.1f | %5.2f\n', ...
            R(c).stage, R(c).max, R(c).avg, R(c).ref(1), R(c).ref(2), ...
            R(c).his(1), R(c).his(2), g(2)*lam_nm, g(3)*lam_nm, ...
            R(c).max/(g(2)*lam_nm));
    end
    save(fullfile(here,sprintf('rodgers1_%s_xpopt.mat',variant)),'R');
    fprintf('\nsaved rodgers1_%s_xpopt.mat + decks + maps\n', variant);
end

function v = rigid_of(e)
    psi = e.psi(:)/norm(e.psi);
    v = [e.Vpt(2)*1e3, atan2d(psi(2), -psi(3))];
end

function t = build_tma(P, K, bias_deg)
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',K(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',K(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',K(3),'spacing_after','derive');
    if isfield(P,'M1_hole_m') && P.M1_hole_m > 0
        t.set_hole('M1', P.M1_hole_m);     % CODE V "CIR HOL" on M1
    end
    if bias_deg ~= 0, t.set_field_bias(bias_deg*60); end
    t.build();
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
