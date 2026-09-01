function run_offaxis_petzval(varargin)
%RUN_OFFAXIS_PETZVAL  The construction the arc has never tried: a
%   PETZVAL-BALANCED double Mersenne, Cassegrain stage + Gregorian stage.
%
%   WHY THIS, AND WHY NOW.  The rung-2/rung-3 decomposition (RESULTS § O.6b)
%   says every design in this study is dominated by POWER -- 60 % of the
%   variance for the coaxial four- to seven-mirror family, 98 % for the
%   off-axis Mersenne, and **99.7 %** for the arc's own parabolic double
%   Mersenne, whose famous 59 374 nm is 3034 nm once power is removed.  For an
%   AFOCAL system residual power means the output beam is not collimated field
%   by field, and the per-field scan confirms the law: on the off-axis
%   Cassegrain Mersenne the rung-2 error runs 3666 -> 18749 nm across the box,
%   a factor 5.11 where theta^2 over the same range predicts 5.90, and it is
%   EXACTLY 0.00 nm on axis.  That is field curvature.
%
%   Field curvature is a PETZVAL sum, and a Petzval sum is not reduced by
%   adding mirrors -- it is reduced by adding them with the RIGHT SIGNS.  A
%   Cassegrain compressor (convex secondary) and a Gregorian compressor
%   (concave secondary) contribute OPPOSITE-SIGN Petzval.  The arc's existing
%   double Mersenne is two CASSEGRAIN stages, so its two contributions ADD --
%   which is a sufficient explanation for why relaxing four conics bought it a
%   factor of 1.7 (§ 4) and left 90 % of the residual still in power.  Nobody
%   has built the mixed pair.
%
%   THE CONSTRUCTION IS EXACT AND NEEDS NO CLOSURE.  Each stage is a confocal
%   parabola pair, so each is exactly afocal at its own magnification, and the
%   cascade is exactly afocal at m1*m2 -- by geometry, for a beam entering
%   anywhere on it.  In particular DESCENT_CLOSE is not involved, so the
%   singularity of § O.5 does not arise: there is nothing left for a closure
%   to solve.  The gap between the stages is FREE (the intermediate beam is
%   collimated), which is the knob that packages the thing.
%
%   WHAT IS SCANNED.  The split of the total 30x between the stages, m1, with
%   m2 = 30/m1.  Petzval balance is a condition on the two stages' focal
%   lengths, so the split is the physical knob; the scan reports rung 2, rung
%   3, and the power fraction, and the question is whether the power term goes
%   through a MINIMUM at some split rather than falling monotonically with
%   stage speed.  A minimum interior to the scan is the signature of
%   cancellation; a monotone curve would mean the mixed pair buys nothing
%   beyond being slower, and that is a real possible outcome reported as such.
%
%   Every row carries traced M, collimation and its ray count, per the
%   standing guard.
%
%   Env: OP_M1 (comma list), OP_F1, OP_GAP, OP_H, OP_OUT.

    ap = fileparts(fileparts(mfilename('fullpath')));
    addpath(ap); addpath(fullfile(ap,'clearing')); addpath(fullfile(ap,'descent'));
    addpath(fullfile(ap,'offaxis'));

    m1s  = str2double(strsplit(getenv_d('OP_M1','2,3,4,5,6,7.5,10,15'), ','));
    f1   = str2double(getenv_d('OP_F1','2.5'));
    gap  = str2double(getenv_d('OP_GAP','0.30'));
    hstr = getenv_d('OP_H','0,0.55');
    hs   = str2double(strsplit(hstr,','));
    outd = getenv_d('OP_OUT', fullfile(ap,'offaxis','decks'));
    if ~exist(outd,'dir'), mkdir(outd); end

    P = afocal4_params();
    macos.init(P.model_size);

    fprintf('\n==== PETZVAL-BALANCED DOUBLE MERSENNE ====\n');
    fprintf(['  stage 1 CASSEGRAIN (convex secondary), stage 2 GREGORIAN ' ...
             '(concave);\n  m1*m2 = %g exactly, f1 = %.2f m, inter-stage gap ' ...
             '%.2f m.\n'], P.M, f1, gap);
    fprintf(['  reference: arc double Mersenne (two CASS stages) 59374 nm ' ...
             'rung2 / 3034 rung3.\n\n']);
    fprintf('  %-5s %6s %8s %8s %10s %10s %7s %9s %10s %6s\n', 'form','m1','m2', ...
            'h','rung2 nm','rung3 nm','%% pow','traced M','coll urad','lost');

    rows = struct('form',{},'m1',{},'m2',{},'h',{},'r2',{},'r3',{}, ...
                  'pow',{},'M',{},'coll',{},'lost',{},'deck',{});
    for form = {'cass_greg','cass_cass'}
      fm = form{1};
      for m1 = m1s
        m2 = P.M/m1;
        for h = hs
            deck = fullfile(outd, sprintf('pz_%s_m%.4g_h%g.in', fm, m1, h));
            try
                build_(P, deck, f1, m1, m2, gap, fm);
            catch ME
                fprintf('  %-5s %6.2f %8.3f %8.2f  BUILD %s\n', fm, m1, m2, h, ...
                        ME.message);   continue;
            end
            oa = struct('nlost',0);
            if h ~= 0
                oa = offaxis_decenter(deck, h, 'fields',P.Fsolve, 'quiet',true);
                tr = oa.traced;
            else
                tr = traced_(deck);
            end
            try
                S = afocal4_score(P, deck, 'fields',P.Fsolve, ...
                                  'nodes',P.solve.nodes_score, 'pupil',false);
            catch ME
                fprintf('  %-5s %6.2f %8.3f %8.2f  SCORE %s\n', fm, m1, m2, h, ...
                        ME.message);   continue;
            end
            pw = 100*(1 - (S.wfe_rung3_max_nm/max(S.wfe_max_nm,eps))^2);
            fprintf('  %-9s %6.2f %8.3f %8.2f %10.1f %10.1f %7.1f %9.4f %10.1f %6d\n', ...
                    fm, m1, m2, h, S.wfe_max_nm, S.wfe_rung3_max_nm, pw, ...
                    tr.mag, tr.collimation_urad, oa.nlost);
            rows(end+1) = struct('form',fm,'m1',m1,'m2',m2,'h',h, ...
                'r2',S.wfe_max_nm,'r3',S.wfe_rung3_max_nm,'pow',pw, ...
                'M',tr.mag,'coll',tr.collimation_urad,'lost',oa.nlost, ...
                'deck',deck); %#ok<AGROW>
        end
      end
      fprintf('\n');
    end
    save(fullfile(outd,'offaxis_petzval.mat'), 'rows','P','-v7.3');

    % ---- the reading: is there an interior minimum? ----------------------
    for form = {'cass_greg','cass_cass'}
      for h = hs
        m = arrayfun(@(r) strcmp(r.form,form{1}) && r.h==h, rows);
        if nnz(m) < 3, continue; end
        rr = rows(m);   [~,b] = min([rr.r2]);
        interior = b > 1 && b < numel(rr);
        fprintf(['  %-9s h %.2f : best rung2 %9.1f nm at m1 = %.2f  (%s)\n'], ...
                form{1}, h, rr(b).r2, rr(b).m1, ...
                tern_(interior, ...
                  'INTERIOR minimum -- consistent with Petzval cancellation', ...
                  'at a scan END -- monotone, no cancellation signature'));
      end
    end
    fprintf('\n');
end

% =====================================================================
function build_(P, deck, f1, m1, m2, gap, form)
%BUILD_  Two confocal parabola pairs in cascade.  Exact by construction: each
%   pair is afocal at its own magnification, so the cascade is afocal at
%   m1*m2 and no closure is solved.
    f2 = f1/m1;                       % stage 1 secondary
    sep1 = f1 - f2;                   % Cassegrain: convex secondary
    f3 = f1/m1;                       % stage 2 primary, sized off the beam it gets
    f4 = f3/m2;
    switch form
    case 'cass_greg', sep2 = f3 + f4;   cvx2 = false;   % GREGORIAN: concave
    case 'cass_cass', sep2 = f3 - f4;   cvx2 = true;    % the arc's existing form
    end
    if sep1 <= 0.02 || sep2 <= 0.02
        error('macos:design:petzval:degenerate', ...
              'stage separation %.4f / %.4f m is degenerate', sep1, sep2);
    end
    t = macos.design.Telescope('family','tma', 'aperture_diameter_m',P.D, ...
            'wavelength_m',P.lambda, 'grid_npts',P.ngrid, ...
            'model_size',P.model_size);
    t.add_mirror('M1','radius_m',2*f1,'spacing_after_m',sep1,'convex',false,'conic',-1);
    t.add_mirror('M2','radius_m',2*f2,'spacing_after_m',gap, 'convex',true, 'conic',-1);
    t.add_mirror('M3','radius_m',2*f3,'spacing_after_m',sep2,'convex',false,'conic',-1);
    t.add_mirror('M4','radius_m',2*f4,'spacing_after_m',P.iface,'convex',cvx2,'conic',-1);
    t.add_exit_reference('ColdStop','dist_m',P.iface);
    if P.bias_deg ~= 0, t.set_field_bias(P.bias_deg*60); end
    t.build(deck);
end

function s = traced_(deck)
    tk = regexp(fileread(deck),'(?m)^\s*Aperture=\s*([^\n]*)','tokens','once');
    Dap = sscanf(strrep(tk{1},'D','E'),'%f',1);
    macos.load_rx(deck);
    tr = macos.trace(macos.num_elt());   ri = macos.get_ray_info(tr.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
    if ~any(ok)
        s = struct('mag',NaN,'collimation_urad',NaN,'exit_dia',NaN);  return;
    end
    dd = ri.dir(:,ok);   dd = dd ./ vecnorm(dd);
    dm = mean(dd,2);     dm = dm/norm(dm);
    q  = ri.pos(:,ok) - mean(ri.pos(:,ok),2);   q = q - dm*(dm.'*q);
    dia = 2*max(vecnorm(q));
    s = struct('mag',Dap/max(dia,realmin), ...
               'collimation_urad',max(acos(min(1,dm.'*dd)))*1e6, 'exit_dia',dia);
end

function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
