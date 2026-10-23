function OUT = rodgers3(opts)
%RODGERS3  Gate the five rodgers3 decks against Mike's WFE ladder.
%   Stage-0 reproduction runner (moved verbatim from the sandbox
%   run_r3_s0.m; artifact names kept), rung by rung, with the rodgers1
%   run_seq section-0 discipline: layout + traced convention gates
%   BEFORE any WFE number is believed.
%
%   Per rung:
%     0. LAYOUT GATE -- re-walk the .seq thickness chain independently and
%        check every emitted element datum; print the stations.
%     1. TRACED CONVENTION GATES (centre field):
%        - chief aimed by real-ray iteration through the STOP CENTRE
%          (CODE V aims reference rays at the stop);
%        - beam footprint at the stop plane vs the paraxially implied
%          stop diameter (EPD 75 imaged through m1);
%        - chief landing on the FP vs the deck SI origin -- HIS recenter
%          refit re-centres the image, so a correct convention map lands
%          the chief AT the SI origin.  A wrong ODD/EVEN choice misses by
%          ~2x the recenter YDE (~220 mm): decisive without touching WFE.
%     2. STRICT WFE over HIS 9-point field set (XAN/YAN tan-composed),
%        reference = the deck's own FPA (VERBATIM from the .seq -- no
%        refit), sphere anchored at the exit pupil (crossing of two aimed
%        chiefs = the stop image), piston-only removal, both centroid
%        (primary, Dave 2026-07-31 ruling) and chief references.
%     3. Verdict vs his "RMS WFE <= x": compared against BOTH max-over-
%        fields and field-RMS of the per-field strict RMS WFE.
%
%   Gate order is strict (NOTES_s0.md): a failed rung stops the ladder
%   (later rungs are reported as NOT RUN) unless 'force' is set.
%
%   Name-value:
%     'rungs'   which rungs to run (default 1:5, in order)
%     'suffix'  deck-set suffix from build_r3 (default '')
%     'force'   true = keep going past a failed gate (diagnosis runs)
%     'quiet'   suppress per-field lines (default false)
%
%   Writes r3_s0.mat and appends nothing -- write_report.m formats a
%   fresh r3_report.txt from OUT (the committed r3_s0_report.txt is the
%   FROZEN Stage-0 record and is never overwritten).

    arguments
        opts.rungs (1,:) double = 1:5
        opts.suffix (1,:) char = ''
        opts.force (1,1) logical = false
        opts.quiet (1,1) logical = false
        opts.map_n (1,1) double = 11   % dense-map grid (slide ceiling); 0 = off
        opts.save (1,1) logical = true % false: suite runs don't dirty the dir
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','mmacos_setup.m'));
    addpath(here);

    B = load(fullfile(here, sprintf('r3_build%s.mat',opts.suffix)));  D = B.D;
    S = rodgers3_seq();
    gates_nm = [159 8810 168 117 53];
    rungs = {'r1','r2','r3','r4','r5'};

    macos.init(256);
    OUT = struct('gate_nm',gates_nm,'suffix',opts.suffix);
    stopped = false;

    for k = opts.rungs
        rk = rungs{k};  G = D.(rk);  R = S.(rk);
        if stopped && ~opts.force
            OUT.(rk) = struct('ran',false);
            fprintf('\n[%s] SKIPPED -- earlier gate failed (strict order)\n', rk);
            continue
        end
        banner('%s  --  %s   (gate %g nm)', rk, G.title, gates_nm(k));

        % ============ 0. LAYOUT GATE =====================================
        z_m1   = (R.s(2).th + R.s(3).th)*1e-3;
        z_stop = z_m1 + R.s(4).th*1e-3;
        z_m3   = z_stop + R.s(5).th*1e-3 + R.s(6).th*1e-3;
        fprintf(['  stations (m): m1 %+11.7f   stop/m2 %+11.7f   m3 %+11.7f\n'], ...
                z_m1, z_stop, z_m3);
        dz = [abs(G.M(1).Vpt(3)-z_m1), abs(G.M(2).Vpt(3)-z_stop), ...
              abs(G.M(3).Vpt(3)-z_m3)];
        assert(max(dz) < 1e-12, 'layout gate: station mismatch %g', max(dz));
        for m = 1:3
            E = G.M(m);  ss = R.s([4 6 7]);  ss = ss(m);
            fprintf('  %s: Kr %+13.9f  Kc %+13.9f  Vpt(y) %+10.6f mm  alpha %+9.5f deg\n', ...
                E.name, E.Kr, E.Kc, E.Vpt(2)*1e3, atan2d(E.psi(2),-E.psi(3)));
            assert(abs(E.Kr - ss.R*1e-3) < 1e-15*max(1,abs(ss.R)) && ...
                   abs(E.Kc - ss.K) < 1e-12, 'layout gate: %s Kr/Kc', E.name);
            if ~isempty(E.asph)
                fprintf('      asph (m^-3,-5,-7): %+.6e %+.6e %+.6e  (= -ASP*1e9/1e15/1e21)\n', E.asph);
            end
            if ~isempty(E.zern)
                fprintf('      ZRN: lMon %.6f m, %d modes (max %d), piston C2 carried\n', ...
                        E.zern.lMon, numel(E.zern.modes), max(E.zern.modes));
            end
        end
        fprintf('  stop centre (m): [%+.6f %+.6f %+.6f]   (CODE V YDE %+.4f mm, sgn %+d)\n', ...
                G.stopC, getf_(R.s(5),'YDE',0), G.sgn.sgn_stop);
        fprintf('  SI:  Vpt [%+.6f %+.6f %+.6f]  psi [%+.7f %+.7f %+.7f]\n', ...
                G.V_si, G.psi_si);
        fprintf('  layout gate: PASS (emitted == .seq chain to round-off)\n');

        % ============ 1. TRACED CONVENTION GATES =========================
        txt0 = fileread(G.deck);
        tmp  = [tempname '.in'];
        cu   = onCleanup(@() delete_if_(tmp));

        % paraxially implied stop semi-diameter: on-axis parallel pencil
        % of semi-height h through m1, measured at the stop plane
        r_in = G.EPD_m/2;
        magf = stop_mag_(G);            % d(h_stop)/d(h_in) through m1
        r_stop_par = abs(magf)*r_in;

        % centre field, aimed
        yanC = R.YAN(2);
        d0   = tancomp_(0, yanC);
        [p0, aim] = aim_chief_(txt0, tmp, G, d0);
        st = trace_full_(txt0, tmp, p0, d0);
        okB = st.ok;  okB(1) = false;

        % footprint at the stop plane (from post-m1 states of the bundle)
        stf = trace_elt_(txt0, tmp, p0, d0, 1);
        okS = stf.ok; okS(1) = false;
        tS  = (G.stopC(3) - stf.pos(3,okS)) ./ stf.dir(3,okS);
        QS  = stf.pos(:,okS) + stf.dir(:,okS).*tS;
        fpc = mean(QS,2);  fpr = max(vecnorm(QS(1:2,:) - fpc(1:2),2,1));
        fprintf('\n  chief aiming: %d iter, stop-plane miss %.3e m\n', aim.niter, aim.miss);
        fprintf('  footprint at stop: centre [%+.4f %+.4f] mm vs stop [%+.4f %+.4f] mm\n', ...
                fpc(1:2)*1e3, G.stopC(1:2)*1e3);
        fprintf('    max radius %.3f mm vs paraxial stop semi-diam %.3f mm (EPD %g mm)\n', ...
                fpr*1e3, r_stop_par*1e3, G.EPD_m*1e3);

        % chief landing on FP vs SI origin -- THE recenter convention gate
        Nd = G.psi_si/norm(G.psi_si);  Vd = G.V_si;
        p1 = st.pos(:,1);  d1 = st.dir(:,1);
        cFP = p1 + d1*(dot(Nd, Vd - p1)/dot(Nd, d1));
        dSI = cFP - Vd;
        fprintf('  centre-field chief on FP vs SI origin: [%+.4f %+.4f %+.4f] mm  |.| = %.4f mm\n', ...
                dSI*1e3, norm(dSI)*1e3);
        fprintf('    (recenter parity gate: correct map ~ sub-mm; wrong map ~ 2x YDE ~ %.0f mm)\n', ...
                2*abs(getf_(R.s(8),'YDE',0)));
        fprintf('  chief AOI on FP: %.4f deg\n', ...
                acosd(min(1,abs(dot(Nd, d1)))));

        % ============ 2. STRICT WFE, HIS 9 FIELDS ========================
        nF = numel(R.XAN);
        w_cen = nan(nF,1);  w_chf = nan(nF,1);  nr = zeros(nF,1);
        miss  = nan(nF,1);
        for q = 1:nF
            dq = tancomp_(R.XAN(q), R.YAN(q));
            [pq, aimq] = aim_chief_(txt0, tmp, G, dq);
            sq = trace_full_(txt0, tmp, pq, dq);
            ok = sq.ok;  ok(1) = false;
            if nnz(ok) < 10, continue; end
            miss(q) = aimq.miss;
            % probe field for the exit pupil (aimed too): + on x-angle
            bx = asin(dq(1));  by = asin(dq(2));
            dp = [sin(bx+1e-5); sin(by); sqrt(1-sin(bx+1e-5)^2-sin(by)^2)];
            pp = aim_chief_(txt0, tmp, G, dp, pq);
            sp = trace_full_(txt0, tmp, pp, dp);
            X  = fex_cross_(sq.pos(:,1), sq.dir(:,1), sp.pos(:,1), sp.dir(:,1));
            rf = strict_refs(sq.pos(:,ok), sq.dir(:,ok), sq.opl(ok), ...
                             sq.pos(:,1), sq.dir(:,1), Vd, Nd, X);
            w_cen(q) = rf.wfe_centroid*1e9;
            w_chf(q) = rf.wfe_chief*1e9;
            nr(q) = nnz(ok);
            if ~opts.quiet
                fprintf('   [XAN %+5.1f  YAN %+5.1f]  strict RMS %10.3f nm (centroid) %10.3f (chief)  %4d rays\n', ...
                        R.XAN(q), R.YAN(q), w_cen(q), w_chf(q), nr(q));
            end
        end

        % ============ 2b. DENSE MAP -- the slide ceiling ==================
        % Each pptx slide shows a 20x20-deg FIELD MAP; "RMS WFE <= x" is
        % the ceiling of that MAP, not the max over the 9 .seq optimization
        % points (r2 proves it: 9-pt max 4022 nm vs slide 8810 -- the r2
        % design is "uncontrolled in used FOV" so its peak is off the
        % sample points).  Sample the full box on a map_n x map_n grid.
        wm_cen = [];  wm_chf = [];
        if opts.map_n > 0
            xg = linspace(-10, 10, opts.map_n);
            yg = linspace(R.YAN(2)-10, R.YAN(2)+10, opts.map_n);
            [XG, YG] = meshgrid(xg, yg);
            wm_cen = nan(size(XG));  wm_chf = nan(size(XG));
            p_prev = [];
            for q = 1:numel(XG)
                dq = tancomp_(XG(q), YG(q));
                if isempty(p_prev)
                    [pq, aq] = aim_chief_(txt0, tmp, G, dq);
                else
                    % warm start from the neighbouring field (2 deg away)
                    [pq, aq] = aim_chief_(txt0, tmp, G, dq, p_prev);
                end
                if aq.miss > 1e-6      % cold-restart fallback
                    [pq, aq] = aim_chief_(txt0, tmp, G, dq);
                end
                if aq.miss < 1e-6, p_prev = pq; else, p_prev = []; end
                sq = trace_full_(txt0, tmp, pq, dq);
                ok = sq.ok;  ok(1) = false;
                if nnz(ok) < 10, continue; end
                bx = asin(dq(1));  by = asin(dq(2));
                dp = [sin(bx+1e-5); sin(by); sqrt(1-sin(bx+1e-5)^2-sin(by)^2)];
                pp = aim_chief_(txt0, tmp, G, dp, pq);
                sp = trace_full_(txt0, tmp, pp, dp);
                X  = fex_cross_(sq.pos(:,1), sq.dir(:,1), sp.pos(:,1), sp.dir(:,1));
                rf = strict_refs(sq.pos(:,ok), sq.dir(:,ok), sq.opl(ok), ...
                                 sq.pos(:,1), sq.dir(:,1), Vd, Nd, X);
                wm_cen(q) = rf.wfe_centroid*1e9;
                wm_chf(q) = rf.wfe_chief*1e9;
            end
            [mx, im] = max(wm_cen(:));
            fprintf('\n  DENSE %dx%d map over the 20x20 deg box:\n', opts.map_n, opts.map_n);
            fprintf('    map max %10.3f nm (centroid) at XAN %+6.2f YAN %+6.2f;  chief-ref map max %10.3f nm\n', ...
                    mx, XG(im), YG(im), max(wm_chf(:)));
        end

        % ============ 3. VERDICT =========================================
        fin = isfinite(w_cen);
        if ~any(fin)
            st3 = @(w) nan(1,3);
        else
            st3 = @(w) [max(w(fin)), sqrt(mean(w(fin).^2)), mean(w(fin))];
        end
        Tc = st3(w_cen);  Th = st3(w_chf);
        g  = gates_nm(k);
        fprintf('\n  STRICT WFE over his 9 fields (nm @ 1000 nm), FPA = deck-verbatim .seq:\n');
        fprintf('    reference   %10s %10s %10s\n','max','field-RMS','mean');
        fprintf('    centroid    %10.3f %10.3f %10.3f\n', Tc);
        fprintf('    chief       %10.3f %10.3f %10.3f\n', Th);
        fprintf('    HIS number  %10.1f  ("RMS WFE <= ", ceiling of his 20x20 map)\n', g);
        fprintf('    ratios (centroid): 9pt-max/his %.4f   fRMS/his %.4f\n', Tc(1)/g, Tc(2)/g);

        % gate verdict: the BAND rule (rodgers1) -- the reproduction must
        % land at his scale, not be tuned to his digit.  His number is the
        % DENSE-MAP ceiling, so gate on the dense-map max (centroid
        % primary, chief reported); PASS if within [0.8, 1.25]x.
        if ~isempty(wm_cen)
            gate_stat = max(wm_cen(:));  gate_lbl = sprintf('dense %dx%d map max', opts.map_n, opts.map_n);
        else
            gate_stat = Tc(1);           gate_lbl = '9-pt max';
        end
        fprintf('    GATE statistic (%s): %.3f nm -> ratio %.4f\n', gate_lbl, gate_stat, gate_stat/g);
        pass = gate_stat/g > 0.8 && gate_stat/g < 1.25;
        fprintf('  GATE %g nm: %s\n', g, tern_(pass,'PASS','FAIL'));
        if ~pass, stopped = true; end

        OUT.(rk) = struct('ran',true, 'pass',pass, 'his_nm',g, ...
            'w_cen_nm',w_cen, 'w_chf_nm',w_chf, 'nrays',nr, ...
            'max_cen',Tc(1),'frms_cen',Tc(2),'mean_cen',Tc(3), ...
            'max_chf',Th(1),'frms_chf',Th(2),'mean_chf',Th(3), ...
            'map_cen',wm_cen, 'map_chf',wm_chf, 'map_n',opts.map_n, ...
            'gate_stat',gate_stat, 'gate_lbl',gate_lbl, ...
            'dSI_mm',dSI*1e3, 'fp_stop_mm',[fpc(1:2)*1e3; fpr*1e3; r_stop_par*1e3], ...
            'aim_miss',miss, 'deck',G.deck);
    end

    % ---- gate table ------------------------------------------------------
    banner('STAGE 0 GATE TABLE  (strict WFE, centroid ref, deck-verbatim FPA)');
    fprintf('  rung | his nm  | map max   | 9pt max   | 9pt fRMS  |  map/his | verdict\n');
    for k = 1:5
        rk = rungs{k};
        if ~isfield(OUT,rk) || ~OUT.(rk).ran
            fprintf('   %s  | %7.0f |     --    |     --    |     --    |    --    | NOT RUN\n', rk, gates_nm(k));
        else
            o = OUT.(rk);
            fprintf('   %s  | %7.0f | %9.2f | %9.2f | %9.2f | %8.3f | %s\n', rk, o.his_nm, ...
                    o.gate_stat, o.max_cen, o.frms_cen, o.gate_stat/o.his_nm, tern_(o.pass,'PASS','FAIL'));
        end
    end
    if opts.save
        save(fullfile(here, sprintf('r3_s0%s.mat',opts.suffix)), 'OUT');
        fprintf('\nsaved r3_s0%s.mat\n', opts.suffix);
    end
end

% =====================================================================
function d = tancomp_(xan_deg, yan_deg)
%TANCOMP_  CODE V XAN/YAN -> unit direction (tangent composition).
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end

function mag = stop_mag_(G)
%STOP_MAG_  Paraxial d(h_stop)/d(h_in) for an axis-parallel input ray
%   through m1 (mirror, vertex radius Kr signed toward +z centre).
%   Slope after reflection of a +z paraxial ray at height h: the mirror
%   adds -2h/R to the (reversed) ray; propagate |z_m1 - z_stop|.
    Rm = G.M(1).Kr;                    % signed, m (= CODE V R in m)
    Lz = abs(G.z_m1 - G.z_stop);
    mag = 1 + 2*Lz/Rm*sign(1);         % h_stop = h(1 + 2L/R) [convex R>0 grows]
end

function [p0, aim] = aim_chief_(txt0, tmp, G, cdir, seed)
%AIM_CHIEF_  Newton-iterate ChfRayPos so the traced chief crosses the
%   stop plane at the stop centre.  Post-m1 chief state extrapolated to
%   z = z_stop (no elements in between; exact).
    if nargin < 5
        cdR  = [cdir(1); cdir(2); -cdir(3)];
        tq   = (G.z_m1 - G.stopC(3))/cdir(3);
        q    = G.stopC - tq*cdR;
        seed = q - (0.75/cdir(3))*cdir;
    end
    p0 = seed;
    h = 1e-4;  tol = 1e-9;  aim = struct('niter',0,'miss',inf);
    r0 = stop_miss_(txt0, tmp, G, p0, cdir);
    if norm(r0) >= tol
        % one FD Jacobian, then fixed-J Newton (the map is near-affine)
        rx = stop_miss_(txt0, tmp, G, p0+[h;0;0], cdir);
        ry = stop_miss_(txt0, tmp, G, p0+[0;h;0], cdir);
        J  = [(rx-r0)/h, (ry-r0)/h];
        for it = 1:8
            dp = -J\r0;
            p0 = p0 + [dp(1); dp(2); 0];
            r0 = stop_miss_(txt0, tmp, G, p0, cdir);
            aim.niter = it;
            if norm(r0) < tol, break; end
        end
    end
    aim.miss = norm(r0);
end

function r = stop_miss_(txt0, tmp, G, p0, cdir)
    st = trace_elt_(txt0, tmp, p0, cdir, 1);
    p = st.pos(:,1);  d = st.dir(:,1);
    t = (G.stopC(3) - p(3))/d(3);
    q = p + d*t;
    r = q(1:2) - G.stopC(1:2);
end

function st = trace_elt_(txt0, tmp, p0, cdir, ie)
    emit_src_(txt0, tmp, p0, cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), error('rodgers3:load','deck failed to load: %s', tmp); end
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
    st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                'ok', ri.ok_trace(:) & ri.ok_pass(:));
end

function st = trace_full_(txt0, tmp, p0, cdir)
    emit_src_(txt0, tmp, p0, cdir);
    macos.load_rx(tmp);
    nE = macos.num_elt();
    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
    st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                'ok', ri.ok_trace(:) & ri.ok_pass(:));
end

function emit_src_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function v = getf_(s, f, d)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function s = tern_(c,a,b), if c, s=a; else, s=b; end, end

function delete_if_(p), if exist(p,'file'), delete(p); end, end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
