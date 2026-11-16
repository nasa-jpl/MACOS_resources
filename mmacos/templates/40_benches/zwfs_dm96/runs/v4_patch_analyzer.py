"""Add the vector sensor's ANALYZER leak (cube + quarter-wave plate) to the gauge, params and runner.
Usage: python3 patch_analyzer.py <dir holding dm_gauge_lib/ and zwfs_dm96/>"""
import sys, re
root = sys.argv[1]
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    open(path, 'w').write(s)
    print('patched', path)

# ---------------- dmg_zwfs_gauge.m ----------------
g = root + '/dm_gauge_lib/dmg_zwfs_gauge.m'
patch(g, [
# 1. parse the option
("""varm = 'none';  vlaser = 45;  vdph = 0;  vdam = 0;
""",
"""varm = 'none';  vlaser = 45;  vdph = 0;  vdam = 0;
% V4 (2026-09-14): the ANALYZER (quarter-wave plate + polarizing cube) mixes
% the two masked images: camera A (Ip) sees P_main*(|a+|^2 + lA |a-|^2 +
% 2 Re(a+ conj(a-) cA)), camera B (Im) the same with (lB, cB) and the roles
% swapped -- l the incoherent leak (the cube's finite extinction), c the
% coherent one (a plate retardance error delta gives |c| = delta/2, an
% azimuth error theta gives |c| = theta; opposite signs in the two ports).
% V_ANALYZER: a struct with lA, cA, lB, cB (dmg_analyzer_maps: the engine's
% polarized traces of the two channel decks), or 'none'.  The solver knows
% nothing of it: the error it produces is the price of an uncalibrated
% analyzer.
ana = struct('mode', 'none', 'lA', 0, 'cA', 0, 'lB', 0, 'cB', 0);
if isfield(opt, 'V_ANALYZER') && isstruct(opt.V_ANALYZER)
    ana = struct('mode', 'maps', 'lA', opt.V_ANALYZER.lA, 'cA', opt.V_ANALYZER.cA, ...
                 'lB', opt.V_ANALYZER.lB, 'cB', opt.V_ANALYZER.cB);
end
"""),
# 2. store + handles
("""ZW.Vm = Vm;  ZW.ccm = ccm;  ZW.leak = leak;  ZW.arm = arm;
""",
"""ZW.Vm = Vm;  ZW.ccm = ccm;  ZW.leak = leak;  ZW.arm = arm;  ZW.ana = ana;
"""),
("""ZW.frameV   = @(M) frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm);  % -> [Ip, Im]
""",
"""ZW.frameV   = @(M) frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm, ana);  % -> [Ip, Im]
"""),
("""        [Ipf, Imf] = frameV_(zeros(macos.get_elt_grid_size(iTO)), iTO, iMASK, iDET, V, Vm, N_WF, leak, arm);
""",
"""        [Ipf, Imf] = frameV_(zeros(macos.get_elt_grid_size(iTO)), iTO, iMASK, iDET, V, Vm, N_WF, leak, arm, ana);
"""),
("""ZW.measV    = @(M) measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm);
""",
"""ZW.measV    = @(M) measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm, ana);
"""),
# 3. frameV_: fields then mix
("""function [Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm) %#ok<INUSD>
""",
"""function [Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm, ana) %#ok<INUSD>
"""),
("""if nargin < 9 || isempty(arm), arm = struct('mode', 'none', 'qL', [], 'qR', []); end
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
""",
"""if nargin < 9 || isempty(arm), arm = struct('mode', 'none', 'qL', [], 'qR', []); end
if nargin < 10 || isempty(ana), ana = struct('mode', 'none'); end
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
"""),
("""if strcmp(arm.mode, 'none')
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, V);
    Ip = abs(se*macos.complex_field(iDET, 'reset_trace', false) + lk*E0s).^2;
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, Vm);
    Im = abs(se*macos.complex_field(iDET, 'reset_trace', false) + lk*E0s).^2;
else
    Ip = abs(se*chain_(M, arm.qL, V,  iTO, iMASK, iDET) + lk*arm.qR.*E0s).^2;
    Im = abs(se*chain_(M, arm.qR, Vm, iTO, iMASK, iDET) + lk*arm.qL.*E0s).^2;
end
end
""",
"""if strcmp(arm.mode, 'none')
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, V);
    EA = se*macos.complex_field(iDET, 'reset_trace', false) + lk*E0s;    % camera A's field: the +phi image
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, Vm);
    EB = se*macos.complex_field(iDET, 'reset_trace', false) + lk*E0s;    % camera B's: the -phi image
else
    EA = se*chain_(M, arm.qL, V,  iTO, iMASK, iDET) + lk*arm.qR.*E0s;
    EB = se*chain_(M, arm.qR, Vm, iTO, iMASK, iDET) + lk*arm.qL.*E0s;
end
if strcmp(ana.mode, 'none')
    Ip = abs(EA).^2;  Im = abs(EB).^2;
else   % V4: the analyzer mixes the two images (dmg_analyzer_maps' model, main-channel gain removed)
    Ip = abs(EA).^2 + ana.lA*abs(EB).^2 + 2*real(EA.*conj(EB)*ana.cA);
    Im = abs(EB).^2 + ana.lB*abs(EA).^2 + 2*real(EB.*conj(EA)*ana.cB);
end
end
"""),
("""function h = measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm)
[Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm);
""",
"""function h = measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm, ana)
[Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm, ana);
"""),
])

# ---------------- zwfs_params.m ----------------
p = root + '/zwfs_dm96/zwfs_params.m'
s = open(p).read()
m = re.search(r"^(P\.mask\.v_ar_n\s*=.*\n)", s, re.M)
assert m, 'v_ar_n line'
ins = m.group(1) + """P.mask.v_analyzer = 'none';  % V4: the analyzer's leak between the two images: 'none' (ideal cube and
                             % quarter-wave plate) | 'engine' (dmg_analyzer_maps: the engine's polarized
                             % traces of the two channel decks -- the MacNeille cube's extinction and the
                             % plate's errors below) | a struct with lA, cA, lB, cB
P.mask.v_qwp_err = 0;        % the plate's retardance error, waves (a zero-order plate: 1/300 typical spec)
P.mask.v_qwp_az  = 0;        % the plate's fast-axis azimuth error, degrees
"""
s = s.replace(m.group(1), ins, 1)
open(p, 'w').write(s); print('patched', p)

# ---------------- zwfs_run.m ----------------
r = root + '/zwfs_dm96/zwfs_run.m'
patch(r, [
("""    'V_ARM_DPHASE', P.mask.v_arm_dphase, 'V_ARM_DAMP', P.mask.v_arm_damp);
end
""",
"""    'V_ARM_DPHASE', P.mask.v_arm_dphase, 'V_ARM_DAMP', P.mask.v_arm_damp);
if isstruct(P.mask.v_analyzer), g.V_ANALYZER = P.mask.v_analyzer; end   % V4 (resolved from 'engine' in the bench stage)
end
"""),
# bench stage: resolve 'engine' before the gauge is built -- anchor on the AR-coat block
("""if P.mask.v_arm_ar
""",
"""if ischar(P.mask.v_analyzer) && strcmp(P.mask.v_analyzer, 'engine')     % V4: the analyzer's leak from the engine
    ana = dmg_analyzer_maps(P, 'qwp_err', P.mask.v_qwp_err, 'qwp_az', P.mask.v_qwp_az, 'NGRID', P.NGRID, 'MODEL', P.MODEL, 'deck_dir', pwd);   % the run's own size: no model-size transition in this process
    P.mask.v_analyzer = ana;  analyzer = ana;
    macos.load_rx(deck);  coat_oap_(P, G.bt, rep);                                   % back to the record deck
else
    analyzer = [];
end
if P.mask.v_arm_ar
"""),
# print after the arm block, before the G4 pokes: anchor on the Afig line
("""    Afig = zeros(cfg.nact);  Afig(4:8:end, 4:8:end) = P.mask.v_gate_nm*1e-6;
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Afig));
    Et = macos.complex_field(iDET);
""",
"""    if ~strcmp(ZW.ana.mode, 'none')
        if ~isempty(analyzer)
            sa = analyzer.stats;
            dmg_say(rep, 'V4 analyzer (engine: quarter-wave plate retardance error %.4f waves, azimuth error %.2f deg, MacNeille cube; cone at the pupil image %.2f deg): camera A main %s %.4f, incoherent leak %.2e (rms %.1e, max %.1e), coherent %.2e at %+.2f rad; camera B main %s %.4f, incoherent %.2e (rms %.1e, max %.1e), coherent %.2e at %+.2f rad; split A/B %.4f\\n', ...
                analyzer.qwp_err, analyzer.qwp_az, analyzer.cone_deg, ...
                sa.A.main, sa.A.Pmain, sa.A.leak_mean, sa.A.leak_rms, sa.A.leak_max, abs(ZW.ana.cA), angle(ZW.ana.cA), ...
                sa.B.main, sa.B.Pmain, sa.B.leak_mean, sa.B.leak_rms, sa.B.leak_max, abs(ZW.ana.cB), angle(ZW.ana.cB), analyzer.split);
        else
            dmg_say(rep, 'V4 analyzer (given): lA %.2e, cA %.2e at %+.2f rad; lB %.2e, cB %.2e at %+.2f rad\\n', ZW.ana.lA, abs(ZW.ana.cA), angle(ZW.ana.cA), ZW.ana.lB, abs(ZW.ana.cB), angle(ZW.ana.cB));
        end
        dmg_say(rep, '  the solver knows nothing of the analyzer: the V error below is its price\\n');
    end
    Afig = zeros(cfg.nact);  Afig(4:8:end, 4:8:end) = P.mask.v_gate_nm*1e-6;
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Afig));
    Et = macos.complex_field(iDET);
"""),
("""S.summary.maskfig = maskfig;                     % the focal spot + mask windows for <tag>_mask.png
""",
"""S.summary.maskfig = maskfig;                     % the focal spot + mask windows for <tag>_mask.png
S.summary.analyzer = analyzer;                   % V4: the analyzer's leak maps and stats ([] when 'none')
"""),
("""    priced = (ZW.leak.eta < 1 && strcmp(P.mask.v_cal, 'ideal')) || ...
             (~strcmp(ZW.arm.mode, 'none') && ~strcmp(P.mask.v_cal, 'map'));
""",
"""    priced = (ZW.leak.eta < 1 && strcmp(P.mask.v_cal, 'ideal')) || ...
             (~strcmp(ZW.arm.mode, 'none') && ~strcmp(P.mask.v_cal, 'map')) || ...
             ~strcmp(ZW.ana.mode, 'none');
"""),
("""        dmg_say(rep, '  (G4 not asserted: an uncalibrated metasurface / arm error is being priced -- the V error above IS the number)\\n');
""",
"""        dmg_say(rep, '  (G4 not asserted: an uncalibrated metasurface / arm / analyzer error is being priced -- the V error above IS the number)\\n');
"""),
])
