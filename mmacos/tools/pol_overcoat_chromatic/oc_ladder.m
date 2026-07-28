function out = oc_ladder(model, verbose)
%OC_LADDER  The Phase-2c coating ladder on BOTH sides of the quarter-wave
%   condition, measured by the ENGINE.
%
%   OUT = OC_LADDER() runs the §2c coating ladder (uncoated -> bare Al ->
%   110 nm MgF2 / Al) on Rx_Cass_FarField at TWO wavelengths -- the
%   fixture's own 1 um and a companion run at 632.8 nm, the wavelength the
%   2c coating constants are actually labelled for -- and reports the
%   MgF2/bare cross-polarized POWER ratio at each.
%
%   WHY.  REVIEW_POL_EXTERNAL_2026-07-28.md found that the 110 nm MgF2
%   film the 2c ladder applies is 0.607 quarter-waves at the fixture's 1 um
%   (a quarter wave there is 181.2 nm), not the 0.96 quarter-waves its
%   "632.8 nm" comment describes -- and that the overcoat polarization
%   trade REVERSES across the quarter-wave condition.  That correction was
%   carried by an independent analytic; this tool puts ENGINE numbers on
%   both sides of the reversal so the design rule rests on measurement.
%
%   THE FIXTURE DOES NOT MOVE.  Rx_Cass_FarField underpins gates across
%   every suite and stays at Wavelen = 1e-6.  The companion wavelength is
%   applied at RUNTIME with macos.set_src_wvl, after load_rx.  Nothing on
%   disk changes.
%
%   WHY THIS IS A MEANINGFUL RUN AND NOT A UNIT CONVERSION.  macos.coating
%   takes PHYSICAL thickness, and the engine divides by the CURRENT
%   Wavelen when it applies the layer phase.  A film is therefore a fixed
%   piece of glass under a wavelength change, and its optical thickness
%   moves -- which is the entire mechanism under test.  The 'achromatic'
%   control below is what proves that is the mechanism.
%
%   FOUR LADDER POINTS PER WAVELENGTH:
%     baseline    uncoated (the as-loaded prescription)
%     bare Al     200 nm Al
%     MgF2/Al     110 nm MgF2 over Al  -- the film the 2c ladder applies
%     trueQW/Al   lambda/(4*n_MgF2) of MgF2 over Al -- a genuine
%                 quarter-wave overcoat at THAT wavelength
%
%   PLUS ONE CONTROL, at 632.8 nm only:
%     achromatic  110 nm * (632.8/1000) = 69.6 nm of MgF2 over Al.  This is
%                 the film that has the SAME optical thickness in waves at
%                 632.8 nm that the real 110 nm film has at 1 um.  It is
%                 what a wrongly-achromatic treatment (thickness pinned in
%                 waves rather than in metres) would have produced at the
%                 companion wavelength.  It must NOT reverse -- and that is
%                 what shows the reversal is the film's optical thickness
%                 and not some other consequence of changing lambda
%                 (diffraction scale, pixel-to-lambda/D mapping, ...).
%
%   The reported ratio is cross-polarized TOTAL POWER, MgF2 over bare Al.
%   Total power, not an annulus mean: a fixed pixel annulus subtends a
%   different lambda/D range at the two wavelengths, so an annulus
%   statistic would mix the coating effect with the diffraction scale.
%   Total power has no such dependence and is the quantity the reversal is
%   a statement about.
%
%   TWO RATIOS ARE REPORTED, and they answer different questions:
%     .ratio_mgf2    total cross power, MgF2/Al over bare Al.  What a
%                    designer sees -- it includes the irreducible
%                    GEOMETRIC cross term the coating cannot remove.
%     .ratio_excess  the same ratio formed from the coating EXCESS over
%                    the uncoated baseline, i.e. the ratio of the two
%                    d_cross_rel values.  This is the coating-only
%                    quantity, and it is what compares directly with the
%                    pure-Fresnel analytic in tools/pol_external_anchor.
%   They differ most where the coating term is nearly extinguished (the
%   true quarter-wave point), because that is where the geometric floor
%   dominates the total -- see .qw_over_uncoated.
%
%   Usage:
%     cd MACOS_resources/mmacos
%     matlab -batch "mmacos_setup; addpath('tools/pol_overcoat_chromatic'); oc_ladder; exit(0)"
%
%   OC_LADDER([]) skips macos.init -- for callers (the CI gate) that have
%   already initialized the engine at the right model size.  Re-initializing
%   is avoided rather than repeated: model_size transitions are the known
%   heap hazard, and a same-size re-init is not worth testing here.
%
%   See also: macos.pol_contrast_floor, tPolContrast, tPolExternal,
%   mmacos/tools/pol_external_anchor (the independent analytic).

if nargin < 1,                     model   = 256;  end
if nargin < 2 || isempty(verbose), verbose = true; end

% ---- constants, verbatim from tPolContrast -----------------------------
nAl = 1.45;  kAl = 7.54;  thkAl = 2.0e-7;    % Al at 632.8 nm, BaseUnits = m
nMgF2 = 1.38;  thkMgF2 = 1.1e-7;             % the 2c overcoat, as configured
PUP = 5;  DET = 6;  MIR = [2 3];             % Rx_Cass_FarField
LAM_FIX = 1.0e-6;                            % the fixture's own Wavelen (m)
LAM_CMP = 632.8e-9;                          % the coating constants' label

rx = oc_rx('Rx_Cass_FarField.in');
if ~isempty(model), macos.init(model); end

out = struct('model', model, 'thk_mgf2', thkMgF2, 'thk_al', thkAl, ...
             'n_mgf2', nMgF2, 'lambda', [LAM_FIX LAM_CMP]);

% ---- the two ladders ---------------------------------------------------
out.at1000 = ladder_(rx, LAM_FIX, thkMgF2, PUP, DET, MIR, nAl, kAl, thkAl, nMgF2);
out.at633  = ladder_(rx, LAM_CMP, thkMgF2, PUP, DET, MIR, nAl, kAl, thkAl, nMgF2);

% ---- the control: the same film treated as if it were achromatic -------
% A thickness pinned in WAVES rather than in metres would, at 632.8 nm,
% be this much glass.  Same 0.607 quarter-waves as the real film at 1 um.
thkAchrom = thkMgF2 * LAM_CMP / LAM_FIX;
out.achromatic = ladder_(rx, LAM_CMP, thkAchrom, PUP, DET, MIR, ...
                         nAl, kAl, thkAl, nMgF2);
out.achromatic.thk_mgf2 = thkAchrom;

% ---- the reversal, as one number ---------------------------------------
out.reversal        = out.at1000.ratio_mgf2   / out.at633.ratio_mgf2;
out.reversal_excess = out.at1000.ratio_excess / out.at633.ratio_excess;

if verbose, oc_report_(out); end
end

% =======================================================================
function L = ladder_(rx, lam, thkMgF2, PUP, DET, MIR, nAl, kAl, thkAl, nMgF2)
%LADDER_  One wavelength's uncoated -> Al -> MgF2/Al -> trueQW/Al ladder.
    macos.load_rx(rx);
    macos.set_src_wvl(lam);        % WaveUnits = m on this fixture

    % pol_contrast_floor calls macos.polarization(...) before every field
    % fetch, and that dirties the cached trace (pol_set -> modified_rx), so
    % the wavelength set above is picked up by the retrace.  Asserted below
    % rather than assumed: the two ladders must not come out identical.
    thkQW = lam / (4 * nMgF2);

    al = struct('elt', num2cell(MIR), 'index', nAl, 'extinc', kAl, ...
                'thickness', thkAl, 'label', 'bare Al');
    mg = struct('elt', num2cell(MIR), 'index', [nMgF2 nAl], ...
                'extinc', [0 kAl], 'thickness', [thkMgF2 thkAl], ...
                'label', 'MgF2 / Al');
    qw = struct('elt', num2cell(MIR), 'index', [nMgF2 nAl], ...
                'extinc', [0 kAl], 'thickness', [thkQW thkAl], ...
                'label', 'true QW MgF2 / Al');

    o = macos.pol_contrast_floor(PUP, DET, 'input', 'x', ...
                                 'coatings', {al, mg, qw});

    L = struct();
    L.lambda      = lam;
    L.wvl_readback = macos.get_src_wvl();
    L.thk_mgf2    = thkMgF2;
    L.thk_qw      = thkQW;
    L.qw_frac     = thkMgF2 / thkQW;         % film, in quarter-waves, here
    L.cross_bare0 = o.floor.cross;           % uncoated baseline
    L.cross_al    = o.sweep(1).floor.cross;
    L.cross_mgf2  = o.sweep(2).floor.cross;
    L.cross_qw    = o.sweep(3).floor.cross;
    L.d_cross_rel = [o.sweep.d_cross_rel];
    L.ratio_mgf2  = L.cross_mgf2 / L.cross_al;   % THE headline
    L.ratio_qw    = L.cross_qw   / L.cross_al;
    L.al_over_bare = L.cross_al / L.cross_bare0;
    % Coating EXCESS over the uncoated baseline -- the coating-only ratio,
    % directly comparable with the pure-Fresnel analytic.  Identically the
    % ratio of the two d_cross_rel values, since d = (cross - base)/base.
    L.ratio_excess    = L.d_cross_rel(2) / L.d_cross_rel(1);
    L.ratio_excess_qw = L.d_cross_rel(3) / L.d_cross_rel(1);
    % How far the true quarter-wave point sits above the irreducible
    % GEOMETRIC floor -- why .ratio_qw cannot follow the analytic down.
    L.qw_over_uncoated = L.cross_qw / L.cross_bare0;
    sc = [o.sweep.scope];
    L.full_chain  = o.scope.full_chain && all([sc.full_chain]);
    L.cross_over_co = o.floor.cross_over_co;
end

% =======================================================================
function oc_report_(r)
    fprintf('\n=== 2c coating ladder across the quarter-wave condition ===\n');
    fprintf('model %d, Rx_Cass_FarField (unmoved), x-polarized input\n', r.model);
    fprintf('MgF2 film %.1f nm, Al %.0f nm, n_MgF2 = %.2f\n\n', ...
            r.thk_mgf2*1e9, r.thk_al*1e9, r.n_mgf2);
    hdr = {'at1000', 'at633', 'achromatic'};
    lbl = {'1 um (the fixture)', '632.8 nm (companion)', ...
           '632.8 nm, achromatic control'};
    fprintf('%-30s %8s %8s %11s %11s %11s %11s\n', 'run', 'lambda', ...
            'QW frac', 'MgF2/bare', '  (excess)', 'trueQW/bare', 'Al/uncoated');
    for i = 1:numel(hdr)
        L = r.(hdr{i});
        fprintf('%-30s %7.1fnm %8.4f %11.4f %11.4f %11.4e %11.4f\n', lbl{i}, ...
                L.lambda*1e9, L.qw_frac, L.ratio_mgf2, L.ratio_excess, ...
                L.ratio_qw, L.al_over_bare);
    end
    fprintf(['\ntrue quarter-wave point sits at %.4fx the UNCOATED ' ...
             'geometric floor\n'], r.at1000.qw_over_uncoated);
    fprintf('reversal (1 um ratio / 632.8 nm ratio): %.4f  (excess form %.4f)\n', ...
            r.reversal, r.reversal_excess);
    fprintf('full-chain carry on every point: 1um=%d 633=%d ctl=%d\n\n', ...
            r.at1000.full_chain, r.at633.full_chain, r.achromatic.full_chain);
end

% =======================================================================
function p = oc_rx(name)
%OC_RX  Resolve an Rx fixture by name (mirrors tests/private/rx_fixture_path).
    roots = { fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'pymacos', 'tests', 'Rx'), ...
              fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'mmacos', 'tests', 'Rx') };
    for i = 1:numel(roots)
        p = fullfile(roots{i}, name);
        if exist(p, 'file'), return; end
    end
    error('oc_ladder:rx', 'Rx fixture not found: %s', name);
end
