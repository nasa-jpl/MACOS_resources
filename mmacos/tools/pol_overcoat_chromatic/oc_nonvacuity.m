function out = oc_nonvacuity(model)
%OC_NONVACUITY  Show the reversal gate FAILS when the reversal is absent.
%
%   The gate
%   tPolContrast/test_overcoat_trade_reverses_across_the_quarter_wave_condition
%   asserts that the 110 nm MgF2 overcoat SUPPRESSES cross-polarized power
%   at 632.8 nm and COSTS it at 1 um.  A gate that only ever sees the real
%   engine cannot show it would notice the reversal going away.
%
%   The counterfactual is specific and physical: an engine that treated
%   coating thickness as fixed in WAVES rather than in metres -- i.e. that
%   evaluated the "632.8 nm" coating constants achromatically -- would, at
%   632.8 nm, be tracing 110 nm x (632.8/1000) = 69.6 nm of MgF2.  That
%   film has the SAME optical thickness in waves at 632.8 nm that the real
%   film has at 1 um, so it produces the 1 um answer at the companion
%   wavelength and the reversal disappears.
%
%   This routine runs the gate's own 632.8 nm assertions against that
%   counterfactual and reports which ones fail.  It does not need a
%   modified engine: the counterfactual is reachable by asking the real
%   engine for the film the achromatic treatment would have implied.
%
%   Usage:
%     cd MACOS_resources/mmacos
%     matlab -batch "mmacos_setup; addpath('tools/pol_overcoat_chromatic'); oc_nonvacuity; exit(0)"
%
%   See also: oc_ladder, tPolContrast.

if nargin < 1, model = 256; end

r = oc_ladder(model, false);

% The gate's 632.8 nm assertions, applied to the achromatic counterfactual.
chk = { ...
  'ratio_mgf2 < 1 (the film must SUPPRESS at 632.8 nm)', ...
      r.achromatic.ratio_mgf2 < 1, ...
      r.achromatic.ratio_mgf2, 1; ...
  'ratio_mgf2 == 0.20351 within 2%', ...
      abs(r.achromatic.ratio_mgf2/0.20351 - 1) <= 0.02, ...
      r.achromatic.ratio_mgf2, 0.20351; ...
  'reversal == 25.899 within 2%', ...
      abs((r.at1000.ratio_mgf2/r.achromatic.ratio_mgf2)/25.899 - 1) <= 0.02, ...
      r.at1000.ratio_mgf2/r.achromatic.ratio_mgf2, 25.899};

fprintf('\n=== non-vacuity: the gate against an ACHROMATIC counterfactual ===\n');
fprintf('real film at 632.8 nm : %.4f nm of MgF2, ratio %.5f\n', ...
        r.at633.thk_mgf2*1e9, r.at633.ratio_mgf2);
fprintf('achromatic film       : %.4f nm of MgF2, ratio %.5f\n\n', ...
        r.achromatic.thk_mgf2*1e9, r.achromatic.ratio_mgf2);
nfail = 0;
for i = 1:size(chk, 1)
    ok = chk{i,2};
    nfail = nfail + ~ok;
    if ok, tag = 'PASS'; else, tag = 'FAIL'; end
    fprintf('  [%s] %-46s  got %.5f, wanted %.5f\n', ...
            tag, chk{i,1}, chk{i,3}, chk{i,4});
end
fprintf('\n%d of %d gate assertions FAIL against the counterfactual', ...
        nfail, size(chk, 1));
if nfail == size(chk, 1)
    fprintf(' -- the gate is non-vacuous.\n\n');
else
    fprintf(' -- SOME ASSERTION IS VACUOUS.\n\n');
end

out = struct('n_fail', nfail, 'n_total', size(chk, 1), ...
             'thk_real', r.at633.thk_mgf2, 'thk_achrom', r.achromatic.thk_mgf2, ...
             'ratio_real', r.at633.ratio_mgf2, ...
             'ratio_achrom', r.achromatic.ratio_mgf2);
end
