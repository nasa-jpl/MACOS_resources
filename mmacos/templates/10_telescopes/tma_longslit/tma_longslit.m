function OUT = tma_longslit(over)
%TMA_LONGSLIT  The one-call demo: the long-slit TMA ladder at the default parameters.
%
%   OUT = TMA_LONGSLIT() runs first_order -> section -> figure -> e2e through
%   TMA_LONGSLIT_RUN at the default TMA_LONGSLIT_PARAMS (the dyson5 3k front
%   end at Joe's spec, seeded from the SBG VSWIR zig-zag) and prints where the
%   records are.  OUT = TMA_LONGSLIT(OVER) on top of TMA_LONGSLIT_PARAMS(OVER).
%
%   The figure ladder is engine-traced: about 25 min a rung on one core at
%   model 256.  Run the stages one at a time with TMA_LONGSLIT_RUN to resume.
%
%   See also TMA_LONGSLIT_RUN, TMA_LONGSLIT_PARAMS.
if nargin < 1, over = struct(); end
OUT = tma_longslit_run({'first_order', 'section', 'figure', 'e2e'}, over);
P = OUT.P;
fprintf('\ntma_longslit: records in %s\n', P.outdir);
for s = {'first_order', 'section', 'figure', 'e2e'}
    fprintf('  %s_%s.txt / .mat\n', P.tag, s{1});
end
r = OUT.figure.rungs(end);
fprintf('  last rung %s: deck %s; worst FWHM %.2f x %.2f px, worst chief %.3f deg, cone F/%.3f..%.3f, clearance %+.1f mm\n', ...
        r.name, r.deck, max(r.M.fwhm_x_px), max(r.M.fwhm_y_px), max(r.M.chief_deg), min([r.M.fno_x r.M.fno_y]), ...
        max([r.M.fno_x r.M.fno_y]), r.M.clear.min_mm);
E = OUT.e2e.E.rows(1).S.e2e;
fprintf('  end to end (rung %s, roll %d): smile %.3f / keystone %.3f / CRF %.2f / SRF %.2f px, admits %.3f\n', OUT.e2e.rung_name, ...
        OUT.e2e.E.rows(1).roll_deg, E.smile_max, E.keystone_max, E.crf_max, E.srf_max, min(E.pass_frac(:)));
end
