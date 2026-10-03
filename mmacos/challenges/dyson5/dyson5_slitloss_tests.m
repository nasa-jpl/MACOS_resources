function T = dyson5_slitloss_tests(P, tag)
%DYSON5_SLITLOSS_TESTS  The slit-loss factor: three one-knob tests, as a table first.
%   T = dyson5_slitloss_tests(P, tag) runs spectrometer_slit_loss (36 um slit,
%   far-field sandwich to the grating plane at 0.7 m, F/1.8 acceptance) at 380
%   and 2500 nm under BRIEF_to_dyson5 addendum 9's three knobs, one at a time,
%   and writes <tag>_s2l_tests.txt: (1) GRID 255 -> 511 -> 1023 points at the
%   base slit length (the pitch halves and the window doubles each step);
%   (2) WINDOW acceptance-to-window 1.11 -> 2 -> 4 at 255 points (by the
%   modelled slit length: shorter slit, finer input pitch, wider window);
%   (3) SLIT LENGTH 0.15 -> 2 -> 5 mm at 1023 points, scored with the |y|
%   STRIP and with the F/1.8 CIRCLE.  Every row prints the engine loss, the
%   sinc^2 closed form (the exact integral over the acceptance, NOT the
%   1/(pi^2 u0) asymptote), their ratio, the pitch, the window ratio, and the
%   fraction of energy in the outer 5 % of the window (the aliasing tell).
%   Model 1024 throughout (one MATLAB, no size transition).
    fid = fopen([tag '_s2l_tests.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 slit-loss tests (%s) -- 36 um slit, z = 0.7 m, F/%.1f acceptance; model 1024; loss = 1 - energy inside the acceptance\n', datestr(now, 'yyyy-mm-dd HH:MM'), P.Fno);
    pr('closed form = exact integral of sinc^2(w sin(theta)/lambda) over |sin(theta)| <= 1/(2F), relative to |sin(theta)| <= 1\n');
    pr('%-28s %5s %8s %10s %10s %7s %9s %8s %8s %8s %8s\n', 'test', 'nm', 'npts', 'engine', 'sinc^2', 'ratio', 'pitch um', 'win/acc', 'edge5%', 'evan.', 'L mm');
    rows = {};
    first = true;
    function run_(name, ng, slen, acc, propg)
        if nargin < 5, propg = false; end
        R = spectrometer_slit_loss(P, [tag '_s2l_tests_tmp.in'], 'lams', [380e-9 2500e-9], 'model', 1024, 'ngridpts', ng, ...
                                   'slit_len', slen, 'z_grating', 0.7, 'acceptance', acc, 'init', first, 'propagating', propg);
        first = false;
        for j = 1:2
            flag = '';  if R.window_ratio(j) < 1, flag = '  INVALID (window < acceptance)'; end
            pr('%-28s %5.0f %8d %10.5f %10.5f %7.3f %9.2f %8.2f %8.4f %8.4f %8.2f%s\n', name, R.lams(j)*1e9, ng, R.loss_engine(j), R.loss_sinc(j), ...
               R.loss_engine(j)/R.loss_sinc(j), R.dx_m(j)*1e6, R.window_ratio(j), R.edge_frac(j), R.evanescent_frac(j), slen*1e3, flag);
            rows(end+1, :) = {name, R.lams(j), ng, R.loss_engine(j), R.loss_sinc(j), R.dx_m(j), R.window_ratio(j), R.edge_frac(j), R.evanescent_frac(j), slen, propg};  %#ok<AGROW>
        end
    end
    pr('-- (1) grid, slit length 0.15 mm, strip acceptance\n');
    run_('grid 255', 255, 0.15e-3, 'strip');  run_('grid 511', 511, 0.15e-3, 'strip');  run_('grid 1023', 1023, 0.15e-3, 'strip');
    pr('-- (2) window, 255 points, strip acceptance (window by the modelled slit length)\n');
    run_('window x1.11 (0.15 mm)', 255, 0.15e-3, 'strip');  run_('window x2 (0.083 mm)', 255, 0.0832e-3, 'strip');  run_('window x4 (0.042 mm)', 255, 0.0416e-3, 'strip');
    pr('-- (3) slit length, 1023 points, strip vs circle acceptance\n');
    run_('len 0.15 mm strip', 1023, 0.15e-3, 'strip');   run_('len 0.15 mm circle', 1023, 0.15e-3, 'circle');
    run_('len 2 mm strip', 1023, 2e-3, 'strip');         run_('len 2 mm circle', 1023, 2e-3, 'circle');
    run_('len 5 mm strip', 1023, 5e-3, 'strip');         run_('len 5 mm circle', 1023, 5e-3, 'circle');
    pr('-- (4) normalised to the PROPAGATING region |sin(theta)| <= 1 only (|y|,|x| <= z); strip acceptance\n');
    run_('prop, 255 pts, 0.15 mm', 255, 0.15e-3, 'strip', true);  run_('prop, 255 pts, 0.083 mm', 255, 0.0832e-3, 'strip', true);
    run_('prop, 255 pts, 0.042 mm', 255, 0.0416e-3, 'strip', true);  run_('prop, 1023 pts, 0.15 mm', 1023, 0.15e-3, 'strip', true);
    fclose(fid);
    T = cell2table(rows, 'VariableNames', {'test', 'lambda', 'npts', 'engine', 'sinc2', 'pitch_m', 'window_ratio', 'edge5', 'evanescent', 'slit_len_m', 'propagating'});
    save([tag '_s2l_tests.mat'], 'T', 'P');
end

function dualprint_(fid, varargin)
    fprintf(1, varargin{:});  fprintf(fid, varargin{:});
end
