function cf_la_sweep(stage)
%CF_LA_SWEEP  Released-stroke la table for a ladder stage (Dave
%   2026-09-01, "can we release the stroke limit on 3e?"): for every
%   cached round Jacobian of <stage>, the linear-achievable floor at
%   stroke bounds 50 / 100 / 200 / 500 nm and UNBOUNDED (full rank),
%   with the stroke the unbounded floor needs.  The EFC dig itself is
%   never stroke-limited; only the la diagnostic carries the bound.
%   Appends the table to <stage>_report.txt.
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    lib = cf_efc_lib();
    d = dir(fullfile(here, sprintf('%s_G_*_r*.mat', stage)));
    assert(~isempty(d), 'cf_la_sweep: no %s G caches', stage);
    rn = cellfun(@(n) str2double(regexp(n,'_r(\d+)\.mat$','tokens','once')), ...
                 {d.name});
    [rn, ix] = sort(rn);  d = d(ix);
    rep = fullfile(here, sprintf('%s_report.txt', stage));
    L = {};
    L{end+1} = sprintf('---- %s la sweep, stroke bound released (%s) ----', ...
                       stage, datestr(now,31)); %#ok<DATST>
    L{end+1} = sprintf('%5s | %10s %10s %10s %10s | %12s @ %8s (rank)', ...
        'round','la@50nm','la@100','la@200','la@500','la@UNBOUND','stroke');
    for k = 1:numel(d)
        J = load(fullfile(here, d(k).name));
        la = lib.linfloor(J, 50);           % carries the full curves
        c = la.curve_con;  st = la.curve_stroke_nm;
        pick = @(b) c(max([1, find(st <= b, 1, 'last')]));
        L{end+1} = sprintf('%5d | %10.3e %10.3e %10.3e %10.3e | %12.3e @ %6.0f nm (%d)', ...
            rn(k), pick(50), pick(100), pick(200), pick(500), ...
            c(end), st(end), numel(c)); %#ok<AGROW>
    end
    txt = strjoin(L, newline);
    fprintf('%s\n', txt);
    fid = fopen(rep, 'a');  fprintf(fid, '%s\n', txt);  fclose(fid);
end
