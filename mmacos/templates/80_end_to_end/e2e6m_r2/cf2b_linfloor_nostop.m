% cf2b_linfloor_nostop -- post-hoc linear-achievable floors for the four
% preserved NO-STOP family Jacobians (the comparison record): was each
% family's floor the controller or the substrate?
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'..','..','..','mmacos_setup.m'));
addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
lib = cf_efc_lib();
for k = {'hard','apl','aplc','blc'}
    J0 = load(sprintf('cf2_G_%s.mat', k{1}));
    J1 = load(sprintf('cf2_G_%s_r1.mat', k{1}));
    S  = load(sprintf('cf2_nostop_%s_run.mat', k{1}));
    la1 = lib.linfloor(J1, 50);
    av = cellfun(@(x) x(x~=0), S.res.a, 'UniformOutput', false);
    ach = 1e9 * rms(vertcat(av{:}));
    ok = la1.curve_stroke_nm <= ach;
    if ~any(ok), rk = 1; else, rk = find(ok, 1, 'last'); end
    fprintf(['LINF %-6s relin %.3e | lin-ach AT ACHIEVED %.1f nm: %.3e ' ...
             '(rank %d) -> ratio %.2fx | at 50 nm: %.3e\n'], ...
            k{1}, S.res.c_relin, ach, la1.curve_con(rk), rk, ...
            S.res.c_relin/la1.curve_con(rk), la1.floor);
end
