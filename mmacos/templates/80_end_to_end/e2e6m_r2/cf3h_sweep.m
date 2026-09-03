function OUT = cf3h_sweep(sigmas, over)
%CF3H_SWEEP  Run the feather-sigma pareto arms sequentially and append
%   the summary table to cf3h_report.txt.  ONE MATLAB, one arm at a time
%   (the box rule).  Each arm resumes from its cf3h_s<sigma>_run.mat
%   checkpoint if present; its row is written to cf3h_report.txt the
%   moment the arm returns, so a killed sweep still leaves the completed
%   arms in the table.
%
%   The pareto columns (Dave's deliverable): sigma, thru, floor reached
%   (c_end), la@50nm (stroke-bounded linear-achievable floor), laU
%   (stroke-released), and the strokes (bounded / released).
%
%   cf3h_sweep([1 3 4])          the overnight sweep
%   cf3h_sweep([1 3 4], over)    with over.cf3h knobs (wall_h, target, ...)
%
%   See also CF3H_SIGMA, CF3F_FEATHER.

    arguments
        sigmas (1,:) double = [1 3 4]
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    if ~isfield(over,'cf3h'),        over.cf3h = struct(); end
    if ~isfield(over.cf3h,'wall_h'), over.cf3h.wall_h = 2.0; end

    rep = fullfile(here, 'cf3h_report.txt');
    fid = fopen(rep, 'a');
    fprintf(fid, '==== e2e6m CF3h -- feather-sigma pareto (Mac arms) %s\n', ...
            datestr(now,31)); %#ok<DATST>
    fprintf(fid, 'sigma |  thru | floor(c_end) |   la@50nm  |    laU     | strk_nm | strkU_nm | rounds\n');
    fclose(fid);

    OUT = struct('sigma',{},'thru',{},'floor',{},'la50',{},'laU',{}, ...
                 'strk',{},'strkU',{},'rounds',{});
    for sig = sigmas(:).'
        ov = over;  ov.cf3h.feather_px = sig;
        fprintf('\n===== CF3h sweep arm sigma = %g px =====\n', sig);
        try
            o = cf3h_sigma(ov);
            H = o.hist(end);
            row = struct('sigma',sig, 'thru',o.thru, 'floor',H.c_end, ...
                'la50',H.la_floor, 'laU',H.la_unbound, 'strk',H.stroke_nm, ...
                'strkU',H.stroke_unbound, 'rounds',numel(o.hist));
        catch ME
            fprintf(2, 'CF3h arm sigma %g FAILED: %s\n', sig, ME.message);
            row = struct('sigma',sig, 'thru',NaN, 'floor',NaN, 'la50',NaN, ...
                'laU',NaN, 'strk',NaN, 'strkU',NaN, 'rounds',0);
        end
        OUT(end+1) = row; %#ok<AGROW>
        fid = fopen(rep, 'a');
        fprintf(fid, ' %5.1f | %5.3f | %12.3e | %10.3e | %10.3e | %7.1f | %8.1f | %6d\n', ...
            row.sigma, row.thru, row.floor, row.la50, row.laU, ...
            row.strk, row.strkU, row.rounds);
        fclose(fid);
    end
    fprintf('\nCF3h sweep complete -- table appended to %s\n', rep);
end
