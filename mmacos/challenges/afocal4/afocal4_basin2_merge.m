function R = afocal4_basin2_merge(opts)
%AFOCAL4_BASIN2_MERGE  Collect the per-trade-point basin-2 long solves.
%
%   AFOCAL4_BASIN2 is run one process per interface standoff, because the
%   solves are hours and they are independent; each writes
%   afocal4_basin2_<tag>.mat.  This gathers them into one
%   afocal4_basin2.mat in trade order, and PRUNES what does not need
%   keeping: the per-seed pupil_map structs and the per-evaluation solver
%   histories are large, reproducible and of no interest once the run is
%   over, while the delivered design, its score, its gradient and its
%   descent probe are the record.
%
%   It also folds in the probe of the S4b delivered designs
%   (afocal4_basin2_s4bpoints.mat, written by the same driver in 'probe'
%   mode) when that file is present, because the answer to "was the old
%   number under-solved" is a COMPARISON and both halves belong in one
%   place.
%
%   Name-value:  'tags' (default 050 090 140 220 343 mm), 'prune' (true),
%                'save' (true), 'clean' (delete the per-tag files, false)
%
%   See also AFOCAL4_BASIN2.

    arguments
        opts.tags  (1,:) cell    = {'050mm','090mm','140mm','220mm','343mm'}
        opts.prune (1,1) logical = true
        opts.save  (1,1) logical = true
        opts.clean (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    PT = [];   src = {};
    for i = 1:numel(opts.tags)
        f = fullfile(here, sprintf('afocal4_basin2_%s.mat', opts.tags{i}));
        if ~isfile(f), fprintf('  (%s missing)\n', f);  continue; end
        q = load(f);
        if ~isfield(q,'R') || ~isfield(q.R,'pt') || isempty(q.R.pt)
            fprintf('  (%s has no delivered point)\n', f);  continue;
        end
        p = q.R.pt;
        if opts.prune, p = prune_(p); end
        if isempty(PT), PT = p; else, PT = [PT, p]; end %#ok<AGROW>
        src{end+1} = f; %#ok<AGROW>
        if opts.clean, delete(f); end
    end
    if isempty(PT), error('macos:design:afocal4_basin2_merge:none', ...
                          'no per-tag results to merge'); end
    [~,o] = sort([PT.iface]);   PT = PT(o);

    R = struct('pt',PT, 'sources',{src}, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    fs = fullfile(here,'afocal4_basin2_s4bpoints.mat');
    if isfile(fs)
        q = load(fs);
        if isfield(q,'R') && isfield(q.R,'pt')
            R.s4b_probe = keep_(q.R.pt, {'iface','D','S','grad','descent'});
        end
        if opts.clean, delete(fs); end
    end

    fprintf('\n  %-8s %10s %10s %9s %9s %9s %8s %8s\n', 'iface','merit', ...
            'WFE nm','blur um','breathe%','wander um','|g|','gain %');
    for i = 1:numel(PT)
        g = NaN;   gain = NaN;
        if isstruct(PT(i).grad),    g    = norm(PT(i).grad.g); end
        if isfield(PT,'descent') && isstruct(PT(i).descent)
            gain = 100*PT(i).descent.gain;
        end
        fprintf('  %6.0f mm %10.4f %10.1f %9.1f %9.4f %9.1f %8.3g %8.3g\n', ...
                PT(i).iface*1e3, PT(i).merit, PT(i).S.wfe_max_nm, ...
                PT(i).S.blur_um, PT(i).S.breathe_pct, PT(i).S.wander_um, ...
                g, gain);
    end
    if opts.save
        f = fullfile(here,'afocal4_basin2.mat');
        save(f, 'R', '-v7.3');
        d = dir(f);
        fprintf('\n  saved %s (%.1f MB)\n', f, d.bytes/1e6);
    end
end

% =====================================================================
function p = prune_(p)
%PRUNE_  Drop what is large and reproducible, keep what is the record.
    for i = 1:numel(p)
        if isstruct(p(i).all)
            a = p(i).all;
            for j = 1:numel(a)
                if isstruct(a(j).S) && isfield(a(j).S,'pm'), a(j).S.pm = []; end
            end
            p(i).all = a;
        end
        if isstruct(p(i).build) && isfield(p(i).build,'C')
            % the closure is re-derivable from D in one call; the deck is
            % committed beside it
            p(i).build = rmfield_(p(i).build, {'zi'});
        end
    end
end

function s = rmfield_(s, f)
    for i = 1:numel(f)
        if isfield(s, f{i}), s = rmfield(s, f{i}); end
    end
end

function q = keep_(p, f)
    q = struct();
    for i = 1:numel(p)
        for j = 1:numel(f)
            if isfield(p, f{j}), q(i).(f{j}) = p(i).(f{j}); end
        end
    end
end
