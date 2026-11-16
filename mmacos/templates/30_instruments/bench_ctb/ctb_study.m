function out = ctb_study(opts)
%CTB_STUDY  Reconstruct the CTB control/physics/vector-vortex study from one config.
%   out = CTB_STUDY() re-runs the SESSIONS 10-14 study (deck slides
%   9-13) at the shipped configuration; name-value overrides re-run it
%   at a different parameter point with every stage, tag, cache file,
%   and figure derived from the config -- no hand-assembled call
%   sequences, no reused-stale-cache traps.
%
%   The sequence (stage -> driver):
%     'jac'        ctb_dm_jacobian on the masked chain (G at flat DMs)
%     'efc'        ctb_efc closed loop, fixed G
%     'relin'      ctb_dm_jacobian about the dug state + warm-start ctb_efc
%     'physics'    ctb_efc_physics: pol-only, band-only, band+pol
%     'bandwidth'  ctb_vortex_bandwidth (superset-Jacobian sweep, colors
%                  by the 2.5%-spacing rule)
%     'vvc'        the ctb_vvc ladder: ideal (+analyzed), zero-order
%                  unpolarized per band, circular sandwich ladder (+full
%                  stack, +per-lambda recovery), crossed-linear ladder
%                  (+full stack)
%     'summary'    ctb_vvc_summary verdict figure
%
%   Config name-value:
%     'fpm_kind'    'vortex' (default) | 'hard'
%     'charge'      vortex charge (4)
%     'r_lyot_frac' Lyot fraction (0.60)
%     'bands'       band ladder ([0 0.05 0.10 0.20]); colors derived per
%                   band as max(3, 2*round(band/0.05)+1)
%     'stages'      subset of the list above (default: all)
%     'niter_efc'   fixed-G loop iterations (19); 'niter' others (12)
%     'tag'         override the auto suffix (see below)
%     'force'       re-run stages whose run states already exist (false)
%     'dry'         print the full call plan -- drivers, tags, cache
%                   files, rough cost -- and return WITHOUT engine work
%
%   TAGS AND CACHES.  The shipped config maps to the historical tags
%   ('vortex', 'vortex_r1', 'bb', 'bwsweep', 'ideal', 'circ10s', ...) so
%   a default run resumes off the existing run states and caches.  Any
%   other config gets an auto suffix (e.g. '_c6L070') appended to every
%   tag, cache, and output -- configs never collide on disk.  Every
%   cached Jacobian is additionally verified against its stored
%   chain_opts stamp on load (ctb_jac_check): the file NAME is a hint,
%   the stamp is the authority.
%
%   RESUMABILITY.  A stage whose run-state .mat already exists is
%   SKIPPED and its numbers folded into the manifest ('force' re-runs).
%   A default-config ctb_study() over complete states therefore costs
%   seconds and returns the study manifest -- the audit that everything
%   on the deck is reconstructible.
%
%   Limitation: ctb_vvc_summary assumes the standard band ladder; a
%   custom 'bands' still produces every run state, but compose the
%   verdict figure yourself.
%
%   out: config, suffix, manifest (stage/tag/file/c_before/c_after/
%   status), saved as ctb_study<suffix>.mat.
%
%   Run:  >> ctb_study('dry', true);            % the plan, no engine
%         >> out = ctb_study();                 % audit/resume shipped study
%         >> out = ctb_study('charge', 6, 'r_lyot_frac', 0.70);
%   See also: ctb_jac_check, ctb_chain, ctb_vvc, CTB_PROP_STATUS.md.
    arguments
        opts.fpm_kind    (1,:) char {mustBeMember(opts.fpm_kind, {'hard','vortex'})} = 'vortex'
        opts.charge      (1,1) double = 4
        opts.r_lyot_frac (1,1) double = 0.60
        opts.bands       (1,:) double = [0 0.05 0.10 0.20]
        opts.stages      (1,:) cell = {'jac','efc','relin','physics', ...
                                       'bandwidth','vvc','summary'}
        opts.niter_efc   (1,1) double = 19
        opts.niter       (1,1) double = 12
        opts.tag         (1,:) char = ''
        opts.force       (1,1) logical = false
        opts.dry         (1,1) logical = false
        opts.outdir      (1,:) char = ''
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end

    % ---- suffix: '' for the shipped config, else derived (or 'tag') ----
    shipped = strcmp(opts.fpm_kind, 'vortex') && opts.charge == 4 && ...
              opts.r_lyot_frac == 0.60 && ...
              isequal(opts.bands, [0 0.05 0.10 0.20]);
    if ~isempty(opts.tag)
        sfx = opts.tag;
        if sfx(1) ~= '_', sfx = ['_' sfx]; end
    elseif shipped
        sfx = '';
    else
        sfx = sprintf('_%sc%dL%03d', opts.fpm_kind(1), opts.charge, ...
                      round(100*opts.r_lyot_frac));
    end
    if strcmp(opts.fpm_kind, 'vortex')
        chain = {'fpm_kind','vortex', 'charge',opts.charge, ...
                 'apodizer',false, 'r_lyot_frac',opts.r_lyot_frac};
    else
        chain = {'fpm_kind','hard', 'r_lyot_frac',opts.r_lyot_frac};
    end
    jtag  = [opts.fpm_kind sfx];
    bandsp = opts.bands(opts.bands > 0);

    % ---- build the plan -------------------------------------------------
    % each step: name, done (run-state .mat, '' = always run), call
    % (display string), run (thunk returning a result struct), est_min
    P = {};
    chs = cell2str_(chain);
    jfile  = sprintf('ctb_dm_jacobian_N512_%s.mat', jtag);
    j1file = sprintf('ctb_dm_jacobian_N512_%s_r1.mat', jtag);

    if any(strcmp(opts.stages, 'jac'))
        P{end+1} = step_('jac', jfile, ...
            sprintf('ctb_dm_jacobian(''chain'', %s, ''tag'', ''%s'')', chs, jtag), ...
            @() ctb_dm_jacobian('chain', chain, 'tag', jtag), 12);
    end
    if any(strcmp(opts.stages, 'efc'))
        P{end+1} = step_('efc', sprintf('ctb_efc_%s.mat', jtag), ...
            sprintf('ctb_efc(''jac'', ''%s'', ''niter'', %d, ''tag'', ''%s'')', ...
                jfile, opts.niter_efc, jtag), ...
            @() ctb_efc('jac', fullfile(here, jfile), ...
                'niter', opts.niter_efc, 'tag', jtag), 10);
    end
    if any(strcmp(opts.stages, 'relin'))
        P{end+1} = step_('relin-jac', j1file, ...
            sprintf('ctb_dm_jacobian(''chain'', %s, ''tag'', ''%s_r1'', ''a0'', <efc.a>)', chs, jtag), ...
            @() relin_jac_(here, jtag, chain), 12);
        P{end+1} = step_('relin-efc', sprintf('ctb_efc_%s_r1.mat', jtag), ...
            sprintf('ctb_efc(''jac'', ''%s'', ''a0'', <efc.a>, ''niter'', %d, ''tag'', ''%s_r1'')', ...
                j1file, opts.niter, jtag), ...
            @() relin_efc_(here, jtag, opts.niter), 8);
    end
    if any(strcmp(opts.stages, 'physics'))
        phy = {'pol',  {'pol',true};
               'bb',   {'band',true, 'lfracs',[0.95 1.00 1.05]};
               'bbpol',{'band',true, 'lfracs',[0.95 1.00 1.05], 'pol',true}};
        for i = 1:size(phy, 1)
            ptag = [phy{i,1} sfx];
            nv = phy{i,2};
            P{end+1} = step_(['physics-' phy{i,1}], sprintf('ctb_efc_phys_%s.mat', ptag), ...
                sprintf('ctb_efc_physics(%s, ''chain'', %s, ''tag'', ''%s'')', ...
                    cell2str_(nv, false), chs, ptag), ...
                @() ctb_efc_physics(nv{:}, 'chain', chain, 'tag', ptag, ...
                    'niter', opts.niter), 20);
        end
    end
    if any(strcmp(opts.stages, 'bandwidth'))
        colors = arrayfun(@(b) max(3, 2*round(b/0.05) + 1) * (b > 0) + (b == 0), opts.bands);
        btag = 'bwsweep';  if ~isempty(sfx), btag = ['bwsweep' sfx]; end
        bmat = 'ctb_vortex_bandwidth.mat';
        if ~strcmp(btag, 'bwsweep'), bmat = sprintf('ctb_vortex_bandwidth_%s.mat', btag); end
        P{end+1} = step_('bandwidth', bmat, ...
            sprintf('ctb_vortex_bandwidth(''bands'', %s, ''colors'', %s, ''chain'', %s, ''tag'', ''%s'')', ...
                mat2str(opts.bands), mat2str(colors), chs, btag), ...
            @() ctb_vortex_bandwidth('bands', opts.bands, 'colors', colors, ...
                'chain', chain, 'tag', btag, 'niter', opts.niter), 120);
    end
    if any(strcmp(opts.stages, 'vvc'))
        vv = {'ideal',          {'tier','ideal'},                                          8;
              'ideal_analyzed', {'tier','ideal', 'analyzer','circular'},                   8};
        for b = bandsp
            vv(end+1, :) = {sprintf('c%02d', round(100*b)), {'band', b}, 10};              %#ok<AGROW>
        end
        for b = opts.bands
            vv(end+1, :) = {sprintf('circ%02d', round(100*b)), ...
                {'input','circular', 'analyzer','circular', 'band', b}, 10};               %#ok<AGROW>
        end
        if any(opts.bands == 0.10)
            vv(end+1, :) = {'circ10s', {'input','circular', 'analyzer','circular', ...
                'band', 0.10, 'pol', true}, 15};
        end
        if any(opts.bands == 0.05)
            vv(end+1, :) = {'circ05_perlam', {'input','circular', 'analyzer','circular', ...
                'band', 0.05, 'jac_perlam', true}, 90};
        end
        for b = opts.bands
            vv(end+1, :) = {sprintf('lin%02d', round(100*b)), ...
                {'input','linear', 'analyzer','linear', 'band', b}, 8};                    %#ok<AGROW>
        end
        if any(opts.bands == 0.10)
            vv(end+1, :) = {'lin10s', {'input','linear', 'analyzer','linear', ...
                'band', 0.10, 'pol', true}, 12};
        end
        for i = 1:size(vv, 1)
            vtag = [vv{i,1} sfx];
            nv = [vv{i,2}, {'charge', opts.charge, 'r_lyot_frac', opts.r_lyot_frac}];
            P{end+1} = step_(['vvc-' vv{i,1}], sprintf('ctb_vvc_%s.mat', vtag), ...
                sprintf('ctb_vvc(%s, ''tag'', ''%s'')', cell2str_(nv, false), vtag), ...
                @() ctb_vvc(nv{:}, 'tag', vtag, 'niter', opts.niter), vv{i,3});
        end
        % the shared 2-mask Jacobian is measured by the first vvc step
        % (~26 min) and reused by every later one -- reflected in est
    end
    if any(strcmp(opts.stages, 'summary'))
        P{end+1} = step_('summary', '', ...
            sprintf('ctb_vvc_summary(''suffix'', ''%s'')', sfx), ...
            @() ctb_vvc_summary('suffix', sfx), 1);
    end

    % ---- dry: print the plan and stop ----------------------------------
    fprintf('[study] config: %s charge %d Lyot %.2f bands %s -> suffix ''%s''\n', ...
        opts.fpm_kind, opts.charge, opts.r_lyot_frac, mat2str(opts.bands), sfx);
    if opts.dry
        tot = 0;
        for i = 1:numel(P)
            s = P{i};
            have = ~isempty(s.done) && isfile(fullfile(here, s.done));
            mark = '     ';  if have, mark = '[has]'; end
            fprintf('  %2d %s %-16s %s\n', i, mark, s.name, s.call);
            if ~have, tot = tot + s.est; end
        end
        fprintf('[study] DRY: %d steps, ~%.1f hr of engine work outstanding\n', ...
            numel(P), tot/60);
        out = struct('config', {chain}, 'suffix', sfx, 'plan', {P});
        return
    end

    % ---- live: run/skip each step, collect the manifest ----------------
    M = {};
    for i = 1:numel(P)
        s = P{i};
        f = fullfile(here, s.done);
        if ~isempty(s.done) && isfile(f) && ~opts.force
            r = load(f);
            if isfield(r, 'chain_opts'), ctb_jac_check(r, chain, s.done); end
            M(end+1, :) = {s.name, s.done, pick_(r, 'c_before'), ...
                pick_(r, 'c_after'), 'skipped (state exists)'};       %#ok<AGROW>
            fprintf('[study] %2d %-16s SKIP (%s exists)\n', i, s.name, s.done);
            continue
        end
        fprintf('[study] %2d %-16s RUN  %s\n', i, s.name, s.call);
        r = s.run();
        M(end+1, :) = {s.name, s.done, pick_(r, 'c_before'), ...
            pick_(r, 'c_after'), 'ran'};                              %#ok<AGROW>
    end
    out = struct('config', {chain}, 'suffix', sfx, ...
        'manifest', {cell2table(M, 'VariableNames', ...
            {'stage','file','c_before','c_after','status'})});
    save(fullfile(opts.outdir, ['ctb_study' sfx '.mat']), '-struct', 'out');
    disp(out.manifest);
    fprintf('[study] manifest saved: ctb_study%s.mat\n', sfx);
end

function J = relin_jac_(here, jtag, chain)
    E = load(fullfile(here, sprintf('ctb_efc_%s.mat', jtag)));
    J = ctb_dm_jacobian('chain', chain, 'tag', [jtag '_r1'], 'a0', E.a);
end

function o = relin_efc_(here, jtag, niter)
    E = load(fullfile(here, sprintf('ctb_efc_%s.mat', jtag)));
    o = ctb_efc('jac', fullfile(here, sprintf('ctb_dm_jacobian_N512_%s_r1.mat', jtag)), ...
        'a0', E.a, 'niter', niter, 'tag', [jtag '_r1']);
end

function s = step_(name, done, call, run, est)
    s = struct('name',name, 'done',done, 'call',call, 'run',run, 'est',est);
end

function v = pick_(r, f)
    v = NaN;
    if isstruct(r) && isfield(r, f), v = r.(f); end
end

function t = cell2str_(c, wrap)
    if nargin < 2, wrap = true; end
    p = cell(1, numel(c));
    for i = 1:numel(c)
        v = c{i};
        if ischar(v),          p{i} = ['''' v ''''];
        elseif islogical(v),   w = {'false','true'};  p{i} = w{v+1};
        else,                  p{i} = mat2str(v);
        end
    end
    t = strjoin(p, ', ');
    if wrap, t = ['{' t '}']; end
end
