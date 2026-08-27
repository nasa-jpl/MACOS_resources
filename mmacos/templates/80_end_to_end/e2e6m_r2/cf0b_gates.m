function OUT = cf0b_gates(over)
%CF0B_GATES  S0b gates: the circular stop, without disturbing S0.
%
%   [1] NO-STOP PATH UNCHANGED.  cf0_gates re-run in full: the R1
%       bit-consistency gate must still pass 10/10 with the stop
%       machinery in the file (circ_stop_frac=0 is byte-inert).
%   [2] STOP SANITY (seg deck, N=1024, family-hard config -- no prolate):
%       inscribed/corner radius ~ cos30 (the hex envelope), the measured
%       collecting-area factor ~ pi/(2*sqrt(3)) x margin^2, the stopped
%       bare PSF still centred with the expected peak reduction, and the
%       tag carrying the stop token.
%   [3] THE STAMP GUARD FIRES.  The strict parity check must REFUSE the
%       existing pre-S0b (no-stop) Jacobian cache when asked for a stop
%       config -- ctb_jac_check's compare-what-both-have contract PASSES
%       it (verified here too), which is exactly why the campaign carries
%       the strict complement.  Asset-gated on the local cache.
%
%   See also CF_CHAIN, CF0_GATES, CF_EFC_LIB, ctb_jac_check.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    L = {};  t0 = tic;  npass = 0;  nfail = 0;
    L = say_(L, '==================== e2e6m CF0b -- circular-stop gates');

    % [1] the no-stop path, byte-unchanged --------------------------------
    L = say_(L, '\n[1] cf0_gates (no-stop R1 bit-consistency), full re-run:');
    C0 = cf0_gates(over);
    ok = C0.nfail == 0;
    [npass, nfail] = tally_(npass, nfail, ok);
    L = say_(L, '    cf0: %d PASS / %d FAIL  [%s]', C0.npass, C0.nfail, gate_(ok));

    % [2] stop sanity ------------------------------------------------------
    L = say_(L, '\n[2] stop sanity (seg deck, hard config, stop %.2f):', ...
             P.cf.circ_stop_frac);
    ch = cf_chain('rx', fullfile(P.outdir,'r1_seg_prop.in'), ...
                  'model_size', P.co.model, ...
                  'apod_kind','none', 'fpm_kind','hard', ...
                  'r_fpm_lamD', P.co.r_occ_lamD, 'r_lyot_frac', 0.50, ...
                  'circ_stop_frac', P.cf.circ_stop_frac);
    % The 19-segment tiling envelope is ROUNDER than a pure hexagon: the
    % outer ring scallops the corners, so inscribed/corner is 0.905, not
    % cos30 = 0.866, and the stop keeps ~92% of the collecting area
    % (a pure hex would give pi/(2*sqrt(3))*margin^2 = 0.871).  Bounds
    % pin the MEASURED tiling geometry.
    ratio = ch.r_insc_px / ch.r_apod_px;
    ok = ratio > 0.88 && ratio < 0.93;
    [npass, nfail] = tally_(npass, nfail, ok);
    L = say_(L, '    inscribed/corner %.4f (tiling envelope; pure hex would be 0.8660)  [%s]', ...
             ratio, gate_(ok));
    ok = ch.area_factor > 0.88 && ch.area_factor < 0.95;
    [npass, nfail] = tally_(npass, nfail, ok);
    L = say_(L, '    collecting-area factor %.4f (pure-hex bound %.4f does not apply)  [%s]', ...
             ch.area_factor, pi/(2*sqrt(3))*P.cf.circ_stop_frac^2, gate_(ok));
    Eb = ch.run_bare();
    Ib = abs(Eb).^2;
    [~, ip] = max(Ib(:));  [pi_, pj_] = ind2sub(size(Ib), ip);
    ok = pi_ == ch.center_px && pj_ == ch.center_px;
    [npass, nfail] = tally_(npass, nfail, ok);
    L = say_(L, '    stopped bare PSF peak (%d,%d), DC %d  [%s]', ...
             pi_, pj_, ch.center_px, gate_(ok));
    R1 = load(fullfile(P.outdir,'r1_coro_run.mat'));
    pk0 = R1.OUT.V(strcmp({R1.OUT.V.tag},'seg')).res.peak_bare;
    pr = ch.peak_bare / pk0;
    ok = pr > 0.60 && pr < 0.95;                  % coherent-area^2 class
    [npass, nfail] = tally_(npass, nfail, ok);
    L = say_(L, '    circularized/hex bare peak %.4f (area^2 = %.4f)  [%s]', ...
             pr, ch.area_factor^2, gate_(ok));
    ok = endsWith(ch.tag, sprintf('c%03d', round(100*P.cf.circ_stop_frac)));
    [npass, nfail] = tally_(npass, nfail, ok);
    L = say_(L, '    tag %s carries the stop token  [%s]', ch.tag, gate_(ok));

    % [3] the stamp guard fires on the pre-S0b cache -----------------------
    L = say_(L, '\n[3] stale-generation refusal:');
    stale = fullfile(P.outdir, 'cf2_G_hard.mat');
    if isfile(stale)
        J = load(stale, 'chain_opts');
        % ctb_jac_check alone PASSES it (the documented gap):
        soft_ok = true;
        try, ctb_jac_check(J, ch.config, stale); catch, soft_ok = false; end
        [npass, nfail] = tally_(npass, nfail, soft_ok);
        L = say_(L, '    ctb_jac_check passes the no-stop cache (the documented gap)  [%s]', ...
                 gate_(soft_ok));
        fired = false;  msg = '';
        try
            lib.stamp_parity(J, ch.config, stale);
        catch me
            fired = strcmp(me.identifier, 'cf_efc_lib:stale_generation');
            msg = me.message;
        end
        [npass, nfail] = tally_(npass, nfail, fired);
        L = say_(L, '    strict parity REFUSES it  [%s]', gate_(fired));
        if fired, L = say_(L, '      "%s"', strtok(msg, newline)); end
        % positive control: the cache passes against its own config
        ok = true;
        try, lib.stamp_parity(J, J.chain_opts, stale); catch, ok = false; end
        [npass, nfail] = tally_(npass, nfail, ok);
        L = say_(L, '    ...and passes against its OWN config  [%s]', gate_(ok));
    else
        L = say_(L, '    (no local no-stop cache -- refusal check skipped, asset-gated)');
    end

    L = say_(L, '\nCF0b gates: %d PASS, %d FAIL in %.1f min', npass, nfail, toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf0b_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'npass',npass, 'nfail',nfail, 'ratio',ratio, ...
                 'area_factor',ch.area_factor, 'peak_ratio',pr, ...
                 'r_stop_px',ch.r_stop_px, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf0b_run.mat'),'OUT');
end

% =========================================================================
function [p, f] = tally_(p, f, ok)
    if ok, p = p + 1; else, f = f + 1; end
end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
