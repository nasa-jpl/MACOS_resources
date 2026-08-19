function R = calib_afocal_probe(opts)
%CALIB_AFOCAL_PROBE  Does CALIB work on a deck that ends in a flat, and if
%   it runs, WHAT is it minimising there?  (PLAN_AFOCAL4 S1d -- short,
%   non-gating, for the record.  The sanctioned optimisation path for this
%   study is a MATLAB outer solve either way, Dave 2026-08-02.)
%
%   The question is not academic.  CALIB's merit is the engine's OPD at
%   OptWFElt, and `SUBROUTINE OPD` builds NO reference sphere -- it reports
%   the spread of the cumulative optical path ON THE ELEMENT SURFACE
%   (rodgers1 Addendum 3 A.1).  On a focal deck FEX supplies the missing
%   reference.  On an AFOCAL deck the last element is a flat, so the merit
%   becomes the OPL spread across a collimated beam measured on a fixed
%   plane -- and the moment that plane is not normal to the exit chief, the
%   spread is dominated by a TILT term of order (beam radius) x tan(angle),
%   which is millimetres where the wavefront error is nanometres.
%
%   This probe MEASURES that rather than asserting it:
%
%     Section 0  WHICH ELEMENT TYPE the interface flat should be.  The plan
%                proposed a flat RETURN.  It must not be: `Element= Return`
%                REVERSES the ray directions at that surface, so a metric
%                that builds its reference from the exit chief builds it
%                backwards.  Measured below.
%     Section 1  the engine's own OPD at the interface, field by field,
%                against the afocal ladder's three rungs.  The ratio is the
%                answer: if the engine OPD tracks rung 1 the merit is
%                usable, and if it runs 4-5 orders high it is measuring the
%                coldstop's tilt.
%     Section 2  does CALIB LOAD and RUN on such a deck at all -- with the
%                interface emitted as Reference and as Return, the two
%                candidate terminal element types.
%     Section 3  what one short CALIB solve does to the design: which way
%                the conic moves, and whether the afocal ladder improves or
%                degrades under it.
%
%   R = CALIB_AFOCAL_PROBE() runs all three and returns the numbers.
%
%   Name-value:  'variant' (3 = S3_newconics)  'iters' (5)  'quiet' (false)

    arguments
        opts.variant (1,1) double = 3
        opts.iters   (1,1) double = 5
        opts.quiet   (1,1) logical = false
        opts.keep    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);

    S  = rodgers2_seq();
    V  = S.v(opts.variant);
    %#ok<*NASGU>
    F  = S.Frel;                          % his 3x3 solve set
    macos.init(256);

    R = struct('variant',V.name, 'iters',opts.iters);

    % =================================================================
    banner_('0.  which element type should the interface flat be?');
    % =================================================================
    R.kinds = struct('kind',{},'chief_dir',{},'rungs_nm',{},'opl_std_m',{});
    for kind = {'Reference','Return','FocalPlane'}
        f = [tempname '.in'];
        rodgers2_deck(opts.variant, 'coldstop', kind{1}, 'file', f);
        Lk = afocal_ladder_deck(f, F);
        macos.load_rx(f);
        tr = macos.trace(macos.num_elt());   ri = macos.get_ray_info(tr.nRays);
        ok = ri.ok_trace & ri.ok_pass;
        e = struct('kind',kind{1}, 'chief_dir',ri.dir(:,1).', ...
                   'rungs_nm',max(Lk)*1e9, 'opl_std_m',std(ri.opl(ok)));
        if isempty(R.kinds), R.kinds = e; else, R.kinds(end+1) = e; end
        fprintf(['  %-11s chief dir [%+9.6f %+9.6f %+9.6f]  rungs max ' ...
                 '%12.3f %10.3f %10.3f nm  OPL std %.6e m\n'], ...
            e.kind, e.chief_dir, e.rungs_nm, e.opl_std_m);
        delete_if_(f);
    end
    fprintf(['\n  Reference and FocalPlane are identical.  RETURN REVERSES THE\n' ...
             '  RAY DIRECTIONS -- the chief comes back negated -- so a reference\n' ...
             '  plane built from the exit chief is built backwards and rung 1\n' ...
             '  reads millimetres.  The OPL itself is unaffected (identical std),\n' ...
             '  which is why this hides from any piston-only check.  The interface\n' ...
             '  element must be Element= Reference.\n']);

    % =================================================================
    banner_('1.  the engine OPD at the interface vs the afocal ladder');
    % =================================================================
    o = rodgers2_deck(opts.variant, 'coldstop','Reference', ...
                      'file', fullfile(here,'rodgers2_probe_ref.in'));
    [L, ~] = afocal_ladder_deck(o.file, F);
    eng = engine_opd_(o.file, F);
    R.eng_opd_m = eng;   R.ladder_m = L;
    fprintf('  %6s %6s | %12s | %12s %12s %12s | %10s\n', ...
            'thx','thy','engine OPD','rung1','rung2','rung3','eng/rung1');
    for k = 1:size(F,1)
        fprintf('  %6.2f %6.2f | %12.4e | %12.4e %12.4e %12.4e | %10.3g\n', ...
            F(k,1)*180/pi, F(k,2)*180/pi, eng(k), L(k,1), L(k,2), L(k,3), ...
            eng(k)/L(k,1));
    end
    R.eng_over_rung1 = eng ./ L(:,1);
    fprintf(['\n  The engine merit runs %.3g .. %.3g times rung 1.  It is the OPL\n' ...
             '  spread on the COLDSTOP SURFACE, and the coldstop is tilted %.3f deg\n' ...
             '  off the exit chief by design -- so what CALIB would minimise here is\n' ...
             '  the beam''s angle against a fixed plane, not its wavefront error.\n'], ...
            min(R.eng_over_rung1), max(R.eng_over_rung1), V.coldstop_ADE_deg);

    % =================================================================
    banner_('2.  does CALIB load and run on a deck terminating in a flat?');
    % =================================================================
    R.run = [];
    for kind = {'Reference','Return'}
        r = try_calib_(here, opts, kind{1}, F);
        if isempty(R.run), R.run = r; else, R.run(end+1) = r; end
        fprintf('  %-10s : loaded %d  ran %d  rtn %s  merit %s -> %s   %s\n', ...
            r.kind, r.loaded, r.ran, num2str(r.rtn), ...
            fmt_(r.old), fmt_(r.new), r.msg);
    end

    % =================================================================
    banner_('3.  what the solve did to the design');
    % =================================================================
    for i = 1:numel(R.run)
        r = R.run(i);
        if ~r.ran, continue; end
        fprintf('  %-10s : K_M2 %.12f -> %.12f   (delta %+.3e)\n', ...
                r.kind, r.K_before, r.K_after, r.K_after - r.K_before);
        fprintf('               afocal ladder max, rung 2: %.3f -> %.3f nm\n', ...
                r.lad_before*1e9, r.lad_after*1e9);
    end

    banner_('VERDICT');
    fprintf(['  Recorded for PACKET.md section 4.  The MATLAB outer solve over\n' ...
             '  the afocal ladder is the sanctioned path for this study\n' ...
             '  regardless of the answer here (Dave, 2026-08-02).  Making CALIB\n' ...
             '  usable on an afocal deck needs an afocal REFERENCE inside the\n' ...
             '  engine -- the plane analogue of what FEX supplies on a focal\n' ...
             '  deck -- which is a follow-on engine task, not part of this plan.\n']);
    save(fullfile(here,'rodgers2_calib_probe.mat'), 'R');
    if ~opts.keep
        delete_if_(fullfile(here,'rodgers2_probe_ref.in'));
    end
end

% =====================================================================
function r = try_calib_(here, opts, kind, F)
%TRY_CALIB_  Emit an Opt-block deck with this terminal kind and run CALIB.
    r = struct('kind',kind, 'loaded',false, 'ran',false, 'rtn',NaN, ...
               'nfov',NaN, 'old',[], 'new',[], 'msg','', ...
               'K_before',NaN, 'K_after',NaN, ...
               'lad_before',NaN, 'lad_after',NaN);
    % The OPT deck lives in a TEMP file, never in the study directory: CALIB
    % writes its optimised state back over the deck it was loaded from, and
    % that saved copy has the LAST probe field's chief ray baked in AND -- a
    % separate save round-trip gap -- no ApStop line at all.  Scoring it
    % would be scoring neither the design nor the field.
    f = [tempname '.in'];
    S = rodgers2_seq();   V = S.v(opts.variant);
    o = rodgers2_deck(opts.variant, 'coldstop', kind, 'file', f);
    txt = add_opt_block_(o.txt, o.nElt, F, o.ApStop, o.stand, opts.iters);
    fid = fopen(f,'w');  fprintf(fid,'%s',txt);  fclose(fid);

    % Score BEFORE *first*, from a clean transcription deck.  Order is
    % load-bearing: afocal_ladder_deck loads decks of its own, and loading
    % ANY other Rx resets nVarElt/nOptFov, so a "before" score taken after
    % the Opt deck is loaded silently disarms CALIB (it then returns FAIL
    % from its nVarElt<1 pre-check and the probe reports "CALIB refuses to
    % run" -- which is not what happened).
    fb = [tempname '.in'];
    rodgers2_deck(opts.variant, 'coldstop', kind, 'file', fb);
    Lb = afocal_ladder_deck(fb, F);   r.lad_before = max(Lb(:,2));

    macos.load_rx(f);
    r.loaded = macos.has_rx();
    if ~r.loaded, r.msg = 'load failed';  delete_if_(f); delete_if_(fb); return; end
    r.K_before = macos.get_elt_kc(2);

    % NOTE: no macos.stop() call.  The deck declares its own ApStop 50 mm
    % ahead of M1 (his STO surface); macos.stop(1) would move it to the M1
    % vertex and silently change the system being optimised.
    try
        c = macos.calib();
        r.ran = true;  r.rtn = c.rtn_flag;  r.nfov = c.n_fov;
        r.old = c.old_wfe(:).';  r.new = c.new_wfe(:).';
        r.msg = ternary_(c.converged, 'converged', 'did not converge');
    catch ME
        r.msg = ME.message;  delete_if_(f);  delete_if_(fb);  return;
    end
    r.K_after = macos.get_elt_kc(2);

    % score AFTER by REBUILDING the transcription deck with the solved
    % conic, so the field set and the ApStop are the study's, not CALIB's
    Va = V;   Va.K(2) = r.K_after;
    fa = [tempname '.in'];
    rodgers2_deck(opts.variant, 'variant', Va, 'coldstop', kind, 'file', fa);
    La = afocal_ladder_deck(fa, F);   r.lad_after = max(La(:,2));
    delete_if_(f);  delete_if_(fb);  delete_if_(fa);
end

function txt = add_opt_block_(txt, nElt, F, apst, stand, iters)
%ADD_OPT_BLOCK_  The minimal CALIB configuration, appended to a deck.
%   OptFEX stays No: CALIB's FEX call is hard-wired to element nElt-1
%   (smacos_compute.inc), which on this deck is M3 -- a Reflector, and
%   rodgers1 measured that FEX overwrites such a target's surface
%   parameters outright (OPTFEX_DEFAULT_PROBE).  The probe is about the
%   afocal merit, not about re-running that hazard.
    v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
    tk = regexp(txt,'ChfRayDir=\s*([^\n]*)','tokens','once');
    cdir0 = sscanf(tk{1},'%f',3);
    L = {};
    L{end+1} = '        OptTarget=  WFE';
    L{end+1} = sprintf('         OptWFElt=  %d', nElt);
    L{end+1} = sprintf('       OptMaxItrs=  %d', iters);
    L{end+1} = '           OptFEX=  No';
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    n = 0;
    for j = 1:size(F,1)
        if all(abs(F(j,:)) < 1e-12), continue; end   % field 1 is IMPLICIT
        bx = bx0 + F(j,1);  by = by0 + F(j,2);
        d  = [sin(bx), sin(by), sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
        cp = apst - stand*d;
        L{end+1} = ['     OptChfRayDir=  ' v3(d)];   %#ok<AGROW>
        L{end+1} = ['     OptChfRayPos=  ' v3(cp)];  %#ok<AGROW>
        n = n + 1;
    end
    L{end+1} = ['         OptFOVWt=  ' strtrim(repmat('1  ', 1, n+1))];
    blk = [strjoin(L, newline) newline];
    % the Opt block belongs with the source definition, ahead of nElt
    i = regexp(txt, '\n[^\n]*nElt=', 'once');
    txt = [txt(1:i) blk txt(i+1:end)];
    % M2 conic is the single design variable: VarElt mask
    % [TIP TILT CLOCK DX DY PIST ROC CONIC]
    txt = insert_after_elt_(txt, 2, '           VarElt=  0 0 0 0 0 0 0 1');
end

function txt = insert_after_elt_(txt, ie, line)
%INSERT_AFTER_ELT_  Put LINE on its own line just after the `iElt= IE` line.
%   Done line-wise on purpose: a regex whose trailing \s* can eat the
%   newline splices the new key onto the END of the iElt line, and the
%   parser then never sees it -- which is exactly how the first version of
%   this probe reported "CALIB refuses to run on an afocal deck" when what
%   it had actually done was emit an unparseable VarElt.
    L = regexp(txt, '\n', 'split');
    pat = sprintf('^\\s*iElt=\\s*%d\\s*$', ie);
    k = find(~cellfun(@isempty, regexp(L, pat, 'once')), 1);
    if isempty(k)
        error('calib_afocal_probe:elt','no element block %d', ie);
    end
    txt = strjoin([L(1:k), {line}, L(k+1:end)], newline);
end

function e = engine_opd_(deck, F)
%ENGINE_OPD_  The engine's own RMS OPD at the last element, per field.
%   This is CALIB's merit, read directly.
    txt = regexprep(fileread(deck), '(ApType=\s*)\S+', '$1None');
    cdir0 = grab3_(txt,'ChfRayDir');  cpos0 = grab3_(txt,'ChfRayPos');
    apst  = grab3_(txt,'ApStop');
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    tmp = [tempname '.in'];   e = nan(size(F,1),1);
    for k = 1:size(F,1)
        bx = bx0 + F(k,1);  by = by0 + F(k,2);
        cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
        cpos = apst - stand*cdir;
        s = regexprep(txt,'(ChfRayDir=\s*)[^\n]*', ...
                      ['$1' sprintf('%.16E  %.16E  %.16E',cdir)]);
        s = regexprep(s,  '(ChfRayPos=\s*)[^\n]*', ...
                      ['$1' sprintf('%.16E  %.16E  %.16E',cpos)]);
        fid = fopen(tmp,'w'); fprintf(fid,'%s',s); fclose(fid);
        macos.load_rx(tmp);
        tr = macos.trace(macos.num_elt());
        e(k) = tr.rmsWFE;                  % BaseUnits (m), NOT waves
    end
    delete(tmp);
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function s = fmt_(v)
    if isempty(v), s = '--'; else, s = sprintf('%.4e', max(v(:))); end
end

function y = ternary_(c,a,b), if c, y = a; else, y = b; end, end
function banner_(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
function delete_if_(p), if exist(p,'file'), delete(p); end, end
