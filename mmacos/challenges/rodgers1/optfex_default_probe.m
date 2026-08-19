function optfex_default_probe()
%OPTFEX_DEFAULT_PROBE  What does LOptIfFEX=.TRUE. do on a deck whose CALIB
%   FEX target is INVALID?  This decides whether the unified default can be
%   .TRUE. (physically the right merit) or must be .FALSE. with deck opt-in.
%
%   CALIB's FEX call is hard-wired to element nElt-1
%   (smacos_compute.inc:391-397).  On a plain [M1 M2 M3 FP] deck that is a
%   REFLECTOR.  The interactive FEXIT dispatch rejects a non-Return/Reference
%   target (macos_cmd_loop.inc:2618-2627), but CALIB does NOT go through that
%   dispatch -- MACOS_OPS calls SUBROUTINE FEX directly and then
%   unconditionally overwrites the target element's eElt/fElt/KcElt/KrElt/
%   zElt/psiElt/VptElt/RptElt (macos_ops.F:60-84).  This measures what that
%   actually does to M3.
%
%   Method: emit a real 6-element optimisation deck (which carries a valid
%   Opt block with OptFEX= Yes), then surgically remove the FP_return and
%   ExitPupil elements so the deck is [M1 M2 M3 FP] with nElt-1 = M3, and
%   run CALIB on it.  Record M3's surface parameters before and after.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common();  P.EPD_mm = 4060;

    % ---- 1. emit a valid optimisation deck WITH the pupil ---------------
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_field_bias(P.offset_deg*60);  t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    t.add_pupil();
    optF = macos.design.field_grid(6,3,'units','arcmin','origin',false);
    try
        t.optimize('fields',optF,'dofs',[0 0 0 0 0 0 0 1],'max_iters',1);
    catch ME
        fprintf('#### optimize threw (expected if guarded): %s\n', ME.message);
    end
    txt6 = fileread(t.spec.rx_path);

    % ---- 2. surgery: drop FP_return (4) and ExitPupil (5) ---------------
    L = regexp(txt6, '\n', 'split');
    ib = find(~cellfun(@isempty, regexp(L, '^\s*iElt=', 'once')));
    fprintf('#### emitted deck: %d element blocks at lines %s\n', ...
            numel(ib), mat2str(ib));
    assert(numel(ib) == 6, 'expected 6 element blocks');
    seg = @(k) L(ib(k) : merge_end_(ib, numel(L), k));
    head = L(1:ib(1)-1);
    keep = [seg(1) seg(2) seg(3) seg(6)];         % M1 M2 M3 FP
    keep = regexprep(keep, '^(\s*iElt=\s*)6$', '$14');
    txt4 = strjoin([head keep], sprintf('\n'));
    txt4 = regexprep(txt4, '(nElt=\s*)6', '$14');
    txt4 = regexprep(txt4, '(OptWFElt=\s*)5', '$14');
    txt4 = regexprep(txt4, '(ApType=\s*)\S+', '$1None');
    deck4 = [tempname '.in'];
    fid = fopen(deck4,'w'); fprintf(fid,'%s',txt4); fclose(fid);

    % ---- 3. load and record M3 BEFORE -----------------------------------
    macos.init(P.model_size);  macos.load_rx(deck4);
    nE = macos.num_elt();
    fprintf('#### loaded 4-element deck: nElt=%d, FEX target would be elt %d\n', nE, nE-1);
    before = snap_(3);
    show_('BEFORE', before);
    macos.stop(1);

    % ---- 4. run CALIB ----------------------------------------------------
    fprintf('#### running CALIB with OptFEX= Yes and nElt-1 = M3 (a Reflector)...\n');
    ok = true;
    try
        macos.calib();
    catch ME
        ok = false;
        fprintf('#### CALIB threw: %s\n', ME.message);
    end
    after = snap_(3);
    show_('AFTER ', after);
    d = struct2array_(after) - struct2array_(before);
    fprintf('#### CALIB completed without error: %d\n', ok);
    fprintf('#### M3 CHANGED: %d   (max |delta| = %.6g)\n', any(abs(d) > 0), max(abs(d)));
    if any(abs(d) > 0)
        fprintf(['#### VERDICT: state-mutating -- FEX overwrote a REFLECTOR.\n' ...
                 '####          The unified default must be .FALSE. (deck opt-in).\n']);
    else
        fprintf('#### VERDICT: graceful -- default .TRUE. would be safe.\n');
    end
    delete(deck4);
end

function e = merge_end_(ib, nL, k)
    if k < numel(ib), e = ib(k+1)-1; else, e = nL; end
end

function s = snap_(ie)
    s.Kr  = macos.get_elt_kr(ie);
    s.Kc  = macos.get_elt_kc(ie);
    s.Vpt = macos.get_elt_vpt(ie);
    s.psi = macos.get_elt_psi(ie);
end
function show_(tag, s)
    fprintf('#### %s M3: Kr=%.10g  Kc=%.10g  Vpt=[%.6g %.6g %.6g]  psi=[%.6g %.6g %.6g]\n', ...
            tag, s.Kr, s.Kc, s.Vpt, s.psi);
end
function v = struct2array_(s)
    v = [s.Kr; s.Kc; s.Vpt(:); s.psi(:)];
end
