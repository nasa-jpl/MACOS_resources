function OUT = ctb_dst_2c(N)
%CTB_DST_2C  DST Lane 2c: central amplitude defect + point inclusion at the
%   FPM.  Opaque disk 0.2 lambda/D at the vortex centre (DST's 4 um at
%   F/32.7) + a 0.2 lambda/D point inclusion at 5 lambda/D -- multiplied
%   onto the FPM amplitude (ctb_chain fpm_defect_lamD / fpm_incl_*).  The
%   defect is in ch.config, so it is in BOTH plant and Jacobian.  Matched
%   config: charge 6, Lyot 0.80, N=1024, 625 nm, half-plane 3-8.
%   control == truth, perfect sensing.
%
%   GATE (their chromatic signature): mono floor NEARLY UNCHANGED from the
%   defect-off baseline; broadband 10% floor DEGRADES (chromatic speckle in
%   the hole); switching the defect off recovers the baseline (the S1
%   references below).  References (defect OFF, same config, 625 nm, hp):
%     mono r0        3.461e-13  (S1 narrowband 625 nm, r0)
%     broadband 10%  6.645e-11  (S1 broadband)
%
%   Mono uses r0 fixed-G to match the defect-off narrowband-625 reference
%   methodology; broadband uses the 5-color band.
%
%   See also: ctb_chain (fpm_defect_lamD), ctb_dm_jacobian, ctb_efc,
%   ctb_efc_physics, ctb_dst_s1 (baselines).
    if nargin < 1 || isempty(N), N = 1024; end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);
    base   = {'fpm_kind','vortex','charge',6,'apodizer',false,'r_lyot_frac',0.80,'lam_m',625e-9};
    defcfg = {'fpm_defect_lamD',0.2,'fpm_incl_lamD',0.2,'fpm_incl_pos_lamD',5};
    REF_mono_off = 3.461e-13;    % S1 narrowband 625 nm, r0, hp (defect off)
    REF_bb_off   = 6.645e-11;    % S1 broadband 10%, hp (defect off)

    rep = fullfile(here,'ctb_dst_2c_report.txt');
    logf_(rep,'==== DST Lane 2c -- central defect (0.2 lamD) + inclusion (0.2 lamD @ 5) | N=%d | 625 nm | Lyot 0.80 | hp 3-8 | %s', ...
          N, datestr(now,31)); %#ok<DATST>

    % ---- MONO, defect ON (r0) ------------------------------------------
    Jm = ctb_dm_jacobian('model_size',N,'chain',[base defcfg], ...
            'inner_lamD',3,'outer_lamD',8,'region','halfplane', ...
            'tag',sprintf('c6L080_N%d_2c_mono_hp',N));
    om = ctb_efc('jac',Jm,'niter',20,'save',false);
    logf_(rep,'MONO  defect ON : static %.3e -> EFC(r0) %.3e | defect OFF ref %.3e | ratio %.2fx', ...
          om.c_before, om.c_after, REF_mono_off, om.c_after/REF_mono_off);

    % ---- BROADBAND 10%, defect ON --------------------------------------
    ob = ctb_efc_physics('band',true,'lfracs',[0.95 0.975 1.0 1.025 1.05], ...
            'region','halfplane','niter',20,'save',false, ...
            'chain',[{'model_size',N} base defcfg]);
    logf_(rep,'BBAND defect ON : static %.3e -> EFC     %.3e | defect OFF ref %.3e | ratio %.2fx', ...
          ob.c_before, ob.c_after, REF_bb_off, ob.c_after/REF_bb_off);

    % ---- gate verdict ---------------------------------------------------
    mono_ratio = om.c_after / REF_mono_off;
    bb_ratio   = ob.c_after / REF_bb_off;
    mono_ok = mono_ratio < 3;            % "nearly unchanged" (within ~3x)
    bb_ok   = bb_ratio   > 3;            % "degrades measurably"
    logf_(rep,'GATE 2c: mono ~unchanged (%.2fx, %s) AND broadband degrades (%.2fx, %s) -> %s', ...
          mono_ratio, tf_(mono_ok), bb_ratio, tf_(bb_ok), tf_(mono_ok && bb_ok));
    logf_(rep,'(defect in plant AND Jacobian; OFF ref = S1 same-config baselines = the recovery leg.)');

    OUT = struct('N',N,'mono_on',om.c_after,'mono_off',REF_mono_off, ...
        'bb_on',ob.c_after,'bb_off',REF_bb_off,'mono_ratio',mono_ratio, ...
        'bb_ratio',bb_ratio,'gate_pass',mono_ok && bb_ok);
    save(fullfile(here,sprintf('ctb_dst_2c_N%d.mat',N)),'-struct','OUT');
end

function s = tf_(b), if b, s='PASS'; else, s='FAIL'; end, end
function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep,'a'); fprintf(fid,'%s\n',s); fclose(fid);
    fprintf('%s\n', s);
end
