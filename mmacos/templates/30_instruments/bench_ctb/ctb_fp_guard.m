function do_write = ctb_fp_guard(fp_path, S, replace)
%CTB_FP_GUARD  Refuse to overwrite a Jacobian fingerprint truth.
%   DO_WRITE = CTB_FP_GUARD(FP_PATH, S, REPLACE) decides whether a measure
%   that has just built product S may write its fingerprint to FP_PATH.
%
%     no file at FP_PATH       -> true  (write it)
%     S matches the file       -> false (leave the truth as it is)
%     S differs, REPLACE true  -> true  (re-make the truth deliberately)
%     S differs, REPLACE false -> error 'ctb_dm_jacobian:fpTruth'
%
%   The .fp.json files beside the CTB templates are the committed truths
%   the Jacobian caches are checked against (ctb_efc's cache load).  A
%   measure that silently replaced one would make any cache pass.  The
%   function cannot tell a committed file from a local one without git,
%   so every existing fingerprint is protected.  Dave 2026-10-28,
%   PLAN_CONSOLIDATION 1.2 (bbbb): the measure refuses to overwrite.
%
%   See also: ctb_dm_jacobian, jac_fingerprint, ctb_efc.
    if nargin < 3, replace = false; end
    if ~isfile(fp_path)
        do_write = true;
        return
    end
    [ok, rep] = jac_fingerprint('check', S, fp_path);
    if ok
        do_write = false;
    elseif replace
        do_write = true;
    else
        error('ctb_dm_jacobian:fpTruth', ...
            ['refusing to overwrite the fingerprint truth %s: the new ' ...
             'measurement differs from it (%s).  The .mat was saved; ' ...
             'ctb_efc will refuse it against this truth.  Re-make the ' ...
             'truth deliberately with ''replace_fp'', true, or measure ' ...
             'under another ''tag'' or ''outdir''.'], fp_path, rep);
    end
end
