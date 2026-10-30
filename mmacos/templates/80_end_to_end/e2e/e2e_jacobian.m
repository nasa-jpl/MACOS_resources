function J = e2e_jacobian(exdir, tag, r, e, ch, chain_pick, a0, MODEL, say, jac_opts)
%E2E_JACOBIAN  The e2e thread's tagged CTB Jacobian: reused from disk when it is this run's, else measured.
%   J = E2E_JACOBIAN(EXDIR, TAG, R, E, CH, CHAIN_PICK, A0, MODEL, SAY) returns
%   the ctb_dm_jacobian of the DM deck R (ctb_dm_rx's struct) on the chain CH
%   (built with CHAIN_PICK, the station map E among it) about the DM commands
%   A0 ({} = flat), stored as EXDIR/ctb_dm_jacobian_N<MODEL>_<TAG>.mat beside
%   its fingerprint truth <...>.fp.json.  A blob on disk is reused only when
%   (1) its fingerprint truth is beside it and holds (jac_fingerprint), (2) it
%   was measured on this deck with this station map, (3) its chain stamp is
%   CH's (the Lyot blocker among it) and (4) its linearization point is A0;
%   otherwise the stale .mat and its truth are removed and the Jacobian is
%   re-measured (ctb_dm_jacobian, central differences; about an hour per
%   Jacobian at model 512 on the coupled train).  The truth is committed, the
%   .mat is not (the e2e README's large-product policy): delete the .mat to
%   re-measure.  SAY is the stage's printer (@(varargin) fprintf(...)).
%   JAC_OPTS (optional cell, default {}) are further ctb_dm_jacobian options
%   ('nact', 'inner_lamD', 'outer_lamD' for a small Jacobian, the tests'); a
%   blob is reused only when its actuator count and dark zone match them.
%
%   Shared by s9_diffraction (measures) and s10_sensing (reuses) so the reuse
%   rule has one copy.
%
%   See also S9_DIFFRACTION, S10_SENSING, CTB_DM_JACOBIAN, JAC_FINGERPRINT.
    if nargin < 10, jac_opts = {}; end
    want = struct('nact', 32, 'inner_lamD', 3, 'outer_lamD', 15);          % ctb_dm_jacobian's defaults
    for k = 1:2:numel(jac_opts), if isfield(want, jac_opts{k}), want.(jac_opts{k}) = jac_opts{k+1}; end, end
    jp = fullfile(exdir, sprintf('ctb_dm_jacobian_N%d_%s.mat', MODEL, tag));  fpp = regexprep(jp, '\.mat$', '.fp.json');
    if isfile(jp)
        J = load(jp);
        try
            assert(isfile(fpp), 'no fingerprint truth beside it');
            [okfp, repfp] = jac_fingerprint('check', J, fpp);  assert(okfp, 'fingerprint: %s', repfp);
            assert(isfield(J, 'elt') && isequal(J.elt, e) && strcmp(J.rx, r.rx_out), 'another deck');
            assert(isfield(J, 'chain_opts') && isequal(J.chain_opts, ch.config), 'another chain stamp (the Lyot blocker among it)');
            assert(J.dm(1).nact == want.nact && J.inner_lamD == want.inner_lamD && J.outer_lamD == want.outer_lamD, 'another actuator count or dark zone');
            if isempty(a0), assert(isempty(J.a0) || all(cellfun(@(x) ~any(x), J.a0)), 'another linearization point');
            else, assert(~isempty(J.a0) && isequal(J.a0, a0), 'another linearization point'); end
            say('    Jacobian %s reused: %s (fingerprint holds, chain stamp and linearization point are this run''s; measured %s in %.1f min)', tag, jp, J.date, J.timing_s/60);
            return
        catch ME
            say('    cached Jacobian %s is not this run''s (%s): re-measuring; the stale .mat and its truth are removed', jp, ME.message);
            delete(jp);  if isfile(fpp), delete(fpp); end
        end
    end
    args = [{'dmrx', r, 'chain', chain_pick, 'tag', tag, 'outdir', exdir, 'model_size', MODEL, 'verbose', true}, jac_opts];
    if ~isempty(a0), args = [args, {'a0', a0}]; end
    J = ctb_dm_jacobian(args{:});
end
