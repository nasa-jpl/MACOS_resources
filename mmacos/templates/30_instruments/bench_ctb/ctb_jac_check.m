function ctb_jac_check(J, chain, src)
%CTB_JAC_CHECK  Refuse a cached Jacobian that mismatches the requested chain.
%   CTB_JAC_CHECK(J, CHAIN, SRC) compares the chain configuration STORED
%   in a loaded Jacobian struct (the `chain_opts` name-value cell every
%   measuring driver stamps into its save) against the chain the caller
%   is about to control with, and errors -- naming the file and every
%   differing field -- on mismatch.
%
%   Why this exists: the Jacobian caches are keyed by FILENAME (N,
%   charge, tag), which does not encode the full mask geometry.  Change
%   the Lyot fraction and an existing cache silently serves the wrong G;
%   the loop then fights a mismatched model, converges slowly, and the
%   numbers look plausible.  The stored config is the authority, not the
%   file name.
%
%   J      loaded Jacobian struct (any struct; checked iff it carries
%          `chain_opts`)
%   chain  the requested config: a name-value cell (ctb_chain option
%          subset) or a ctb_chain handle's `.config`.  Only keys present
%          in BOTH are compared -- a partial request checks partially.
%   src    label for messages (usually the .mat path)
%
%   Legacy caches without `chain_opts` draw one warning (measured before
%   the stamp existed) -- regen to silence.
%
%   See also: ctb_dm_jacobian, ctb_efc_physics, ctb_vvc, ctb_study.
    if ~isfield(J, 'chain_opts')
        warning('ctb_jac_check:legacy', ...
            '%s carries no chain_opts stamp (legacy cache) -- config unverifiable; regen to stamp it', src);
        return
    end
    a = nv2struct_(J.chain_opts);
    b = nv2struct_(chain);
    ka = fieldnames(a);
    bad = {};
    for i = 1:numel(ka)
        k = ka{i};
        if isfield(b, k) && ~isequal(a.(k), b.(k))
            bad{end+1} = sprintf('%s: cached %s vs requested %s', ...
                k, fmt_(a.(k)), fmt_(b.(k)));                  %#ok<AGROW>
        end
    end
    if ~isempty(bad)
        error('ctb_jac_check:mismatch', ...
            ['%s was measured on a DIFFERENT chain than requested:\n  %s\n' ...
             'Delete it or use a distinct tag (ctb_study derives tags from the config).'], ...
            src, strjoin(bad, sprintf('\n  ')));
    end
end

function s = nv2struct_(c)
    if isstruct(c), s = c; return; end
    s = struct();
    for i = 1:2:numel(c)-1
        s.(c{i}) = c{i+1};
    end
end

function t = fmt_(v)
    if ischar(v) || isstring(v)
        t = char(v);
    elseif islogical(v)
        w = {'false','true'};  t = w{v+1};
    else
        t = mat2str(v, 6);
    end
end
