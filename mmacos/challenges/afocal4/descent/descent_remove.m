function [S2, info] = descent_remove(P, S, k, mode, opts)
%DESCENT_REMOVE  Take one powered mirror out of a rung -- two ways.
%
%   [S2, info] = DESCENT_REMOVE(P, S, K, MODE) returns the design struct one
%   rung down, with powered mirror K removed by one of two mechanisms that
%   are NOT variants of each other:
%
%     'retain'  drive the mirror's power to ZERO and keep the element.  It is
%               now a FLAT -- a fold.  Dave's ruling 4: a retained flat is
%               not a mirror, so the rung's powered count drops by one, and
%               the flat is listed (count + role) but not counted.
%     'delete'  remove the element entirely and merge its two spacings.
%
%   WHY THE CHOICE IS NOT COSMETIC -- and this is the descent's own parity
%   law (README section 3) applied to its own move.  The vertex stations are
%   an alternating sum, so the closure's last spacing enters with sign
%   (-1)^(K-1) in the ELEMENT count K, not the powered count:
%
%     'retain' keeps the reflection, so K is unchanged and the parity of the
%              whole back end is unchanged;
%     'delete' removes a reflection, so K drops by one and the parity FLIPS
%              -- the same flip S4b measured when a mirror was ADDED.
%
%   Measured over a common grid, that is the difference between ~90 % of
%   closures putting the last powered mirror behind the primary and ~0.03 %
%   of them doing so.  So "N = 6 cannot be built" and "N = 6 cannot be built
%   WITHOUT KEEPING A FLAT" are different statements, and the ladder has to
%   report which one it measured.  Both modes are run at every rung for
%   exactly that reason.
%
%   WHICH MIRRORS MAY BE REMOVED.  The free ones, 3..K-2 by default:
%     * M1 and M2 are Rodgers' front end and are what this study changes
%       last (the S4b anchor);
%     * the last two elements' powers are CONSUMED by the closure -- they
%       are what makes the train afocal, 30x and pupil-correct -- so they
%       cannot be driven to zero without giving up the specification.  A
%       rung that wants them gone is a rung with a different N, which is
%       what the ladder is walking anyway.
%   'allow' widens it, deliberately and with the consequence stated.
%
%   THE MERGE IS A SUM, AND ONLY PARAXIALLY.  Deleting element K joins its
%   two spacings, t(k-1) + t(k), which is exact in the unfolded thin-lens
%   model the closure works in.  What it is NOT is a statement that the
%   emitted train is unchanged: one fewer reflection re-folds every station
%   after it, which is precisely the parity effect above.
%
%   Name-value:
%     'allow'  indices that may be removed (default 3 : K-2)
%     'flat_R' the radius a retained flat is given, m (1e12 -- phi = 2e-12,
%              flat to twelve figures against a 0.8 /m primary, and finite
%              so the emitter and the engine never see Inf)
%
%   Returns S2 (a DESCENT_CLOSE / DESCENT_BUILD spec) and INFO with .mode
%   .k .n_powered .n_flat .K_elements .parity_flips .why.
%
%   See also DESCENT_CLOSE, DESCENT_BUILD, DESCENT_LADDER.

    arguments
        P (1,1) struct %#ok<INUSA>
        S (1,1) struct
        k (1,1) double
        mode (1,:) char {mustBeMember(mode,{'retain','delete'})}
        opts.allow  (1,:) double = []
        opts.flat_R (1,1) double = 1e12
    end

    Kel = S.N;                       % elements the closure indexes
    nfree = Kel - 2;
    allow = opts.allow;
    if isempty(allow), allow = 3:nfree; end
    if ~ismember(k, allow)
        error('macos:design:descent_remove:index', ...
              ['mirror %d is not removable here (allowed: %s).  M1 and M2 ' ...
               'are the held front end; the last two elements'' powers are ' ...
               'CONSUMED by the closure and cannot go to zero without giving ' ...
               'up the afocal / magnification / pupil conditions.'], ...
              k, mat2str(allow));
    end

    nflat0 = 0;
    if isfield(S,'n_flat'), nflat0 = S.n_flat; end

    switch mode
    case 'retain'
        S2 = S;
        S2.R(k)      = opts.flat_R;
        S2.convex(k) = false;
        S2.n_flat    = nflat0 + 1;
        S2.flat_at   = [getf_(S,'flat_at',[]), k];
        info = struct('mode','retain', 'k',k, ...
            'n_powered', Kel - S2.n_flat, 'n_flat', S2.n_flat, ...
            'K_elements', Kel, 'parity_flips', false, ...
            'why', ['the element stays and still reflects, so the fold count ' ...
                    'and every downstream station keep their sign']);
    case 'delete'
        S2 = S;
        S2.N      = Kel - 1;
        S2.R(k)   = [];
        S2.convex(k) = [];
        S2.K(k)   = [];
        % merge the spacings either side of the removed element
        if k == 1
            S2.t(1) = [];
        else
            S2.t(k-1) = S.t(k-1) + S.t(k);
            S2.t(k)   = [];
        end
        S2.n_flat = nflat0;
        info = struct('mode','delete', 'k',k, ...
            'n_powered', (Kel-1) - nflat0, 'n_flat', nflat0, ...
            'K_elements', Kel-1, 'parity_flips', true, ...
            'why', ['a reflection is gone, so every station after it re-folds ' ...
                    'and the packaging parity FLIPS']);
    end
end

function v = getf_(s, f, d),  if isfield(s,f), v = s.(f); else, v = d; end,  end
