function sel = require_elts(discovered, elts, family, reason_fn)
%REQUIRE_ELTS  Apply an explicit element request to an eligibility set, LOUDLY.
%   sel = require_elts(DISCOVERED, ELTS, FAMILY, REASON_FN)
%
%   ELTS empty  -> sel = DISCOVERED (the auto-discovery path, unchanged).
%   ELTS given  -> sel = intersect(DISCOVERED, ELTS); if ANY requested id
%   is not served, error macos:channels:eltNotEligible naming every dropped
%   id with REASON_FN(id) -> char.
%
%   Contract (Dave's ruling, 2026-09-05; macos/BRIEF_luis_round3.md): an
%   element the caller EXPLICITLY requested must never vanish silently.
%   Six channel builders used to intersect the user's 'elts' against their
%   discovered set with no report -- an NSReflector passed to dw_dsurf
%   simply disappeared two bridge levels down (Luis).  Auto-discovery may
%   filter silently; an explicit request either is served or explains
%   itself.
%
%   FAMILY is the channel-family name for the message ('surf', 'zernike',
%   'grid', ...).  REASON_FN receives one dropped element id and returns
%   the reason it cannot be served.
arguments
    discovered (:,1) double
    elts       (:,1) double
    family     (1,:) char
    reason_fn  (1,1) function_handle
end
if isempty(elts)
    sel = discovered;
    return;
end
sel = intersect(discovered, elts);
dropped = setdiff(elts, discovered);
if isempty(dropped)
    return;
end
msg = cell(numel(dropped), 1);
for k = 1:numel(dropped)
    msg{k} = sprintf('  elt %d: %s', dropped(k), reason_fn(dropped(k)));
end
error('macos:channels:eltNotEligible', ...
    ['%s channels: %d of the explicitly requested ''elts'' cannot be ' ...
     'served:\n%s'], family, numel(dropped), strjoin(msg, newline));
end
