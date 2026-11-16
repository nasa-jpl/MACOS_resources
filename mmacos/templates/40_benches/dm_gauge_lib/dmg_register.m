function [P, sgn, c_best, c_next, ccs] = dmg_register(hB, Mb, R)
%DMG_REGISTER  Parity + measurement sign from ONE OFF-CENTER poke (DOF 3+4).
%   [P, sgn, c_best, c_next, ccs] = dmg_register(HB, MB, R) samples the
%   measured off-center-poke map HB into the DM frame under each of the
%   8 flip/transpose candidates and overlaps with the truth MB.  The
%   SELECTION metric is the gate; thresholds are the CALLER's (IFO
%   |corr| >= 0.8; ZWFS >= 0.4 -- the ringed kernel caps raw correlation
%   near 0.5 by physics; separation >= 0.3 both).  Parity and sign are
%   deck-AND-enumeration dependent -- never inherited across decks.
%   R as in dmg_samp (R.P ignored here).  Extracted verbatim from
%   tg96_eprime/zwfs_s2 @ 10cf593 (incl. the NaN->0 off-support guard).
PAR = {[1 2 1 1],[1 2 -1 1],[1 2 1 -1],[1 2 -1 -1], ...
       [2 1 1 1],[2 1 -1 1],[2 1 1 -1],[2 1 -1 -1]};
ccs = zeros(1,8);
for p = 1:8
    Rp = R;  Rp.P = PAR{p};
    hBd = dmg_samp(hB, Rp);
    ok = ~isnan(hBd) & (abs(Mb) > 0);
    if nnz(ok) < 50, ccs(p) = 0; continue; end
    cm = corrcoef(hBd(ok), Mb(ok));  c = cm(1,2);
    if isnan(c), c = 0; end   % candidate maps the truth region off-support
    ccs(p) = c;
end
[~, pbest] = max(abs(ccs));
srt = sort(abs(ccs), 'descend');
P = PAR{pbest};  sgn = sign(ccs(pbest));
c_best = srt(1);  c_next = srt(2);
end
