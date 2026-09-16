function [hd, reg] = tg96_samp(h, PL, xg, msk)
%TG96_SAMP  Sample a DETECTOR-frame map into the DM frame on this bench.
%   HD = TG96_SAMP(H, PL, XG, MSK) evaluates the detector map H at the
%   detector pixel each DM-grid point XG x XG maps to, using tg96_place's
%   OWN affine (PL.frm.Linv) and resolved field parity (PL.parity).
%
%   WHY NOT dmg_samp.  The shared dm_gauge_lib resampler expresses the
%   registration as an axis PERMUTATION plus per-axis signs plus one
%   isotropic scale (dxd_mm*mag).  That is the 8-parity family, and it
%   cannot represent a rotation that is not a multiple of 90 degrees --
%   which is exactly what the two folds of this reflective rig put into the
%   mapping (tg96_place's docstring makes the same point about
%   register_two_pokes / dmg_register).  Using dmg_samp here would
%   mis-register the lattice by the fold angle and would then be read as
%   the tail failing to read.  The lens rig is close enough to axis-aligned
%   that it would very nearly work there -- which is the trap: it would
%   look right on one leg of the two-leg test and be wrong on the other.
%
%   This is tg96_place's mapping evaluated on the DM GRID instead of on the
%   actuator lattice, through the same two helpers, so there is ONE
%   registration convention on this bench rather than two.
%
%   Points that land off the detector grid come back NaN; the caller
%   decides (dmg_act_fit zeroes them).
%
%   The second output REG MEASURES the claim above rather than asserting it.
%   dmg_samp's representable set is (1/(dxd_mm*mag)) times a SIGNED
%   PERMUTATION, so mag*Linv must itself be a signed permutation for the
%   shared resampler to be usable.  REG.rot_deg is the rotation in that
%   matrix folded into [0,90) -- 0 means axis-aligned and dmg_samp would do;
%   REG.perm_err is the relative distance to the nearest signed permutation,
%   i.e. what using dmg_samp here would cost; REG.aniso is the ratio of
%   singular values (1 = the isotropic scale dmg_samp assumes).
b = PL.anchor;                                  % [bx by tax tay]
[gxd, gyd] = meshgrid(xg, xg);
dmx = gxd - b(3);  dmy = gyd - b(4);            % DM-mm from the anchor's truth peak
L = PL.frm.Linv;
du = L(1,1)*dmx + L(1,2)*dmy;                   % detector-mm along u2
dv = L(2,1)*dmx + L(2,2)*dmy;                   % detector-mm along v2
[U, V] = tg96_apply_parity(du, dv, PL.dxd_mm, b(1), b(2), PL.parity);
hn = h;
if nargin >= 4 && ~isempty(msk), hn(~msk) = NaN; end
N = size(h, 1);
hd = interp2(1:size(h,2), (1:N).', hn, U, V, 'linear', NaN);
if nargout >= 2
    M = PL.mag * L;                       % dimensionless: a signed permutation iff dmg_samp fits
    [Us, Ss, Vs] = svd(M);
    R = Us*Vs.';  th = atan2d(R(2,1), R(1,1));
    sv = diag(Ss);
    best = inf;
    for pr = [1 2; 2 1].'                                  % the two permutations
        for s1 = [-1 1]
            for s2 = [-1 1]
                Pm = zeros(2);  Pm(1,pr(1)) = s1;  Pm(2,pr(2)) = s2;
                sc = trace(Pm.'*M)/2;                      % best isotropic scale
                best = min(best, norm(M - sc*Pm,'fro')/max(norm(M,'fro'),eps));
            end
        end
    end
    reg = struct('rot_deg', mod(th, 90), 'perm_err', best, ...
                 'aniso', max(sv)/max(min(sv),eps));
end
end
