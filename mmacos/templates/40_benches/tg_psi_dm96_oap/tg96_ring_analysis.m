function tg96_ring_analysis(matfile)
%TG96_RING_ANALYSIS  Where does a descent's residual actually live?
%   tg96_ring_analysis('runs/samp512/samp512.mat')
%
%   Answered the 2026-09-18/19 redo-bench descent stall in ~20 min of CPU on
%   data already on disk, after ~12 h of run-per-hypothesis ladders had killed
%   four named causes (capture range, uniform over-damping, matrix staleness,
%   detector sampling).  Reach for this BEFORE the next 4 h ladder.
%
%   The finding it produced: the stall is the OUTERMOST ACTUATOR RINGS.  On
%   samp512's 100 nm row, ring 1 (180 act, 2.4% of lit) carries 95-98% of the
%   variance and is left 0.81% uncorrected against the interior's 0.0046% --
%   176x.  The interior is at the photon floor (~3 pm) in every row.  Cause:
%   the beam overfills the DM (lit radius 49.0 of a 47.5 array half-width), so
%   the outer ring's influence functions are truncated, its J columns are weak,
%   and lambda = matrix_lam*median(diag(JtJ)) -- one GLOBAL scalar on the
%   MEDIAN column energy -- damps them as hard as strong interior ones.
%
%   NOTE erosion uses FALSE PADDING, not circshift: this lit set reaches the
%   array edge, where a wrapping erosion joins the opposite side of the array.
S = load(matfile); LO = S.out.loop; lit0 = LO.lit; A0 = LO.A0; R = LO.res;
D = ring_depth_(lit0);                     % 1 = outermost lit ring
a0rms = sqrt(mean(A0(lit0).^2)); u = A0/a0rms;
fprintf('\n%s : lit %d, max depth %d rings, A0 rms %.3f nm\n', ...
    matfile, nnz(lit0), max(D(:)), 1e6*a0rms);
for i = 1:numel(R)
  L = R(i).L; r = L.r_final; e0 = (R(i).start - a0rms)*u;
  fprintf('\n-- start %.0f nm, recal %g : r(K) %.3f pm, rho %.3f\n', ...
      1e6*R(i).start, R(i).amp, 1e9*L.rms(end), L.rho);
  fprintf('   per-ring rms (pm):');
  for d = 1:6, m = (D==d); if nnz(m)>5, fprintf('  r%d %.1f', d, 1e9*std(r(m))); end, end
  fprintf('\n   rms over lit eroded by k rings (FALSE-padded):\n');
  lit = lit0;
  for k = 0:4
    if k>0, lit = erode_(lit); end
    fprintf('     k=%d: %8.3f pm over %d act\n', k, 1e9*std(r(lit)), nnz(lit));
  end
  if abs(R(i).start - a0rms) > 1e-9
    fprintf('   correction efficiency (%% of what each ring was ASKED to move, left at cycle K):\n     ');
    for d = [1 2 3 4]
      m = (D==d); if nnz(m)<5, continue; end
      fprintf('r%d %.3f%%  ', d, 100*sqrt(mean(r(m).^2))/sqrt(mean(e0(m).^2)));
    end
    m = lit0 & D>3;
    fprintf('interior %.4f%%\n', 100*sqrt(mean(r(m).^2))/sqrt(mean(e0(m).^2)));
  end
  m1 = (D==1); v1 = sum((r(m1)-mean(r(lit0))).^2); vt = sum((r(lit0)-mean(r(lit0))).^2);
  fprintf('   ring 1: %d act (%.1f%% of lit) carrying %.2f%% of the variance\n', ...
      nnz(m1), 100*nnz(m1)/nnz(lit0), 100*v1/vt);
end
end

function D = ring_depth_(lit)
D = zeros(size(lit)); cur = lit; k = 0;
while any(cur(:)), k = k+1; nb = erode_(cur); D(cur & ~nb) = k; cur = nb; end
end

function e = erode_(m)
Lp = false(size(m)+2); Lp(2:end-1,2:end-1) = m;
e = Lp(2:end-1,2:end-1) & Lp(1:end-2,2:end-1) & Lp(3:end,2:end-1) ...
                        & Lp(2:end-1,1:end-2) & Lp(2:end-1,3:end);
end
