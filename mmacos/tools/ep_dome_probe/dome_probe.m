% DOME_PROBE  Segment-tilt sensitivity read at three surfaces on the e2e6m
% (macos/REPORT_ep_dome_review.md, 2026-09-08).  Run from anywhere:
%   matlab -batch "run('<mmacos>/tools/ep_dome_probe/dome_probe.m')"
% imaging deck: OAPim (nElt-1, powered), the FocalPlane, and a properly
% placed exit-pupil reference sphere (add_pupil pair + FEX).  Plus a global
% field tilt, which an exit-pupil read must show as a tilt and a focal-plane
% read is blind to.  Writes dome_probe.mat + a text table.
here = fileparts(mfilename('fullpath'));
MM = fileparts(fileparts(here));                 % mmacos root
run(fullfile(MM, 'mmacos_setup.m'));
S = [fullfile(here, 'out') filesep];  if ~exist(S, 'dir'), mkdir(S); end
E1 = [fullfile(MM, 'templates', '80_end_to_end', 'e2e6m') filesep];
if ~exist([S 's3_pupil.in'], 'file')
    system(sprintf('python3 %s %s %s', fullfile(here, 'make_pupil_deck.py'), [E1 's3_imager_full.in'], [S 's3_pupil.in']));
end
bare = [E1 's3_imager_full.in'];  pup = [S 's3_pupil.in'];
alpha = 1e-6;                      % segment tilt (rad)
ftilt = 1e-6;                      % global field tilt (rad)
segs  = [1 8];
macos.init(256);
R = struct();
% ---------- bare deck: reads at OAPim (25) and Imager FP (26) ----------
macos.load_rx(bare);  nE = macos.num_elt();  assert(nE == 26);
W0_oap = rd_(25);  W0_fp = rd_(26);
for s = segs
    macos.load_rx(bare);
    macos.perturb(s, 'rotation', [alpha;0;0]);
    R.(sprintf('seg%d',s)).oap = rd_(25) - W0_oap;
    R.(sprintf('seg%d',s)).fp  = rd_(26) - W0_fp;
    macos.load_rx(bare);
    macos.perturb(s, 'translation', [0;0;1e-7]);          % piston: footprint mask
    R.(sprintf('seg%d',s)).fp_piston = rd_(26) - W0_fp;
end
macos.load_rx(bare);
macos.set_src_fov('src_dir', [sin(ftilt); 0; cos(ftilt)]);
R.tilt.oap = rd_(25) - W0_oap;  R.tilt.fp = rd_(26) - W0_fp;
% ---------- pupil deck: FEX places the sphere at 27, read there ----------
macos.load_rx(pup);  nE = macos.num_elt();  assert(nE == 28);
fr = 2.908882e-4;
macos.set_src_fov('src_dir', [sin(fr); 0; cos(fr)]);
macos.stop(1);                                   % as add_pupil does (Seg1)
macos.trace(28);
f = macos.fex(1);
fprintf('FEX placed the EP sphere: rad %.6f  vpt [%.5f %.5f %.5f]\n', f.rad, f.vpt);
macos.set_src_fov('src_dir', [0;0;1]);
% freeze the placed pupil: save the deck so every reload carries it
pupF = [S 's3_pupil_fexed.in'];  macos.save_rx(pupF);
macos.load_rx(pupF);
W0_ep = rd_(27);  W0_fp28 = rd_(28);
for s = segs
    macos.load_rx(pupF);
    macos.perturb(s, 'rotation', [alpha;0;0]);
    R.(sprintf('seg%d',s)).ep   = rd_(27) - W0_ep;
    R.(sprintf('seg%d',s)).fp28 = rd_(28) - W0_fp28;
    macos.load_rx(pupF);
    macos.perturb(s, 'translation', [0;0;1e-7]);
    R.(sprintf('seg%d',s)).ep_piston = rd_(27) - W0_ep;
end
macos.load_rx(pupF);
macos.set_src_fov('src_dir', [sin(ftilt); 0; cos(ftilt)]);
R.tilt.ep = rd_(27) - W0_ep;  R.tilt.fp28 = rd_(28) - W0_fp28;
% ---------- metrics ----------
valid = @(W) isfinite(W) & (W ~= 0);
fid = fopen([S 'dome_probe.txt'],'w');
pr = @(varargin) fprintf(fid, varargin{:});
pr('alpha = %.1e rad segment Rx tilt; field tilt %.1e rad; maps in metres\n\n', alpha, ftilt);
pr('%-6s %-6s %12s %12s %12s %9s %9s %9s\n','seg','read','rms','max','min','|max/min|','ramp%','corr_ep');
for s = segs
    k = sprintf('seg%d',s);  D = R.(k);
    F = valid(D.ep_piston) & abs(D.ep_piston) > 0.5*max(abs(D.ep_piston(:)));   % segment footprint (EP)
    [ii,jj] = find(F);  A = [ones(size(ii)) ii jj];
    ref = D.ep(F);
    for nm = {'oap','fp','fp28','ep'}
        W = D.(nm{1});  v = W(F);
        c = A \ v;  ramp = A(:,2:3)*c(2:3);
        rampfrac = 100*norm(ramp - mean(ramp))/max(norm(v - mean(v)),1e-30);
        c2 = corrcoef(v, ref);  cc = c2(1,2);
        pr('%-6s %-6s %12.3e %12.3e %12.3e %9.3f %9.1f %9.3f\n', k, nm{1}, ...
           rms_(v), max(v), min(v), abs(max(v)/min(v)), rampfrac, cc);
    end
    pr('   footprint pixels %d; whole-map pos/neg extent: oap %.3f  fp %.3f  ep %.3f\n', nnz(F), ...
       abs(max(D.oap(valid(D.oap)))/min(D.oap(valid(D.oap)))), ...
       abs(max(D.fp(valid(D.fp)))/min(D.fp(valid(D.fp)))), ...
       abs(max(D.ep(valid(D.ep)))/min(D.ep(valid(D.ep)))));
end
pr('\nGLOBAL FIELD TILT %.1e rad -- rms over the pupil: oap %.3e  fp %.3e  fp28 %.3e  ep %.3e\n', ftilt, ...
   rms_(R.tilt.oap(valid(R.tilt.oap))), rms_(R.tilt.fp(valid(R.tilt.fp))), ...
   rms_(R.tilt.fp28(valid(R.tilt.fp28))), rms_(R.tilt.ep(valid(R.tilt.ep))));
pr('  expected EP tilt rms ~ ftilt*D/sqrt(12)*... (D=6 m): %.3e\n', ftilt*6/sqrt(12));
fclose(fid);
save([S 'dome_probe.mat'], 'R', 'W0_oap','W0_fp','W0_ep','W0_fp28','f');
type([S 'dome_probe.txt']);
exit(0);

function W = rd_(e), macos.trace(e); W = macos.opd(); end
function r = rms_(v), r = sqrt(mean(v(:).^2)); end
