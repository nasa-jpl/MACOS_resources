function out = oap_focus_probe(varargin)
%OAP_FOCUS_PROBE  Diagnose the ZWFS mask-seat focus on the OAP rig: ray spot rms
%   and focal-field peak/sum at the FocalMask, vs OAP2_SIDE and MASK_TRIM.  Tests
%   whether flipping OAP2_SIDE (two same-plane OAPs' coma adds vs cancels) makes a
%   diffraction-limited focus at the seat.  Gate (dmg_zwfs_gauge): peak/sum >= 0.01
%   and ray blur < 1 lambda F/D.  No files edited; calls zwfs_params + twyman_green.
%
%   Options (name/value; the defaults reproduce the original probe exactly):
%     'AOI'          fold angles to scan, both OAPs set to each (default
%                    [1 3 5 7 9]).  Give a 1x2 to pin OAP1 and OAP2 separately.
%     'SIDE'         [OAP1_SIDE OAP2_SIDE] (default [1 1])
%     'SRC_AT_FOCUS' feed the collimator at its TRUE focus (default false = the
%                    record).  The blur this probe measures is almost entirely
%                    the 25 mm zSource conjugate error, NOT the fold: it is
%                    LINEAR in the angle and vanishes when this is true
%                    (oap_conj_probe, tg_psi_dm96_oap/runs/conj).  Scan with it
%                    ON before concluding anything about a fold angle.
%     'D_RC_L2'      output-optics -> OAP2 standoff (default: zwfs_params')
%     'TRIM'         [lo hi] search bracket for MASK_TRIM (default [2 12]; use
%                    [-6 6] with SRC_AT_FOCUS, where the answer is near 0)
exdir = fileparts(mfilename('fullpath'));
run(fullfile(exdir,'..','..','..','mmacos_setup.m'));
addpath(fullfile(exdir,'..','zwfs_dm96'));
o = struct('AOI',[1 3 5 7 9], 'SIDE',[1 1], 'SRC_AT_FOCUS',false, ...
           'D_RC_L2',[], 'TRIM',[2 12]);
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end
P = zwfs_params();
P.MODEL = 512;  P.NGRID = 65;
if ~isempty(o.D_RC_L2), P.bench.D_RC_L2 = o.D_RC_L2; end
P.bench.SRC_AT_FOCUS = o.SRC_AT_FOCUS;
lamFD = P.LAM * P.bench.F2 / (2*P.bench.R_TO_AP);   % lambda F/D at the seat (mm)
fprintf('lambda F/D at the seat = %.4f mm (%.2f um); gate: peak/sum >= 0.01, blur < 1 lambdaF/D\n', lamFD, lamFD*1e3);
macos.init(P.MODEL);
if ~isfile(P.grid.flat_file), macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G)); end
skip = {};   % coat_* fields are stripped by prefix below (runner knobs, not builder args)
% is the blur reducible by the fold angle?  fine-optimize MASK_TRIM per AOI.
fprintf('sides %+d/%+d; collimator fed at its %s\n', o.SIDE(1), o.SIDE(2), ...
        iff_(o.SRC_AT_FOCUS,'TRUE focus','record conjugate (25 mm inside)'));
fprintf('%-8s | %10s %10s | %s\n','OAP2_AOI','min blur um','blur/lFD','(best MASK_TRIM)');
% a numeric AOI list scans each angle with BOTH OAPs at it (the original
% probe); a CELL list scans each entry, so {[20 25]} pins OAP1 20 / OAP2 25
A = o.AOI;  if ~iscell(A), A = num2cell(A); end
R = nan(numel(A), 3);
for k = 1:numel(A)
  aoi = A{k};
  f = @(mt) probe_blur_(P, skip, aoi, mt, o);
  [mtb, bb] = fminbnd(f, o.TRIM(1), o.TRIM(2), optimset('TolX',0.02,'MaxFunEvals',30));
  fprintf('%-8s | %10.2f %10.3f | mt=%.2f\n', mat2str(aoi), bb*1e3, bb/lamFD, mtb);
  R(k,:) = [aoi(1), bb/lamFD, mtb];
end
out = struct('R', R, 'lamFD', lamFD, 'o', o);
end

function s = iff_(c,a,b)
if c, s = a; else, s = b; end
end

function blur = probe_blur_(P, skip, aoi, mt, o)
  P.bench.optics='oap';
  if numel(aoi) == 2, P.bench.OAP1_AOI=aoi(1); P.bench.OAP2_AOI=aoi(2);
  else,               P.bench.OAP1_AOI=aoi;    P.bench.OAP2_AOI=aoi;    end
  P.bench.OAP1_SIDE=o.SIDE(1); P.bench.OAP2_SIDE=o.SIDE(2); P.bench.MASK_TRIM=mt;
  bf = fieldnames(P.bench);  ba = {};
  for i=1:numel(bf), if ~any(strcmp(bf{i},skip)) && ~strncmp(bf{i},'coat_',5), ba(end+1:end+2)={bf{i},P.bench.(bf{i})}; end, end
  try
    G = macos.design.twyman_green(ba{:}, 'ngridpts',P.NGRID, ...
        'to_grid_file',P.grid.flat_file,'to_grid_n',P.grid.N_G,'to_grid_dx',P.grid.DX_G);
    G.bt.emit('probe_test.in');  macos.load_rx('probe_test.in');
    iM = G.T.iMASK;  s = macos.trace(iM);  ri = macos.get_ray_info(s.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);  pos = ri.pos(:,ok);
    c = mean(pos,2);  blur = sqrt(mean(sum((pos-c).^2,1)));
  catch
    blur = 1;   % 1 mm penalty on a failed build
  end
delete('probe_test.in'); if isfile('probe_ref.in'), delete('probe_ref.in'); end
end
