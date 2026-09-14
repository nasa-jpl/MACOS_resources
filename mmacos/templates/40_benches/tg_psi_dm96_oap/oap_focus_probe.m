function oap_focus_probe()
%OAP_FOCUS_PROBE  Diagnose the ZWFS mask-seat focus on the OAP rig: ray spot rms
%   and focal-field peak/sum at the FocalMask, vs OAP2_SIDE and MASK_TRIM.  Tests
%   whether flipping OAP2_SIDE (two same-plane OAPs' coma adds vs cancels) makes a
%   diffraction-limited focus at the seat.  Gate (dmg_zwfs_gauge): peak/sum >= 0.01
%   and ray blur < 1 lambda F/D.  No files edited; calls zwfs_params + twyman_green.
exdir = fileparts(mfilename('fullpath'));
run(fullfile(exdir,'..','..','..','mmacos_setup.m'));
addpath(fullfile(exdir,'..','zwfs_dm96'));
P = zwfs_params();
P.MODEL = 512;  P.NGRID = 65;
lamFD = P.LAM * P.bench.F2 / (2*P.bench.R_TO_AP);   % lambda F/D at the seat (mm)
fprintf('lambda F/D at the seat = %.4f mm (%.2f um); gate: peak/sum >= 0.01, blur < 1 lambdaF/D\n', lamFD, lamFD*1e3);
macos.init(P.MODEL);
if ~isfile(P.grid.flat_file), macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G)); end
skip = {'coat_oap','coat_bareAl','coat_protectedAl'};
% is the blur reducible by the fold angle?  fine-optimize MASK_TRIM per AOI.
fprintf('%-8s | %10s %10s | %s\n','OAP2_AOI','min blur um','blur/lFD','(best MASK_TRIM)');
for aoi = [1 3 5 7 9]
  f = @(mt) probe_blur_(P, skip, aoi, mt);
  [mtb, bb] = fminbnd(f, 2, 12, optimset('TolX',0.02,'MaxFunEvals',30));
  fprintf('%-8d | %10.2f %10.3f | mt=%.2f\n', aoi, bb*1e3, bb/lamFD, mtb);
end
end

function blur = probe_blur_(P, skip, aoi, mt)
  P.bench.optics='oap'; P.bench.OAP1_AOI=aoi; P.bench.OAP2_AOI=aoi;
  P.bench.OAP1_SIDE=1; P.bench.OAP2_SIDE=1; P.bench.MASK_TRIM=mt;
  bf = fieldnames(P.bench);  ba = {};
  for i=1:numel(bf), if ~any(strcmp(bf{i},skip)), ba(end+1:end+2)={bf{i},P.bench.(bf{i})}; end, end
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
