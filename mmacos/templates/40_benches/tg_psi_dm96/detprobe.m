run(fullfile('..','..','..','mmacos_setup.m')); macos.init(1024);
macos.load_rx('tg96_test.in');
macos.trace();
I = macos.intensity(23);  m = I > 0.1*max(I(:));
dxp = macos.dx_at(23,'mm');
[rr,cc] = find(m);  npx = max(cc)-min(cc)+1;
fprintf('DETPROBE: dx %.5f mm, beam %d px across, detector Nyquist %.1f cyc/pupil\n', dxp, npx, npx/2);
