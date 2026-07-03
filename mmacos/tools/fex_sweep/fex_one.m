% FEX_ONE  Single-Rx FEX check, env-driven (FEXRX / FEXHASSTOP /
% FEXSTOPELT).  Run one MATLAB process per Rx so a loader crash or
% Fortran/MATLAB stdout interleave cannot contaminate the sweep log.
% Driven by run_fex_sweep.sh; not meant for interactive use.
%
% The engine FEX prints both radius legs:
%   ***** FEX: zp_iEm1 = <legacy iEm1->EP>  zp = <EP->next (default)>
% plus any FEX WARNING / AUTOSWITCH / TELECENTRIC / Rx-order lines --
% the sweep log is the compatibility record.
old_dir = cd(fullfile(getenv('HOME'),'dev','MACOS_resources','mmacos'));
mmacos_setup;
rx      = getenv('FEXRX');
hasstop = str2double(getenv('FEXHASSTOP'));
stopelt = str2double(getenv('FEXSTOPELT'));
macos.init(512);
macos.load_rx(rx);
if hasstop == 0
    macos.stop(stopelt);   % Rx lacks ApStop=; heuristic first optic
end
f = macos.fex(1);
fprintf('==FEXOK== rad=%.9e vpt=[%.6g %.6g %.6g]\n', f.rad, f.vpt);
cd(old_dir);
exit(0);
