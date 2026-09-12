function P = pdi_params()
%PDI_PARAMS  The point-diffraction readings' parameter sheet: zwfs_params with
%   the P / PF record settings.  The point-diffraction interferometer readings
%   (P = stepped pinhole at the FocalMask, common path; PF = the P/SRI of Dube
%   et al. 2024, a waveguide reference with a photonic phase shifter) live in
%   the SAME runner as the Zernike sensor -- same bench, same DM truth, same
%   battery, same loop -- so this sheet is zwfs_params with the readings,
%   noise and loop lists set to the PDI record and the camera-drift loop kind
%   added.  Every P.pdi knob is documented in zwfs_params.m.
%       P = pdi_params;  out = pdi_run(P);                 % the record
%       pdi_run('pdi.DIA_LAMD', 1.0, 'stages', {'bench','battery','figs'})
%       ./zwfs_batch.sh TAG "pdi_params, 'stages',{'bench','loop','figs'}"
P = zwfs_params();
P.tag = 'pdi';
P.readings = {'S', 'V', 'P', 'PF'};              % the stepped and vector Zernike readings beside the two PDI forms
P.noise.readings = {'S', 'V', 'P', 'PF'};
P.loop.readings = {'S', 'V', 'P', 'PF'};
P.loop.drifts = {'walk', 'thermal', 'cam'};      % + the camera 1/f drift (P.loop.cam_walk)
end
