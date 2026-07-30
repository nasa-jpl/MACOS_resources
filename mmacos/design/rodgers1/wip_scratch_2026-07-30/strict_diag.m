function strict_diag()
%STRICT_DIAG  Verify set_xp-at-detector-chief gives focus-in, tilt-out WFE
%   (NOT the FEX tautology floor).
    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    P = rodgers_common();  P.EPD_mm = 4060;  lam_nm = P.lambda_m*1e9;
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_field_bias(P.offset_deg*60);  t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    t.add_pupil(numel(t.spec.elt));
    pu = t.spec.pupil;  iEP = pu.ep_elt;  iFP = pu.fp_elt;
    EPv = t.spec.elt(iEP).Vpt(:);
    h = deg2rad(6/60);
    fprintf('EP elt %d, det FP elt %d, EP vpt=[%.4f %.4f %.4f]\n', iEP, iFP, EPv);
    for F = {[0 0],[0 h],[h 0],[h h]}
        f = F{1};
        t.trace_at_field(f);
        sfp = macos.trace(iFP);  rf = macos.get_ray_info(sfp.nRays);
        ok = rf.ok_trace & rf.ok_pass;  Pp = rf.pos; Pp(:,~ok)=NaN;
        c = mean(Pp(:,ok),2); d2=sum((Pp-c).^2,1); d2(~ok)=inf; [~,ic]=min(d2);
        Cf = rf.pos(:,ic);
        d = Cf-EPv; R=norm(d);
        macos.set_xp(EPv, d/R, -R);
        se = macos.trace(iEP);  W=macos.opd(); v=W(isfinite(W)&W~=0&abs(W)<1e30);
        rms = std(v-mean(v))*lam_nm; ptp=(max(v)-min(v))*lam_nm;
        fprintf('  (%+.0f,%+.0f)'': Cf=[%.4f %.4f %.4f] R=%.3fmm  n=%d rms=%.4f nm ptp=%.4f nm\n', ...
                rad2deg(f(1))*60,rad2deg(f(2))*60, Cf, R*1e3, numel(v), rms, ptp);
    end
    t.trace_at_field([]);
end
