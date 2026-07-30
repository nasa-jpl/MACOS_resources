function recon()
here=fileparts(mfilename('fullpath')); run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P=rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build(); t.align_focal_plane('grid',5,'span_arcmin',6);
nE=numel(t.spec.elt); by=t.spec.field_bias;
% pick the corner box field
h=deg2rad(6/60); off=[h h]; Fabs=[off(1), by+off(2)];
% (1) realize_apertures global at this single field
sg=t.realize_apertures('fields',Fabs,'quiet',true,'metric','global');
fprintf('realize global @corner: %.4f nm\n', sg.wfe*lam_nm);
% (2) manual trace_at_field(off) + trace(nE) + std(opd)
t.trace_at_field(off); s=macos.trace(nE); W=macos.opd(); v=W(isfinite(W)&W~=0&abs(W)<1e30);
fprintf('manual trace_at_field(off) trace(nE) std(opd): %.4f nm  rmsWFE=%.4f  n=%d\n', std(v)*lam_nm, s.rmsWFE*lam_nm, numel(v));
% (3) manual but set trace_field = ABSOLUTE (like realize does F(j,:)=Fabs)
t.trace_at_field([]);  % reset
% realize sets obj.spec.trace_field = F(j,:) directly = Fabs (absolute, NOT offset+bias)
% trace_at_field adds bias: trace_field=[off(1), by+off(2)] = Fabs. SAME. so why differ?
% check what trace_at_field stored vs what realize stores:
t.trace_at_field(off);
fprintf('after trace_at_field(off): (cannot read spec.trace_field, read-only)\n');
% Try: does realize maybe NOT align? re-run realize WITHOUT prior align effect by re-aligning
t.align_focal_plane('grid',5,'span_arcmin',6);
sg2=t.realize_apertures('fields',Fabs,'quiet',true,'metric','global'); 
t.trace_at_field(off); s2=macos.trace(nE); W2=macos.opd(); v2=W2(isfinite(W2)&W2~=0&abs(W2)<1e30);
fprintf('after realign: realize=%.4f nm | manual=%.4f nm\n', sg2.wfe*lam_nm, std(v2)*lam_nm);
t.trace_at_field([]);
end
