function fex_in_loop_check()
%FEX_IN_LOOP_CHECK  Is CALIB's inner loop actually running FEX per field?
%
%   The exit-pupil merit is only the strict metric if the reference sphere
%   is RE-DERIVED at every field.  This measures both sides on the same
%   9-field optimisation box: the OPD at the ExitPupil with the STALE
%   add_pupil sphere, and the OPD there after a per-field macos.fex.
%
%   Measured (EPD 4060, stage-2 optics + add_pupil):
%     no FEX : 1.8e-3 .. 2.6e-3 m off-axis   (image-displacement tilt on a
%              sphere whose CoC is stuck at the on-axis image)
%     + FEX  : 1.1e-7 .. 4.3e-7 m            (= the strict metric)
%   i.e. four orders of magnitude apart, and the on-axis field is the only
%   one where they agree -- which is exactly the tell.
%
%   CALIB's own inner merit, logged during a real solve with OptFEX= Yes,
%   reads 5.5e-3 .. 1.3e-2 m -- the NO-FEX column.  So the deck cannot turn
%   FEX on: msmacosio.inc:327-329 parses OptFEX but has ONLY the
%   "If (LCMP(VALUE,'N',1)) LOptIfFEX=.FALSE." branch -- it is
%   write-only-false.  See PACKET Addendum 5.
    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    addpath(here);
P = rodgers_common(); P.EPD_mm=4060;
t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
        'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build();
t.align_focal_plane('grid',5,'span_arcmin',6);
t.add_pupil(); nE = numel(t.spec.elt);
deck=[tempname '.in']; t.save(deck);
txt = regexprep(fileread(deck),'(ApType=\s*)\S+','$1None');
g=@(k)sscanf(strrep(regexp(txt,[k '=\s*([^\n]*)'],'tokens','once'),'D','E'),'%f',3);
tk=regexp(txt,'ChfRayDir=\s*([^\n]*)','tokens','once'); cdir=sscanf(tk{1},'%f',3);
tk=regexp(txt,'ChfRayPos=\s*([^\n]*)','tokens','once'); cpos=sscanf(tk{1},'%f',3);
tk=regexp(txt,'ApStop=\s*([^\n]*)','tokens','once');    apst=sscanf(tk{1},'%f',3);
stand=dot(apst-cpos,cdir); bx0=asin(cdir(1)); by0=asin(cdir(2));
F = macos.design.field_grid(6,3,'units','arcmin');   % the optimisation box
tmp=[tempname '.in']; macos.init(P.model_size);
fprintf('\n#### %8s %8s %16s %16s\n','thx','thy','EP OPD no-FEX (m)','EP OPD +FEX (m)');
for k=1:size(F,1)
  bx=bx0+F(k,1); by=by0+F(k,2);
  d=[sin(bx);sin(by);sqrt(max(0,1-sin(bx)^2-sin(by)^2))]; p=apst-stand*d;
  v3=@(v)sprintf('%.16E  %.16E  %.16E',v(1),v(2),v(3));
  s=regexprep(txt,'(ChfRayDir=\s*)[^\n]*',['$1' v3(d)]);
  s=regexprep(s,  '(ChfRayPos=\s*)[^\n]*',['$1' v3(p)]);
  fid=fopen(tmp,'w'); fprintf(fid,'%s',s); fclose(fid);
  macos.load_rx(tmp); macos.stop(1);
  a = macos.trace(nE-1);            % stale add_pupil sphere
  macos.fex(1);
  b = macos.trace(nE-1);            % after per-field FEX
  fprintf('#### %8.2f %8.2f %16.6e %16.6e\n', F(k,1)*180/pi*60, F(k,2)*180/pi*60, a.rmsWFE, b.rmsWFE);
end
delete(tmp); delete(deck);
end
