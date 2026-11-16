function diag_stop()
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P=rodgers_common();
t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',256);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.build('/tmp/rod.in'); txt=fileread('/tmp/rod.in');
disp('--- source / stop / aperture lines ---');
lines=strsplit(txt,newline);
for i=1:numel(lines)
    l=lines{i};
    if contains(l,'ApStop')||contains(l,'Aperture')||contains(l,'zSource')||contains(l,'ChfRayPos')||contains(l,'ChfRayDir')||contains(l,'ApVec')
        disp(strtrim(l));
    end
end
exit(0);
end
