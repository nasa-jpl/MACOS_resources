function strict_geom()
%STRICT_GEOM  Dave's strict metric computed GEOMETRICALLY (tautology-proof).
%   Reading OPD by tracing TO the exit-pupil Return sphere invokes the
%   engine's OPL round-trip equalization -> nulls a chief-tied sphere
%   (~2e-4 nm floor).  Instead: at the exit-pupil element get each ray's
%   cumulative OPL and pupil crossing pos, then reference to a TRUE sphere
%   centered at that field's chief-ray intercept Cf on the tilted detector:
%       wfe_i = OPL_i + |pos_i - Cf|     (converging-sphere reference)
%   remove PISTON only (mean); RMS.  Nothing is fit -> cannot be a
%   tautology; keeps focus+astig+coma, removes only the image-displacement
%   tilt that the chief tie (Cf on the chief) takes out.
    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    P = rodgers_common();  P.EPD_mm = 4060;  lam_nm = P.lambda_m*1e9;
    Frel = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
    gt = struct('s2',[374.6 199.9],'s3',[91.6 46.4],'s4',[39.8 22.5]);

    % sanity: does get_ray_info carry a usable OPL at the EP element?
    S = struct('lam_nm',lam_nm);
    for st = [2 3 4]
        t = build_stage(P, st);
        t.add_pupil(numel(t.spec.elt));
        pu = t.spec.pupil;  iEP = pu.ep_elt;  iFP = pu.fp_elt;
        nF = size(Frel,1);  W = nan(nF,1);  N = zeros(nF,1);
        for j = 1:nF
            t.trace_at_field(Frel(j,:));
            % chief intercept on the detector
            sfp = macos.trace(iFP);  rf = macos.get_ray_info(sfp.nRays);
            Cf = chief_pos_(rf);
            % pupil crossings + OPL at the exit-pupil element
            sep = macos.trace(iEP);  re = macos.get_ray_info(sep.nRays);
            ok = re.ok_trace & re.ok_pass & isfinite(re.opl);
            if nnz(ok) < 8, continue; end
            pos = re.pos(:,ok);  opl = re.opl(ok);
            dist = sqrt(sum((pos - Cf).^2,1)).';    % |pupil pt -> Cf|
            wfe = opl + dist;                        % converging-sphere ref
            wfe = wfe - mean(wfe);                    % piston only
            W(j) = std(wfe) * 1e9;                    % opl in metres -> nm
            N(j) = nnz(ok);
        end
        t.trace_at_field([]);
        g = gt.(sprintf('s%d',st));
        S.(sprintf('s%d',st)) = struct('wfe',W,'n',N);
        fprintf(['\nSTAGE %d strict-geom (EPD 4060, box): max %.2f  avg %.2f nm  (%d..%d/%d lit)\n' ...
                 '  Rodgers S%d box: max %.1f avg %.1f -> ratio max %.2fx avg %.2fx\n'], ...
                 st, mx(W), av(W), min(N(N>0)),max(N),nF, st, g(1),g(2), mx(W)/g(1), av(W)/g(2));
    end
    save(fullfile(here,'rodgers1_epd4060_strict_rung.mat'),'S');
    fprintf('\nGATE: S2 strict-geom vs Rodgers 374.6 nm (within ~1.5x = validated)\n');
    fprintf('saved rodgers1_epd4060_strict_rung.mat\n');
end

function C = chief_pos_(rf)
    ok = rf.ok_trace & rf.ok_pass;  P=rf.pos; P(:,~ok)=NaN;
    c = mean(P(:,ok),2); d2=sum((P-c).^2,1); d2(~ok)=inf; [~,ic]=min(d2);
    C = rf.pos(:,ic);
end
function t = build_stage(P, st)
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_field_bias(P.offset_deg*60);  t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    if st==2, return; end
    optF = macos.design.field_grid(P.fov_half_deg*60, 3, 'units','arcmin','origin',false);
    if st==3, t.optimize('fields',optF,'dofs',[0 0 0 0 0 0 0 1],'max_iters',120);
    else, t.optimize('fields',optF,'dofs',[0 0 0 0 0 0 0 1;1 0 0 0 1 0 0 1;1 0 0 0 1 0 0 1],'max_iters',120); end
    t.align_focal_plane('grid',5,'span_arcmin',6);
end
function y=mx(x),x=x(isfinite(x));if isempty(x),y=NaN;else,y=max(x);end,end
function y=av(x),x=x(isfinite(x));if isempty(x),y=NaN;else,y=mean(x);end,end
