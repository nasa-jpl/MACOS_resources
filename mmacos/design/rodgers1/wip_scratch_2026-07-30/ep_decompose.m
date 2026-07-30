function ep_decompose()
%EP_DECOMPOSE  State EXACTLY what each exit-pupil reference convention
%   removes, and compute Dave's strict rung honestly.
%
%   Dave (2026-07-30): "OPD on an exit pupil FEX-generated will have FOCUS
%   but not much tilt -- tying the EP to the chief ray is essentially
%   removing the TILT, except for aberrations producing coma."  So the
%   strict rung = OPD at the exit-pupil sphere with PISTON + TILT removed
%   (the chief-ray tie), FOCUS KEPT.  Re-centering the sphere on the field's
%   own focus (set_xp / FEX per field) removes focus too -> ~0.0002 nm
%   tautology floor; that is NOT the metric.
%
%   For stages 2,3,4 (EPD 4060), OPD at the FIXED add_pupil exit-pupil
%   sphere (CoC = on-axis image), per box field, three removal levels on the
%   codebase's meshgrid pupil convention (same as refsphere_rms_):
%     raw    piston only
%     STRICT piston + tilt      (chief-tie; KEEPS focus)   <-- Dave's rung
%     +focus piston+tilt+2rho^2-1 (what FEX/refsphere also strip)
%   The strict rung uses NO defocus term, so the f/0.86 2rho^2-1 basis
%   artifact cannot enter it.
    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    P = rodgers_common();  P.EPD_mm = 4060;  lam_nm = P.lambda_m*1e9;
    Frel = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
    gt = struct('s2',[374.6 199.9],'s3',[91.6 46.4],'s4',[39.8 22.5]);

    S = struct('lam_nm',lam_nm);
    for st = [2 3 4]
        t = build_stage(P, st);
        t.add_pupil(numel(t.spec.elt));
        pu = t.spec.pupil;  iEP = pu.ep_elt;
        nF = size(Frel,1);
        Raw=nan(nF,1); Tlt=nan(nF,1); Foc=nan(nF,1); N=zeros(nF,1);
        for j=1:nF
            t.trace_at_field(Frel(j,:));
            macos.trace(iEP);                 % OPD at the FIXED EP sphere
            [Raw(j),Tlt(j),Foc(j),N(j)] = ladder_(macos.opd(), lam_nm);
        end
        t.trace_at_field([]);
        S.(sprintf('s%d',st)) = struct('raw',Raw,'tilt',Tlt,'focus',Foc,'n',N);
        g = gt.(sprintf('s%d',st));
        fprintf(['\nSTAGE %d  (EPD 4060, box, nm)   %d..%d/%d lit\n' ...
                 '  raw   (piston only)                 %8.3f %8.3f %8.3f\n' ...
                 '  STRICT (piston+tilt, chief-tie,foc kept) %8.3f %8.3f %8.3f  <-- vs Rodgers\n' ...
                 '  +focus (piston+tilt+defocus)        %8.3f %8.3f %8.3f\n' ...
                 '  Rodgers box   max/avg               %8.1f %30.1f\n'], st, ...
                 min(N(N>0)),max(N),nF, ...
                 mn(Raw),mx(Raw),av(Raw), mn(Tlt),mx(Tlt),av(Tlt), ...
                 mn(Foc),mx(Foc),av(Foc), g(1), g(2));
        fprintf('  STRICT ratio vs Rodgers: max %.2fx  avg %.2fx\n', mx(Tlt)/g(1), av(Tlt)/g(2));
    end
    save(fullfile(here,'rodgers1_epd4060_strict_rung.mat'),'S');
    fprintf('\nsaved rodgers1_epd4060_strict_rung.mat\n');
end

function [raw,tl,fo,n] = ladder_(W, lam_nm)
%LADDER_  piston / piston+tilt / piston+tilt+defocus RMS on the meshgrid
%   pupil convention (identical grid to Telescope.refsphere_rms_).
    [ny,nx] = size(W);
    [X,Y] = meshgrid(linspace(-1,1,nx), linspace(-1,1,ny));
    m = isfinite(W) & (W~=0) & (abs(W)<1e30);
    n = nnz(m);
    if n < 8, raw=NaN; tl=NaN; fo=NaN; return; end
    x=X(m); y=Y(m); w=W(m);
    x=x-mean(x); y=y-mean(y); s=max(hypot(x,y)); if s>0, x=x/s; y=y/s; end
    r2=x.^2+y.^2;
    Bp=ones(size(x));  Bt=[Bp,x,y];  Bf=[Bp,x,y,2*r2-1];
    raw=std(w-Bp*(Bp\w))*lam_nm;
    tl =std(w-Bt*(Bt\w))*lam_nm;
    fo =std(w-Bf*(Bf\w))*lam_nm;
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
function y=mn(x),x=x(isfinite(x));if isempty(x),y=NaN;else,y=min(x);end,end
function y=mx(x),x=x(isfinite(x));if isempty(x),y=NaN;else,y=max(x);end,end
function y=av(x),x=x(isfinite(x));if isempty(x),y=NaN;else,y=mean(x);end,end
