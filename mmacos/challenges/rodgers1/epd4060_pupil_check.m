function epd4060_pupil_check()
%EPD4060_PUPIL_CHECK  Insert an explicit exit pupil into the EPD=4060 stage-4
%   design and compare RMS WFE referenced at the exit pupil vs the WFE the
%   field-map metrics report -- the deliverable cross-check (Dave 2026-07-30).
%
%   Finding it answers: WFE is REFERENCE-SURFACE dependent.  On this fast
%   (f/0.86) offset-field TMA the same optics read from ~0 nm (each field to
%   its own chief focus) to ~1900 nm (std at one global plane), so the
%   "residual gap to Rodgers" is dominated by which surface the RMS is taken
%   on -- bearing directly on Rodgers open-ask #1.  Inserting a PHYSICAL exit
%   pupil (add_pupil) and reading the wavefront ON it gives the honest
%   exit-pupil WFE: 2.64 max / 1.59 avg nm over the +/-6' box.
%
%   NOTE the emit fix this exercised: add_pupil's FP_return + ExitPupil
%   Return surfaces used to emit ApType=Circular at the generous reference
%   ap_r, which clipped the fast beam (Dave: "no obscuration on the FP or
%   return surfaces").  Fixed -> Return kind emits ApType=None; 1305/1305
%   rays now survive to the exit pupil at every field.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    addpath(fullfile(fileparts(fileparts(here)),'design','src'));
    P = rodgers_common();  P.EPD_mm = 4060;  lam_nm = P.lambda_m*1e9;
    Frel = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');

    banner('EPD=4060 STAGE-4 EXIT-PUPIL WFE CROSS-CHECK');

    % ---- OBJECT 1: exit pupil (no realize contamination) -----------
    t = solve_stage4(P);
    nFP = numel(t.spec.elt);
    wfp = probe_grid(t, Frel, nFP, lam_nm);          % FP, per-field own focus

    t.add_pupil(numel(t.spec.elt));                  % insert explicit EP
    pu = t.spec.pupil;
    fprintf('  add_pupil: EP elt %d (sphere R=%.1f mm, CoC at image), det FP elt %d\n', ...
            pu.ep_elt, pu.ep_radius*1e3, pu.fp_elt);
    wep = probe_grid(t, Frel, pu.ep_elt, lam_nm);    % exit-pupil-referenced
    fprintf('  ray survival at EP: min %d / max %d of %d (no clip)\n', ...
            min(wep.n), max(wep.n), max(wep.n));
    t.save(fullfile(here,'rodgers1_epd4060_stage4_pupil.in'));

    % ---- OBJECT 2: field-map metrics (fresh object) ----------------
    t2 = solve_stage4(P);
    by = t2.spec.field_bias;  Fabs = [Frel(:,1), by+Frel(:,2)];
    sg = t2.realize_apertures('fields',Fabs,'quiet',true,'metric','global');
    sr = t2.realize_apertures('fields',Fabs,'quiet',true,'metric','refsphere');
    wg = sg.wfe(isfinite(sg.wfe))*lam_nm;  wr = sr.wfe(isfinite(sr.wfe))*lam_nm;

    banner('READING -- RMS WFE is reference-surface dependent (box +/-6'')');
    fprintf('  %-44s %9s %9s\n','reference surface / metric','max nm','avg nm');
    row('FP, per-field own chief focus (engine)', max(wfp.rms), mean(wfp.rms));
    row('EXIT-PUPIL sphere (engine, CoC@image)',  max(wep.rms), mean(wep.rms));
    row('field-map refsphere (p/t/t+defocus rm)', max(wr),      mean(wr));
    row('field-map global (std @ one plane)',     max(wg),      mean(wg));
    fprintf('  %-44s %9.1f %9.1f\n','Rodgers S4 (CODE V field-map RMS)', 39.8, 22.5);

    R = struct('lam_nm',lam_nm,'ep',pu, ...
        'fp_perfield',[max(wfp.rms) mean(wfp.rms)], ...
        'exit_pupil',[max(wep.rms) mean(wep.rms)], ...
        'exit_pupil_perfield',wep.rms, ...
        'realize_refsphere',[max(wr) mean(wr)], ...
        'realize_global',[max(wg) mean(wg)], ...
        'rodgers_s4',[39.8 22.5], 'nrays',[min(wep.n) max(wep.n)]);
    save(fullfile(here,'rodgers1_epd4060_pupil_check.mat'),'R');
    fprintf('\n  saved rodgers1_epd4060_pupil_check.mat + _stage4_pupil.in\n');
end

function out = probe_grid(t, Frel, elt, lam_nm)
    n = size(Frel,1);  r = zeros(n,1);  sdev = zeros(n,1);  nr = zeros(n,1);
    for j=1:n
        t.trace_at_field(Frel(j,:));
        s = macos.trace(elt);  W = macos.opd();  v = W(isfinite(W)&W~=0);
        r(j) = s.rmsWFE*lam_nm;  nr(j)=numel(v);
        if ~isempty(v), sdev(j) = std(v)*lam_nm; end
    end
    t.trace_at_field([]);
    out = struct('rms',r,'std',sdev,'n',nr);
end

function t = solve_stage4(P)
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_field_bias(P.offset_deg*60);
    t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    optF = macos.design.field_grid(P.fov_half_deg*60, 3, 'units','arcmin','origin',false);
    t.optimize('fields', optF, 'dofs', [0 0 0 0 0 0 0 1;1 0 0 0 1 0 0 1;1 0 0 0 1 0 0 1], 'max_iters',120);
    t.align_focal_plane('grid',5,'span_arcmin',6);
end

function row(name,mx,av), fprintf('  %-44s %9.3f %9.3f\n', name, mx, av); end
function banner(varargin)
    fprintf('\n==================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('==================================================================\n');
end
