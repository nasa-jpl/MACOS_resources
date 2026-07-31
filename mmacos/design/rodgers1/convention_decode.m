function D = convention_decode()
%CONVENTION_DECODE  Which frame are Rodgers' stage-4 YDE/ADE values in?
%
%   Addendum 4 §B left his stage-4 rigid body uninterpretable: applied as a
%   global-frame vertex decenter + local Rx tilt, in the convention
%   rigid_of reports ours, his values score 64.6 um.  This decodes the
%   frame empirically over a BOUNDED set of variants.
%
%   The physical hypothesis being tested.  CODE V carries a coordinate
%   system that FLIPS at every mirror -- that is why his thicknesses
%   alternate in sign (-5267.9, +7106.9, -5095.4) while ours are all
%   emitted against one global frame with psiElt = (0,0,-1) on M1, M2 and
%   M3 alike.  A surface reached after an ODD number of reflections
%   therefore has its local y and z reversed relative to ours, so a CODE V
%   (YDE, ADE) there maps to (-YDE, -ADE) in our frame; after an EVEN
%   number it maps unchanged.  M2 is odd, M3 is even.  That predicts
%   M2 -> (-8.344733 mm, -0.516947 deg), M3 -> (+121.868248, +2.329710).
%
%   Rather than assume it, all 16 independent sign combinations
%   (s_dy, s_alpha) per mirror are screened, plus the two application
%   orders.  ACCEPTANCE IS NOT "does it hit 39.8 nm" -- it is "does his
%   design WORK AT ALL", i.e. land in the hundreds of nm rather than tens
%   of um.  Nothing is tuned toward his WFE.
%
%   Screen (cheap, 2 traces per variant): at the box-centre field, find the
%   arriving bundle's own best-focus point, and report the transverse spot
%   RMS there plus the strict WFE about a sphere centred there.  A design
%   whose geometry is wrong fails this by orders of magnitude; the FPA fit
%   cannot rescue it (Addendum 4 §B measured the best-focus floor at
%   64.4 um for the baseline variant).
%
%   D is the ranked table.  The winner is then run through the full
%   his_designs pipeline separately.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common();
    base = fullfile(here,'rodgers1_epd4060_stage2.in');
    txt  = regexprep(fileread(base), '(ApType=\s*)\S+', '$1None');
    work = [tempname '.in'];
    fid = fopen(work,'w'); fprintf(fid,'%s',txt); fclose(fid);

    % reference: OUR committed stage-4 solve, same screen -- the scale a
    % working design of this class reads on this metric.
    Rc = load(fullfile(here,'rodgers1_epd4060_results.mat'));
    ref = screen_(work, P, Rc.R.K_s4, ...
                  [2 Rc.R.rigid(1,1) Rc.R.rigid(1,2); ...
                   3 Rc.R.rigid(2,1) Rc.R.rigid(2,2)], 1);
    fprintf('\n  REFERENCE -- our committed S4 (a design known to work):\n');
    fprintf('    spot rms %10.4g um    strict-at-own-focus %10.4g nm\n', ...
            ref.spot_um, ref.wfe_nm);
    % and the unperturbed offset design, for the other end of the scale
    ref0 = screen_(work, P, P.K_nom, [], 1);
    fprintf('  REFERENCE -- stage 2, no rigid body:\n');
    fprintf('    spot rms %10.4g um    strict-at-own-focus %10.4g nm\n\n', ...
            ref0.spot_um, ref0.wfe_nm);

    dy = [P.Ydec_M2_mm; P.Ydec_M3_mm];
    al = [P.tilt_M2_deg; P.tilt_M3_deg];
    S  = [1 -1];
    D  = struct('label',{},'sdy',{},'sal',{},'order',{}, ...
                'spot_um',{},'wfe_nm',{});
    fprintf('  %-26s %14s %16s\n','variant','spot rms (um)','strict (nm)');
    for a2 = S, for b2 = S, for a3 = S, for b3 = S, for ord = [1 2]
        rb = [2 S(1)*a2*dy(1) b2*al(1); 3 a3*dy(2) b3*al(2)];
        rb(1,2) = a2*dy(1);
        r = screen_(work, P, P.K_s4, rb, ord);
        lab = sprintf('M2(%+d,%+d) M3(%+d,%+d) o%d', a2,b2,a3,b3,ord);
        D(end+1) = struct('label',lab,'sdy',[a2 a3],'sal',[b2 b3], ...
                          'order',ord,'spot_um',r.spot_um,'wfe_nm',r.wfe_nm); %#ok<AGROW>
        fprintf('  %-26s %14.4g %16.4g\n', lab, r.spot_um, r.wfe_nm);
    end, end, end, end, end
    delete(work);

    [~,ix] = sort([D.wfe_nm]);
    D = D(ix);
    fprintf('\n  RANKED (best first):\n');
    for k = 1:min(5,numel(D))
        fprintf('   %d. %-26s spot %10.4g um   strict %10.4g nm\n', ...
                k, D(k).label, D(k).spot_um, D(k).wfe_nm);
    end
    fprintf('\n  our-S4 reference sits at spot %.4g um / strict %.4g nm\n', ...
            ref.spot_um, ref.wfe_nm);
    save(fullfile(here,'rodgers1_epd4060_convention_decode.mat'),'D','ref','ref0');
end

% ---------------------------------------------------------------------
function r = screen_(work, P, K, rb, ord)
%SCREEN_  One variant: build, trace the box-centre field, evaluate at the
%   bundle's OWN best focus (so the FPA choice cannot mask or cause a
%   failure).  2 traces.
    macos.init(P.model_size);  macos.load_rx(work);
    nE = macos.num_elt();
    for i = 1:3, macos.set_elt_kc(i, K(i)); end
    for r_ = 1:size(rb,1)
        ie = rb(r_,1);  d = rb(r_,2)*1e-3;  a = -rb(r_,3)*pi/180;   % -Rx: Addendum 4 §B
        if ord == 1
            macos.perturb(ie, 'rotation',[a;0;0], 'translation',[0;d;0]);
        else
            macos.perturb(ie, 'rotation',[a;0;0]);
            macos.perturb(ie, 'translation',[0;d;0]);
        end
    end
    macos.modify();
    s  = macos.trace(nE);   b = macos.get_ray_info(s.nRays);
    macos.trace(nE-1);      a2 = macos.get_ray_info(s.nRays);
    macos.trace(nE);
    ok = b.ok_trace(:) & b.ok_pass(:) & a2.ok_trace(:) & a2.ok_pass(:);
    ok(1) = false;
    if nnz(ok) < 10, r = struct('spot_um',inf,'wfe_nm',inf); return; end
    Pb = b.pos(:,ok);  L = b.opl(ok);
    Dg = b.pos(:,ok) - a2.pos(:,ok);  Dg = Dg./vecnorm(Dg);
    A  = nnz(ok)*eye(3) - Dg*Dg.';
    bb = sum(Pb,2) - Dg*sum(Dg.*Pb,1).';
    X  = A \ bb;                                   % bundle best focus
    T  = X - Pb;  T = T - Dg.*sum(Dg.*T,1);        % transverse residual
    r.spot_um = sqrt(mean(sum(T.^2,1)))*1e6;
    % strict WFE about a sphere centred on that focus; R = mean ray path
    % from the last surface back one leg (only enters at O(eps^2/R)).
    R = mean(vecnorm(Pb - a2.pos(:,ok)));
    r.wfe_nm = std(strict_sphere_opl(Pb, Dg, L, X, R))*1e9;
end
