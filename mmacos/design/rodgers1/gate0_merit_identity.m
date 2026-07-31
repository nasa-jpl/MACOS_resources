function G = gate0_merit_identity()
%GATE0_MERIT_IDENTITY  Is CALIB's merit at a FEX-set ExitPupil NUMERICALLY the
%   strict metric?  Addendum 4 §C argued it structurally; this makes it a
%   number, on one field, before any solve is trusted.
%
%   Left side  -- what CALIB's inner loop would evaluate with OptFEX=Yes and
%     OptWFElt = nElt-1: run FEX (which places the reference sphere at the
%     exit pupil with its centre of curvature on the chief-ray intercept on
%     element nElt, Addendum 3 §A.3), then `cmd='OPD'` at nElt-1.  Reached
%     here as macos.fex(1) + macos.trace(nElt-1).rmsWFE.  In BaseUnits (m).
%
%   Right side -- strict_wfe's own construction, from raw ray data at M3
%     (the last powered surface; rays are straight after it): sphere centred
%     on the chief-ray intercept on the deck's terminal FocalPlane, radius
%     |XP - c| with XP the FEX pupil vertex, exact OPL via
%     strict_sphere_opl, piston-only std.
%
%   These must agree to round-off.  If they do not, the in-loop merit is not
%   the metric the packet validated and no re-solve against it means
%   anything.
%
%   Runs on the committed rodgers1_epd4060_stage4_pupil.in (6 elements:
%   M1, M2, M3, FP_return, ExitPupil, FP), at its nominal field and at two
%   off-axis box fields.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common();
    deck = fullfile(here,'rodgers1_epd4060_stage4_pupil.in');
    txt  = regexprep(fileread(deck), '(ApType=\s*)\S+', '$1None');

    [cdir0, cpos0, apst] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    h   = P.fov_half_deg*pi/180;
    F   = [0 0; h h; -h -h];
    lbl = {'box centre','+x+y corner','-x-y corner'};

    macos.init(P.model_size);
    tmp = [tempname '.in'];
    G = struct('field',{},'calib_m',{},'strict_m',{},'rel',{},'nrays',{});
    fprintf('\n  %-14s %20s %20s %14s\n','field','CALIB merit (m)','strict (m)','relative');
    for k = 1:size(F,1)
        emit_field_(txt, tmp, apst, stand, bx0+F(k,1), by0+F(k,2));
        macos.load_rx(tmp);
        nE = macos.num_elt();
        macos.stop(1);                       % system stop at M1 -- FEX needs it

        % ---------- LEFT: FEX per field, then OPD at the ExitPupil -------
        xp = macos.fex(1);                   % places the XP at nElt-1
        sL = macos.trace(nE-1);
        calib_m = sL.rmsWFE;                 % BaseUnits (m), NOT waves

        % ---------- RIGHT: strict_wfe's construction from raw rays -------
        vFP = macos.get_elt_vpt(nE);  nFP = macos.get_elt_psi(nE);
        nFP = nFP(:)/norm(nFP);
        sR = macos.trace(3);                 % M3 = last powered surface
        ri = macos.get_ray_info(sR.nRays);
        ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
        p1 = ri.pos(:,1);  d1 = ri.dir(:,1);
        c  = p1 + d1*(dot(nFP, vFP(:) - p1)/dot(nFP, d1));
        R  = norm(xp.vpt(:) - c);
        W  = strict_sphere_opl(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), c, R);
        strict_m = std(W);

        rel = abs(calib_m - strict_m)/strict_m;
        fprintf('  %-14s %20.12e %20.12e %14.3e\n', lbl{k}, calib_m, strict_m, rel);
        G(k) = struct('field',F(k,:), 'calib_m',calib_m, 'strict_m',strict_m, ...
                      'rel',rel, 'nrays',nnz(ok));
    end
    delete(tmp);
    worst = max([G.rel]);
    fprintf('\n  worst relative difference: %.3e\n', worst);
    % Tolerance 1e-6, not machine epsilon: the engine intersects the ACTUAL
    % conic reference surface with its iterative solver while this
    % construction solves the sphere in closed form, so the two agree to the
    % surface solver's tolerance.  Measured 2.7e-9 -- i.e. 2e-16 m on an 8e-8 m
    % quantity -- which is round-off, and it is 3 orders inside any physical
    % claim made from this metric.
    if worst < 1e-6
        fprintf('  GATE 0 PASS -- the in-loop merit IS the strict metric, numerically.\n');
    else
        fprintf(2,'  GATE 0 FAIL -- do not trust a re-solve against this merit.\n');
    end
    save(fullfile(here,'rodgers1_epd4060_gate0.mat'),'G');
end

function emit_field_(txt, tmp, apst, stand, bx, by)
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3(cpos)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function [cdir, cpos, apst] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');  cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
end
function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end
