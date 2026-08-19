function S = seq_spot_example(opts)
%SEQ_SPOT_EXAMPLE  The deck's spot-diagram panel: chief vs centroid, before
%   and after the solves.
%
%   S = SEQ_SPOT_EXAMPLE()
%
%   One representative field -- by DEFAULT the field of the stage-2 box where
%   the chief-to-centroid separation is LARGEST, chosen by measurement, not by
%   eye -- drawn as a spot diagram on the detector with BOTH references
%   marked, and annotated with the WFE referenced to each.  A companion panel
%   shows the SAME field in stage 4, after the solves, so the before/after
%   reads in one glance.
%
%   Both panels are drawn in the detector's own 2-D frame, in micrometres,
%   with a common scale so the shrink is visually honest (the stage-4 spot is
%   NOT re-zoomed).  A zoomed inset of the stage-4 panel is added underneath
%   because at the shared scale it collapses to a dot -- which is the point,
%   but is not readable on its own.
%
%   Name-value:
%     'field'   1x2 box-relative field offset (rad).  Default: the arg-max of
%               the stage-2 centroid-displacement map.
%     'png'     output path (default rodgers1_seq_spot_example.png)
%
%   See PACKET.md Addendum 9.

    arguments
        opts.field (1,2) double = [NaN NaN]
        opts.png (1,:) char = ''
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common('seq');
    Sq = P.seq;  lam_nm = P.lambda_m*1e9;
    if isempty(opts.png)
        opts.png = fullfile(here,'rodgers1_seq_spot_example.png');
    end

    d2 = fullfile(here,'rodgers1_seq_rodgersS2.in');
    d4 = fullfile(here,'rodgers1_seq_rodgersS4.in');
    for d = {d2,d4}
        if ~isfile(d{1})
            error('seq_spot_example:deck','missing %s -- run run_seq first', d{1});
        end
    end

    % ---- pick the field by MEASUREMENT: largest stage-2 separation -------
    if any(isnan(opts.field))
        s2 = strict_wfe_deck(d2, Sq.Frel, 'reference','strict-centroid');
        [~,k] = max(s2.dcen_m);
        fld = Sq.Frel(k,:);
        fprintf(['  field chosen by measurement: index %d, ' ...
                 '(XAN %+.4f, dYAN %+.4f) deg -- largest stage-2\n' ...
                 '  chief-to-centroid separation (%.3f um of %.3f um max).\n'], ...
                k, fld(1)*180/pi, fld(2)*180/pi, s2.dcen_m(k)*1e6, max(s2.dcen_m)*1e6);
    else
        fld = opts.field;
    end

    A = spot_at_(d2, fld, P);   A.tag = 'Stage 2  (verbatim conics, detector re-fitted)';
    B = spot_at_(d4, fld, P);   B.tag = 'Stage 4  (CODE V solve: conics + M2/M3 tilt-dec)';

    S = struct('field_deg', fld*180/pi, 'stage2', A, 'stage4', B, 'png', opts.png);
    report_(A, lam_nm);  report_(B, lam_nm);

    draw_(A, B, fld, lam_nm, opts.png);
    fprintf('\n  spot panel -> %s\n', opts.png);
    save(fullfile(here,'rodgers1_seq_spot_example.mat'),'S');
end

% =====================================================================
function s = spot_at_(deck, fld, P)
%SPOT_AT_  Trace ONE field of a deck and return the ray intercepts on the
%   frozen detector, in the detector's own 2-D frame, plus both references.
    txt = regexprep(fileread(deck), '(ApType=\s*)\S+', '$1None');
    cdir0 = grab3_(txt,'ChfRayDir');  cpos0 = grab3_(txt,'ChfRayPos');
    apst  = grab3_(txt,'ApStop');
    stand = dot(apst - cpos0, cdir0);
    bx = asin(cdir0(1)) + fld(1);   by = asin(cdir0(2)) + fld(2);
    Vs = grabAll3_(txt,'VptElt');  Ps = grabAll3_(txt,'psiElt');
    Vd = Vs(:,end);  Nd = Ps(:,end)/norm(Ps(:,end));

    macos.init(P.model_size);
    tmp = [tempname '.in'];
    ri  = trace_(txt, tmp, apst, stand, bx, by);
    rp  = trace_(txt, tmp, apst, stand, bx+1e-5, by);
    delete(tmp);

    ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
    p1 = ri.pos(:,1);  d1 = ri.dir(:,1);
    X  = fex_cross_(p1, d1, rp.pos(:,1), rp.dir(:,1));
    r  = strict_refs(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), p1, d1, Vd, Nd, X);

    % ray intercepts on the frozen plane, in the detector frame
    Pm = ri.pos(:,ok);  Dm = ri.dir(:,ok);
    tt = (Nd.'*(Vd - Pm)) ./ (Nd.'*Dm);
    Q  = Pm + Dm .* tt;
    o  = r.c_chief;                                  % origin = chief intercept
    s.x = (r.e1.'*(Q - o)).' * 1e6;                  % um
    s.y = (r.e2.'*(Q - o)).' * 1e6;
    s.cen = r.centroid_2d * 1e6;                     % centroid, um, same frame
    s.wfe_chief_nm    = r.wfe_chief * 1e9;
    s.wfe_centroid_nm = r.wfe_centroid * 1e9;
    s.dcen_um = r.dcen_m * 1e6;
    s.nrays = nnz(ok);
    s.rms_spot_um = sqrt(mean((s.x - s.cen(1)).^2 + (s.y - s.cen(2)).^2));
end

function report_(s, lam_nm) %#ok<INUSD>
    fprintf('\n  %s\n', s.tag);
    fprintf('    rays %d,  spot rms about the centroid %8.3f um\n', s.nrays, s.rms_spot_um);
    fprintf('    chief-to-centroid separation          %8.3f um\n', s.dcen_um);
    fprintf('    WFE referenced to the CHIEF           %8.2f nm\n', s.wfe_chief_nm);
    fprintf('    WFE referenced to the CENTROID        %8.2f nm   <- primary\n', s.wfe_centroid_nm);
end

function draw_(A, B, fld, lam_nm, png) %#ok<INUSL>
    lim = max([abs(A.x); abs(A.y); abs(A.cen)]) * 1.15;
    fig = figure('Visible','off','Position',[80 80 1150 660],'Color','w');
    tl  = tiledlayout(fig, 2, 2, 'TileSpacing','compact','Padding','compact');

    panel_(nexttile(tl,1), A, lim, true);
    panel_(nexttile(tl,2), B, lim, true);
    ax = nexttile(tl,3); axis(ax,'off');
    text(ax, 0.0, 0.5, sprintf([ ...
        'Field  XAN %+.3f deg,  \\DeltaYAN %+.3f deg  (box corner)\n' ...
        '\\lambda = %g nm,  EPD 5000 mm,  f/20,  M1 hole 0.206 linear\n\n' ...
        'The two markers are the two WFE references.  The wavefront is\n' ...
        'referenced to a sphere anchored at the exit pupil and centred on\n' ...
        'one of them; PRIMARY is the centroid (Dave, 2026-07-31) because it\n' ...
        'is what the detector integrates.\n\n' ...
        'Stage 2 -> 4:  separation %.1f -> %.1f \\mum,  centroid-referenced\n' ...
        'WFE %.0f -> %.0f nm.  The solves remove the coma that separates them.'], ...
        fld(1)*180/pi, fld(2)*180/pi, lam_nm, A.dcen_um, B.dcen_um, ...
        A.wfe_centroid_nm, B.wfe_centroid_nm), ...
        'FontSize',10,'VerticalAlignment','middle','Interpreter','tex');

    lim4 = max([abs(B.x); abs(B.y); abs(B.cen)]) * 1.2;
    panel_(nexttile(tl,4), B, lim4, false);
    title(nexttile(tl,4), sprintf('%s\n(zoom: \\pm%.0f \\mum)', B.tag, lim4), 'FontSize',9);

    title(tl, 'Rodgers TMA at the .seq truth: chief ray vs spot centroid', ...
          'FontWeight','bold','FontSize',12);
    exportgraphics(fig, png, 'Resolution', 170);
    close(fig);
end

function panel_(ax, s, lim, showtitle)
    hold(ax,'on'); box(ax,'on'); axis(ax,'equal');
    scatter(ax, s.x, s.y, 7, [0.62 0.68 0.78], 'filled', 'MarkerFaceAlpha',0.75);
    plot(ax, 0, 0, '+', 'Color',[0.85 0.20 0.15], 'MarkerSize',15, 'LineWidth',2.0);
    plot(ax, s.cen(1), s.cen(2), 'o', 'Color',[0.05 0.35 0.75], ...
         'MarkerSize',11, 'LineWidth',2.0);
    plot(ax, [0 s.cen(1)], [0 s.cen(2)], '-', 'Color',[0.35 0.35 0.35], 'LineWidth',1.0);
    xlim(ax,[-lim lim]); ylim(ax,[-lim lim]);
    xlabel(ax,'detector x (\mum)'); ylabel(ax,'detector y (\mum)');
    if showtitle, title(ax, s.tag, 'FontSize',9); end
    legend(ax, {'rays', sprintf('chief  (WFE %.0f nm)', s.wfe_chief_nm), ...
                sprintf('centroid  (WFE %.0f nm)', s.wfe_centroid_nm), ...
                sprintf('\\Delta = %.1f \\mum', s.dcen_um)}, ...
           'Location','southoutside','FontSize',8,'Box','off','NumColumns',2);
    hold(ax,'off');
end

% ---------------------------------------------------------------------
function ri = trace_(txt, tmp, apst, stand, bx, by)
    cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
    cpos = apst - stand*cdir;
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3(cpos)]);
    fid = fopen(tmp,'w'); fprintf(fid,'%s',s); fclose(fid);
    macos.load_rx(tmp);
    nE = macos.num_elt();
    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function M = grabAll3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens');
    M = zeros(3,numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end
