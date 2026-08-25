function OUT = s3b_pupil(over)
%S3B_PUPIL  Take the segmented pupil AS THE ENGINE SEES IT at the apodizer.
%
%   The LP apodizer must be optimized against the pupil the propagation
%   actually carries -- gaps, segment edges, whatever the tilted-fold
%   relay does to the beam -- not against an idealized hexagon redraw.
%   So the amplitude comes from a traced complex field at the Apodizer
%   plane of the s3_seg diffraction deck, at the SAME model size the
%   coronagraph runs at.
%
%   Also measures the pupil's symmetry, because that decides how much
%   the LP can be folded: a pupil symmetric under x-flip AND y-flip has
%   a REAL focal field, which collapses the dark-zone constraints from
%   two (Re, Im) to one and halves the row count.
%
%   Writes s3b_pupil.mat + s3b_pupil.png; reports numerically.
%
%   See also S3_CORO, APODIZER_LP, ctb_aplc.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end

    rx = fullfile(P.outdir, 's3_seg_prop.in');
    assert(isfile(rx), 's3b_pupil: %s missing -- run s3_coro first', rx);
    ix = elt_ix_(rx);
    N  = P.co.model;

    fprintf('[pupil] deck %s\n', rx);
    fprintf('[pupil] Apodizer = elt %d of %d, model %d\n', ...
            ix.Apodizer, ix.Science, N);

    macos.init(N);
    macos.load_rx(rx);
    seed = seed_pair_(ix);
    macos.intensity(seed(1));
    macos.intensity(seed(2), 'reset_trace', false);
    E = macos.complex_field(ix.Apodizer, 'reset_trace', false);
    Amp = abs(E);
    Amp = Amp / max(Amp(:));

    % ---- beam radius + illuminated support ----------------------------
    c = floor(N/2)+1;
    [X,Y] = meshgrid((1:N)-c, (1:N)-c);
    rr = hypot(X,Y);
    thr = 0.02;                                    % ctb's beam_radius_ rule
    lit = Amp > thr;
    r_px = max(rr(lit));
    fill = sum(lit(:)) / max(sum(rr(:) <= r_px), 1);
    fprintf('[pupil] beam radius %.1f px | %d lit px | fill %.4f of the disc\n', ...
            r_px, sum(lit(:)), fill);

    % ---- symmetry, measured -------------------------------------------
    S = struct();
    S.flipx = res_(Amp, fliplr(Amp), lit);
    S.flipy = res_(Amp, flipud(Amp), lit);
    S.rot180 = res_(Amp, rot90(Amp,2), lit);
    fprintf('[pupil] symmetry residual (rms/rms, over lit px):\n');
    fprintf('        x-flip %.3e | y-flip %.3e | 180deg %.3e\n', ...
            S.flipx, S.flipy, S.rot180);
    S.quad_ok = S.flipx < 1e-3 && S.flipy < 1e-3;
    fprintf('        quadrant fold %s (focal field %s)\n', ...
            tern_(S.quad_ok,'USABLE','NOT usable'), ...
            tern_(S.quad_ok,'REAL','COMPLEX'));

    % ---- gap structure, for the record --------------------------------
    inner = rr <= 0.95*r_px;
    gapfrac = 1 - sum(lit(inner)) / max(sum(inner(:)),1);
    fprintf('[pupil] gap+strut fraction inside 0.95R: %.4f\n', gapfrac);

    OUT = struct('Amp',Amp,'lit',lit,'r_px',r_px,'N',N,'ix',ix, ...
                 'sym',S,'fill',fill,'gapfrac',gapfrac,'rx',rx);
    matp = fullfile(P.outdir,'s3b_pupil.mat');
    save(matp,'-struct','OUT','-v7.3');
    fprintf('[pupil] wrote %s\n', matp);

    fig = figure('Visible','off','Color','w','Position',[50 50 1100 500]);
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('engine pupil amplitude at the Apodizer (model %d, R=%.0f px)', ...
          N, r_px), 'FontWeight','bold','Interpreter','none');
    w = round(2.3*r_px);
    ax=nexttile(tl); imagesc(ax, crop_(Amp,w)); axis(ax,'image','off');
    colormap(ax,gray); clim(ax,[0 1]); cb=colorbar(ax); cb.Label.String='amplitude';
    title(ax,'traced amplitude');
    ax=nexttile(tl); imagesc(ax, crop_(double(lit),w)); axis(ax,'image','off');
    colormap(ax,gray); clim(ax,[0 1]);
    title(ax,sprintf('support (>%.0f%% peak): %d px', 100*thr, sum(lit(:))));
    png = fullfile(P.outdir,'s3b_pupil.png');
    exportgraphics(fig,png,'Resolution',150); close(fig);
    fprintf('[pupil] wrote %s\n', png);
end

% ---------------------------------------------------------------- helpers
function r = res_(A, B, m)
    d = A(m) - B(m);
    r = sqrt(mean(d.^2)) / max(sqrt(mean(A(m).^2)), eps);
end
function s = seed_pair_(ix)
    fn = fieldnames(ix);
    p = fn(~cellfun('isempty', regexp(fn,'^Prop\d+_(start|end)$','once')));
    a = p{~cellfun('isempty', regexp(p,'_start$','once'))};
    b = p{~cellfun('isempty', regexp(p,'_end$','once'))};
    s = [ix.(a) ix.(b)];
end
function ix = elt_ix_(rx)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    ix = struct();
    for k = 1:numel(nm)
        ix.(matlab.lang.makeValidName(nm{k}{1})) = k;
    end
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
function setup_(here)
run(fullfile(here,'..','..','..','mmacos_setup.m'));
end
