function tg96_d1_picture(lensmat, oapmat, outpng)
%TG96_D1_PICTURE  D1 placement error + J column-norm over the actuator grid,
%   both rigs (review item 4).  An optical cause of a placement/response deficit
%   is smooth and symmetric about the fold plane; a chief-pixel-reference leak
%   is a dip at the CENTRE.  Reads two matrix-battery .mat files (out struct).
%     tg96_d1_picture('runs/lens/lens.mat','runs/oap/oap.mat','d1_picture.png')
if nargin < 3, outpng = 'tg96_d1_picture.png'; end
L = load(lensmat);  O = load(oapmat);
rigs = {L.out, 'lens'; O.out, 'oap'};
f = figure('Visible','off','Position',[60 60 1150 760]);
for k = 1:2
    o = rigs{k,1};  nm = rigs{k,2};  b = o.battery;  pl = b.place;  PL = pl.PL;
    err = hypot(pl.comU - PL.U, pl.comV - PL.V);
    err(~pl.poked) = NaN;                            % only poked actuators
    subplot(2,2,2*k-1);
    him = imagesc(err);  set(him,'AlphaData',~isnan(err));  axis image ij;  colorbar;
    caxis([0 3]);  title(sprintf('%s: D1 CoM error (px)  [median %.2f]', nm, b.place.med_err));
    xlabel('actuator col');  ylabel('actuator row');
    subplot(2,2,2*k);
    cn = b.cn_map;  cn(cn==0) = NaN;
    him2 = imagesc(cn);  set(him2,'AlphaData',~isnan(cn));  axis image ij;  colorbar;
    title(sprintf('%s: J column norm (per-actuator response energy^{1/2})', nm));
    xlabel('actuator col');  ylabel('actuator row');
end
sgtitle('D1 placement error + response column-norm over the DM (item 4)');
print(f, outpng, '-dpng', '-r140');
fprintf('wrote %s\n', outpng);
end
