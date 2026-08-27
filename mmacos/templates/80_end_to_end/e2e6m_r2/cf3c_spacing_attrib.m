function OUT = cf3c_spacing_attrib(over)
%CF3C_SPACING_ATTRIB  Where the spacing sweep's STATIC cost enters.
%
%   CF3b found the OPPOSITE of the Talbot expectation: the closed-loop
%   floor degrades monotonically with DM spacing because the STATIC
%   degrades (apl: 4.49e-7 / 9.08e-7 / 1.10e-6 / 1.22e-6 at 0.15 /
%   0.40 / 0.70 / 1.10 m) -- the train takes back more than the
%   authority gives.  This probe attributes the static cost by
%   measurement: per emitted deck, DMs flat, the apl chain's radial
%   dark-zone profile and the symmetric-field fraction.
%     - If the extra light is BROADBAND across the annulus with a high
%       symmetric fraction, it is gap-Fresnel structure evolved over
%       the longer DM1->DM2 near-field leg (amplitude-type, the same
%       family as the baseline's substrate).
%     - If it is a localized ring or asymmetric, something in the
%       re-emitted train (fold geometry, prolate redesign) is the
%       carrier and the sweep conflates spacing with train quality.
%
%   See also CF3B_SPACING, CF2_EFC, cf_efc_lib.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end

    ds = [0.15 0.40 0.70 1.10];
    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF3c -- the spacing sweep''s static cost, attributed (apl, DMs flat)');
    R = struct('d',{}, 'con',{}, 'sym',{}, 'rr',{}, 'cc',{});
    for d = ds
        if abs(d - P.b2.d_dm2) < 1e-9
            rx = fullfile(P.outdir, 'r1_seg_dm.in');
        else
            rx = fullfile(P.outdir, sprintf('r1_seg_d%03d_dm.in', round(100*d)));
        end
        assert(isfile(rx), 'cf3c: %s absent -- run cf3b first', rx);
        ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                      'prolate_iter', P.co.prolate_iter, ...
                      'circ_stop_frac', P.cf.circ_stop_frac, FC.apl.cfg{:});
        dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
        E = ch.run();
        con = mean(abs(E(dz_idx)).^2) / ch.peak_bare;
        sym = lib.sym_frac(E, ch.center_px, dz_idx);
        [rr, cc] = macos.radial_contrast(abs(E).^2, ch.peak_bare, ...
                                         ch.lamD_px, P.co.outer_lamD + 3);
        R(end+1) = struct('d',d, 'con',con, 'sym',sym, 'rr',rr, 'cc',cc); %#ok<AGROW>
        L = say_(L, '  d=%.2f m: static %.3e | DZ symmetric fraction %.3f', d, con, sym);
    end

    ratio = R(end).cc ./ max(R(1).cc, realmin);
    inband = R(1).rr >= P.co.inner_lamD & R(1).rr <= P.co.outer_lamD;
    L = say_(L, 'radial ratio (1.10 m / 0.15 m) across the annulus: median %.2f, spread %.2f..%.2f', ...
             median(ratio(inband)), min(ratio(inband)), max(ratio(inband)));
    if all([R.sym] > 0.8) && max(ratio(inband))/max(median(ratio(inband)),realmin) < 3
        L = say_(L, 'VERDICT: broadband across the annulus, symmetric fraction high at every');
        L = say_(L, '  spacing -- the extra static light is gap-Fresnel structure evolved over');
        L = say_(L, '  the longer DM1->DM2 near-field leg: amplitude-type, the SAME substrate');
        L = say_(L, '  family as the baseline.  The spacing dial trades authority against');
        L = say_(L, '  seeing MORE of that structure; on this train the trade never wins.');
    else
        L = say_(L, 'VERDICT: the profile is ringed/asymmetric -- the re-emitted train carries');
        L = say_(L, '  structure beyond gap-Fresnel evolution; the sweep conflates spacing with');
        L = say_(L, '  train quality and the spacing conclusion needs the train term separated.');
    end

    png = fullfile(P.outdir, 'cf3c_spacing_attrib.png');
    f = figure('Visible','off','Color','w','Position',[60 60 760 520]);
    ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); set(ax,'YScale','log');
    for k = 1:numel(R)
        semilogy(ax, R(k).rr, R(k).cc, '-', 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('d = %.2f m (sym %.2f)', R(k).d, R(k).sym));
    end
    xlabel(ax, 'separation  [\lambda/D]');  ylabel(ax, 'contrast (static, DMs flat)');
    title(ax, {'Where the spacing cost enters: radial statics per DM spacing', ...
               'apodized-Lyot chain, circular stop, pre-control'}, 'FontWeight','bold');
    legend(ax, 'Location', 'northeast');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
    L = say_(L, '  figure: %s', png);
    L = say_(L, 'CF3c DONE in %.1f min', toc(t0)/60);

    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf3c_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('R',R, 'text',txt, 'figure',png, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf3c_run.mat'),'OUT');
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
