function E = dyson5_envelope(P, tag, Pr, opts)
%DYSON5_ENVELOPE  The closure envelope (BRIEF_to_dyson5 addendum 11): for
%   which parameters does the R4 design close?
%
%   E = dyson5_envelope(P, tag, Pr) re-solves the R4 rung (all eleven
%   variables, warm-started from the design of record Pr) at every point of
%   a one-axis-at-a-time sweep from the record -- F-number, block radius,
%   slit length (pixel count follows), pixel pitch (the FPA stays 54 x 9 mm,
%   the pixel count follows), glass -- emits each deck with apertures,
%   scores it in the ENGINE and judges it against the spec: smile and
%   keystone < P.smile_px, CRF < P.xrf_px, SRF < P.srf_px; the first metric
%   to fail is named.  A solve that ends ON a bound is recorded as such and
%   is not called closed (a solve on its bounds is not a closed design).
%   Then the two-axis CORNER: the first failing point of each of the two
%   worst axes, together.
%
%   Returns E.table (one row per point), E.axes, E.corner, E.sentence (what
%   the run-it-yourself slide says), and writes <tag>_s4env.{txt,png}.
    arguments
        P struct
        tag (1,:) char
        Pr struct
        opts.max_iter (1,1) double = 30
        opts.quiet (1,1) logical = false
    end
    pr = @(varargin) print_(opts.quiet, varargin{:});
    fpa_m = P.npix .* P.pixel_m;                       % the FPA stays 54 x 9 mm
    axes_ = struct('name', {}, 'vals', {}, 'label', {}, 'apply', {});
    axes_(end+1) = struct('name', 'Fno', 'vals', P.env_Fno, 'label', 'F-number', 'apply', @(Q, v) setf_(Q, 'Fno', v));
    axes_(end+1) = struct('name', 'block_r_m', 'vals', P.env_block_r_m, 'label', 'block radius (m)', 'apply', @(Q, v) setf_(Q, 'block_r_m', v));
    axes_(end+1) = struct('name', 'slit_m', 'vals', P.env_slit_m, 'label', 'slit length (m)', ...
                          'apply', @(Q, v) setf_(setf_(Q, 'npix', [round(v/Q.pixel_m), Q.npix(2)]), 'slit_m', v));
    axes_(end+1) = struct('name', 'pixel_m', 'vals', P.env_pixel_m, 'label', 'pixel (m)', ...
                          'apply', @(Q, v) setf_(setf_(Q, 'pixel_m', v), 'npix', round(fpa_m/v)));
    axes_(end+1) = struct('name', 'glass', 'vals', P.env_glass, 'label', 'glass', 'apply', @(Q, v) setf_(Q, 'glass', v));
    rows = {};  pts = struct('axis', {}, 'value', {}, 'P', {}, 'rung', {}, 'pass', {}, 'fail', {}, 'on_bounds', {});
    pr('dyson5 envelope: %d axes, R4 re-solved per point (%d iterations) from the record\n', numel(axes_), opts.max_iter);
    pr('%-16s %-10s %8s %8s %7s %7s %6s  %-5s %-12s %s\n', 'axis', 'value', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'close', 'fails', 'on bounds');
    for a = 1:numel(axes_)
        A = axes_(a);  vals = A.vals;
        if ~iscell(vals), vals = num2cell(vals); end
        for v = vals
            v = v{1};
            [row, pt] = point_(P, tag, Pr, A, v, opts.max_iter);
            rows(end+1, :) = row;  pts(end+1) = pt;   %#ok<AGROW>
            pr('%-16s %-10s %8.4f %8.4f %7.3f %7.3f %6.3f  %-5s %-12s %s\n', A.label, vstr_(v), row{3:7}, tern_(row{8}, 'yes', 'NO'), row{9}, row{10});
        end
    end
    T = cell2table(rows, 'VariableNames', {'axis', 'value', 'smile_px', 'keystone_px', 'CRF_px', 'SRF_px', 'EE_1px', 'closes', 'fails', 'on_bounds'});
    % ---- the two-axis corner: the first failing value of each failing axis, pairwise (worst two axes)
    corner = struct('axes', {}, 'values', {}, 'row', {});
    failing = {};
    for a = 1:numel(axes_)
        sel = strcmp(T.axis, axes_(a).label) & ~T.closes;
        if any(sel)
            i = find(sel, 1);  failing{end+1} = struct('axis', axes_(a), 'value', pts(i).value);   %#ok<AGROW>
        end
    end
    if numel(failing) >= 2
        f1 = failing{1};  f2 = failing{2};
        Q = f1.axis.apply(P, f1.value);  Q = f2.axis.apply(Q, f2.value);
        Ac = struct('name', 'corner', 'vals', [], 'label', sprintf('%s + %s', f1.axis.label, f2.axis.label), 'apply', @(Q0, v) Q);
        [row, pt] = point_(P, tag, Pr, Ac, [], opts.max_iter);
        corner(1) = struct('axes', {{f1.axis.label, f2.axis.label}}, 'values', {{f1.value, f2.value}}, 'row', {row});
        pr('CORNER %s = %s, %s: %s (fails %s)\n', f1.axis.label, vstr_(f1.value), vstr_(f2.value), tern_(row{8}, 'closes', 'does NOT close'), row{9});
    end
    % ---- the sentence
    E.table = T;  E.axes = axes_;  E.corner = corner;  E.points = pts;
    E.sentence = sentence_(T, axes_);
    pr('%s\n', E.sentence);
    % ---- figure: per axis, the spec metrics vs the value, closed points filled
    f = figure('Visible', 'off', 'Position', [40 40 1400 330], 'Color', 'w');
    for a = 1:numel(axes_)
        ax = subplot(1, numel(axes_), a);  hold(ax, 'on');  grid(ax, 'on');
        sel = find(strcmp(T.axis, axes_(a).label));
        xv = 1:numel(sel);  lab = cellfun(@(c) vstr_(c), T.value(sel), 'uni', 0);
        plot(ax, xv, T.CRF_px(sel)/P.xrf_px, 'o-', xv, T.SRF_px(sel)/P.srf_px, 's-', xv, max(T.smile_px(sel), T.keystone_px(sel))/P.smile_px, 'd-');
        yline(ax, 1, 'k--');  set(ax, 'XTick', xv, 'XTickLabel', lab, 'FontSize', 8);
        cl = T.closes(sel);  plot(ax, xv(~cl), ones(1, nnz(~cl))*1.02, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        title(ax, axes_(a).label, 'FontSize', 9);  if a == 1, ylabel(ax, 'metric / spec (1 = spec)'); end
        if a == numel(axes_), legend(ax, {'CRF', 'SRF', 'smile|keystone', 'does not close'}, 'Location', 'best', 'FontSize', 7); end
        ylim(ax, [0, max(1.3, max(ylim(ax)))]);
    end
    sgtitle(f, 'dyson5 closure envelope: R4 re-solved from the record, one axis at a time (engine scores / spec)', 'FontSize', 10);
    print(f, [tag '_s4env.png'], '-dpng', '-r130');  close(f);
end

function [row, pt] = point_(P, tag, Pr, A, v, max_iter)
    if isempty(v), Q = A.apply(P, []); else, Q = A.apply(P, v); end
    seed = Pr;
    if isfield(Q, 'block_r_m'), seed.block_r = Q.block_r_m; end         % the block radius is a P field in the ladder's base
    tagp = sprintf('%s_s4env_%s_%s', tag, A.name, regexprep(vstr_(v), '[^A-Za-z0-9]', ''));
    deck = [tagp '.in'];
    ok = true;  err = '';
    try
        L = dyson_ladder(Q, tagp, 'rungs', 5, 'seed', seed, 'deck', deck, 'nx', P.ladder_nx, 'nlam', P.ladder_nlam, ...
                         'w_dist', P.ladder_w_dist, 'w_blur', P.ladder_w_blur, 'clear_m', P.ladder_clear_m, 'max_iter', max_iter, 'quiet', true);
        r = L.rung(1);
    catch e
        ok = false;  err = e.message;  r = struct('engine', struct('smile_max', NaN, 'keystone_max', NaN, 'crf_max', NaN, 'srf_max', NaN, 'ee_min', NaN), 'on_bounds', {{}}, 'P', Q);
    end
    Re = r.engine;
    fails = {};
    if ~ok, fails{end+1} = 'solve'; end
    if ~(Re.smile_max < P.smile_px), fails{end+1} = 'smile'; end
    if ~(Re.keystone_max < P.smile_px), fails{end+1} = 'keystone'; end
    if ~(Re.crf_max < P.xrf_px), fails{end+1} = 'CRF'; end
    if ~(Re.srf_max < P.srf_px), fails{end+1} = 'SRF'; end
    onb = r.on_bounds;
    closes = isempty(fails) && isempty(onb);
    if isempty(fails) && ~isempty(onb), fails{end+1} = 'on bounds'; end
    row = {A.label, vstr_(v), Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, closes, strjoin(fails, ','), strjoin(onb, ',')};
    pt = struct('axis', A.label, 'value', v, 'P', r.P, 'rung', r, 'pass', closes, 'fail', {fails}, 'on_bounds', {onb});
    if ~ok, row{10} = ['ERROR: ' err]; end
end

function s = sentence_(T, axes_)
    parts = {};
    for a = 1:numel(axes_)
        sel = strcmp(T.axis, axes_(a).label);  cl = T.closes(sel);  vals = T.value(sel);
        if all(cl), parts{end+1} = sprintf('every %s tried (%s)', axes_(a).label, strjoin(vals', ', '));   %#ok<AGROW>
        elseif ~any(cl), parts{end+1} = sprintf('NO %s tried', axes_(a).label);   %#ok<AGROW>
        else, parts{end+1} = sprintf('%s %s (not %s)', axes_(a).label, strjoin(vals(cl)', ', '), strjoin(vals(~cl)', ', '));   %#ok<AGROW>
        end
    end
    fl = T.fails(~T.closes);
    if isempty(fl), first = 'none'; else, first = strjoin(unique(fl)', '; '); end
    s = sprintf('Designs close for %s; the metrics that fail outside are: %s.', strjoin(parts, '; '), first);
end

function Q = setf_(Q, f, v), Q.(f) = v; end
function s = vstr_(v)
    if isempty(v), s = 'corner'; elseif ischar(v) || isstring(v), s = char(v); elseif numel(v) > 1, s = mat2str(v, 4); else, s = sprintf('%.4g', v); end
end
function t = tern_(c, a, b), if c, t = a; else, t = b; end, end
function print_(quiet, varargin), if ~quiet, fprintf(varargin{:}); end, end
