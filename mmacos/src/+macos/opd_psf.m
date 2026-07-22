function out = opd_psf(rx_path, opts)
%MACOS.OPD_PSF  Load a prescription, trace, and display/save its OPD (+ PSF).
%   OUT = macos.opd_psf(RX_PATH, ...) is a simple, reusable front-end for
%   eyeballing or capturing the wavefront of ANY .in file: it loads RX_PATH,
%   (optionally) cleans up the exit pupil with FEX, traces to the wavefront
%   element, and returns / displays / saves the OPD.  Optionally it also runs
%   INTENSITY to produce a PSF.
%
%   Name-value options:
%     'model_size'  source-grid sampling / model dimension (default 512).
%     'wf_elt'      element to evaluate the OPD at.  Default -1 = num_elt-1
%                   (the usual exit-pupil slot).
%     'fex'         true = re-centre the exit pupil on the chief ray (FEX)
%                   to strip gross pupil tilt/piston.  Needs a STOP + >3 elts.
%     'psf'         true = also compute a PSF via INTENSITY at 'psf_elt'.
%     'psf_elt'     element for the PSF.  Default -1 = num_elt (focal plane).
%     'save_png'    write <prefix>_opd.png (and _psf.png) to 'outdir'.
%     'save_mat'    write <prefix>_opd.mat (the returned struct) to 'outdir'.
%     'prefix'      filename stem (default: the Rx basename).
%     'outdir'      output directory (default: the Rx's directory).
%     'show'        display figures (default true).
%     'psf_log'     PSF display on log10 scale (default true).
%
%   Returns OUT with fields rx_path, wf_elt, opd (NxN, WaveUnits), rmsWFE,
%   nRays, and (when 'psf') psf_elt + psf.
%
%   NOTE: GridFile= entries in a prescription resolve from the process CWD
%   (engine GridInit), so opd_psf cd's to the Rx's directory while it runs
%   (restored on return) -- co-located grid files then load correctly.
%
%   Example:
%     macos.opd_psf('my_rx.in', 'wf_elt', 55, 'psf', true, ...
%                   'save_png', true, 'save_mat', true);
%
%   See also: macos.opd, macos.intensity, macos.fex, macos.trace.
arguments
    rx_path          (1,:) char {mustBeNonempty}
    opts.model_size  (1,1) double {mustBeInteger, mustBePositive} = 512
    opts.wf_elt      (1,1) double {mustBeInteger} = -1
    opts.fex         (1,1) logical = false
    opts.psf         (1,1) logical = false
    opts.psf_elt     (1,1) double {mustBeInteger} = -1
    opts.save_png    (1,1) logical = false
    opts.save_mat    (1,1) logical = false
    opts.prefix      (1,:) char = ''
    opts.outdir      (1,:) char = ''
    opts.show        (1,1) logical = true
    opts.psf_log     (1,1) logical = true
end

if exist(rx_path, 'file') ~= 2
    error('macos:opd_psf:norx', 'prescription not found: %s', rx_path);
end
[rxdir, rxstem] = fileparts(rx_path);
if isempty(rxdir),        rxdir = pwd;        end
if isempty(opts.prefix),  opts.prefix = rxstem; end
if isempty(opts.outdir),  opts.outdir = rxdir;  end

% GridFile= resolves from the CWD (engine GridInit) -- run from the Rx dir.
old = cd(rxdir);  restoreCwd = onCleanup(@() cd(old)); %#ok<NASGU>

m = macos.Session(opts.model_size);
m.load_rx(rx_path);
n  = macos.num_elt();
wf = opts.wf_elt;  if wf < 1, wf = n - 1; end

if opts.fex
    m.trace(wf);
    macos.fex(1);          % re-reference the exit pupil to the chief ray
end

tr = m.trace(wf);
W  = macos.opd();
out = struct('rx_path', rx_path, 'wf_elt', wf, 'opd', W, ...
             'rmsWFE', tr.rmsWFE, 'nRays', tr.nRays, 'model_size', opts.model_size);
fprintf('opd_psf: %s  OPD@elt %d  nRays=%d  rmsWFE=%.4g\n', ...
        rxstem, wf, tr.nRays, tr.rmsWFE);

if opts.psf
    pe = opts.psf_elt;  if pe < 1, pe = n; end
    P  = macos.intensity(pe);
    out.psf_elt = pe;  out.psf = P;
    fprintf('opd_psf: PSF@elt %d  peak=%.4g  sum=%.4g\n', pe, max(P(:)), sum(P(:)));
end

if opts.show || opts.save_png
    show_(out, opts);
end
if opts.save_mat
    matf = fullfile(opts.outdir, [opts.prefix '_opd.mat']);
    save(matf, '-struct', 'out');
    fprintf('opd_psf: wrote %s\n', matf);
end
end

% -------------------------------------------------------------------------
function show_(out, opts)
haspsf = isfield(out, 'psf');
f = figure('Visible', tern_(opts.show, 'on', 'off'), ...
           'Position', [60 60 560*(1+haspsf) 500]);

% --- OPD panel (zeros outside the pupil masked white) ---
subplot(1, 1+haspsf, 1);
M = out.opd;  M(M == 0) = NaN;
h = imagesc(M);  set(h, 'AlphaData', ~isnan(M));
axis image off;  set(gca, 'Color', 'w');  colormap(gca, parula);  colorbar;
title(sprintf('OPD @ elt %d  (rms %.3g)', out.wf_elt, out.rmsWFE), 'Interpreter', 'none');

% --- PSF panel ---
if haspsf
    subplot(1, 2, 2);
    P = out.psf;  P = P / max(P(:) + eps);
    if opts.psf_log
        P = log10(max(P, 1e-8));
        ttl = sprintf('PSF @ elt %d  (log10, peak-norm)', out.psf_elt);
    else
        ttl = sprintf('PSF @ elt %d  (peak-norm)', out.psf_elt);
    end
    imagesc(P);  axis image off;  colormap(gca, hot);  colorbar;
    title(ttl, 'Interpreter', 'none');
end

[~, stem] = fileparts(out.rx_path);
sgtitle(stem, 'Interpreter', 'none');
if opts.save_png
    png = fullfile(opts.outdir, [opts.prefix '_opd.png']);
    print(f, png, '-dpng', '-r140');
    fprintf('opd_psf: wrote %s\n', png);
end
if ~opts.show, close(f); end
end

function s = tern_(c, a, b)
if c, s = a; else, s = b; end
end
