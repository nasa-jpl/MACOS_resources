function figs = oi_demo_show(src)
%OI_DEMO_SHOW  Render an oi_demo_step result on screen — the reveal.
%
%   OI_DEMO_SHOW(OUT)          renders the run OUT (from oi_demo_step).
%   OI_DEMO_SHOW('<..>_run.mat') loads a saved run record and renders it.
%   OI_DEMO_SHOW()             the NEWEST *_run.mat in demo_adjacent/.
%
%   Three windows, plus the verdict block re-printed to the command
%   window:
%     1  the solved instrument, LIVE-traced — macos.view_std of the
%        emitted deck (a real re-render, content-framed, not the PNG),
%     2  the dense WFE field map (the run's saved figure),
%     3  the solve-field / ray panel (saved figure).
%
%   oi_demo_step calls this automatically at completion on a desktop
%   session ('show' option), so by the reveal the windows are already
%   up in the solve MATLAB.  Safe any time after the fact: everything
%   renders from the run record.
%
%   See also OI_DEMO_STEP, RUN_OI_DEMO.
arguments
    src = ''
end
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
addpath(here);

if isempty(src)
    d = dir(fullfile(here, 'demo_adjacent', '*_run.mat'));
    assert(~isempty(d), 'oi_demo_show:none', ...
           'no *_run.mat found in demo_adjacent/');
    [~, k] = max([d.datenum]);
    src = fullfile(d(k).folder, d(k).name);
end
if ischar(src) || isstring(src)
    S = load(char(src), 'OUT');
    OUT = S.OUT;
else
    OUT = src;
end
assert(~(isfield(OUT, 'refused') && OUT.refused), 'oi_demo_show:refused', ...
       'this run was refused — there is no design to show');

figs = gobjects(0);

% ---- 1: the solved instrument, live ------------------------------------
ttl = sprintf('the adjacent instrument — box %g x %g deg', OUT.box_deg);
try
    try, macos.num_elt(); catch, macos.init(OUT.P.model); end
    ok = macos.load_rx(OUT.files.deck);
    assert(ok ~= 0);
    figs(end+1) = macos.view_std('title', ttl);                 %#ok<AGROW>
catch me
    warning('oi_demo_show:layout', ...
            'live layout render failed (%s); showing the saved PNG', ...
            me.message);
    figs(end+1) = png_window_(OUT.files.layout, ttl);           %#ok<AGROW>
end

% ---- 2 + 3: the run's own figures --------------------------------------
figs(end+1) = png_window_(OUT.files.map, 'WFE over the field'); %#ok<AGROW>
if isfield(OUT.files, 'fields') && exist(OUT.files.fields, 'file')
    figs(end+1) = png_window_(OUT.files.fields, 'solve fields'); %#ok<AGROW>
end

% ---- the verdict, again -------------------------------------------------
if exist(OUT.files.verdict, 'file')
    fprintf('%s\n', fileread(OUT.files.verdict));
end
figs = figs(isgraphics(figs));
end

function f = png_window_(png, name)
f = figure('Name', name, 'Color', 'w', 'NumberTitle', 'off');
if exist(png, 'file')
    image(imread(png));
    axis image off
else
    text(0.5, 0.5, sprintf('missing: %s', png), 'HorizontalAlignment', ...
         'center', 'Interpreter', 'none');
    axis off
end
end
