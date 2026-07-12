function out = segmirmaker_run(parent_in, opts)
%SEGMIRMAKER_RUN  Drive the SegMirMaker generator non-interactively.
%
%   out = macos.design.segmirmaker_run(parent_in, Name=Value) runs the
%   standalone SegMirMaker binary (segmirmaker/) on a parent MACOS
%   prescription by scripting its interactive dialog over stdin, in a
%   fresh scratch directory.  Returns the paths of the generated
%   segment prescription blocks (.presc) and edge-sensor measurement
%   matrix (<name>Hx.m).
%
%   SegMirMaker has no native batch mode (design-layer plan item Q7);
%   this driver IS the batch mode: it assembles the ordered answer list
%   for the dialog (blank line = accept the bracketed default) and
%   pipes it to the binary.  The authoritative prompt order is pinned
%   by the committed fixtures segmirmaker/test_in/e5pie.stdin /
%   e5hex2.stdin and the tSegMirMaker byte-identity regression.
%
%   Options (dialog answers; [] = accept the dialog default):
%     elt         (1)     parent element number to segment
%     name        ('segout') output base name -> <name>.presc, <name>Hx.m
%     dofs        (6)     segment DOFs: 3 (tip/tilt/piston) or 6
%     start_elt   ([])    starting segment element number (default 1)
%     rings       (1)     number of segment rings (1 -> 7 seg, 2 -> 19)
%     grid        ('Hex') ray grid type: Circular|Square|Hex|Pie|Flower
%                         (echoed verbatim into the .presc header)
%     psi         ([])    mirror principal axis 3-vec (default: parent's)
%     segxgrid    ([])    segment-grid x direction 3-vec (default [1 0 0])
%     f           ([])    focal length (default: parent's)
%     ecc         ([])    eccentricity (default: parent's)
%     gap         (0)     inter-segment gap
%     standoff    ([])    segment-definition-plane standoff (default f/2)
%     meas_config (1)     edge sensors: 1 = inner edges, 2 = all adjacencies
%     seg_size    ([])    segment size side-to-side (size option 1;
%                         default Aperture/(2*rings+1))
%     ap_diam     ([])    mirror aperture diameter (size option 2);
%                         mutually exclusive with seg_size
%     binary      ('')    SegMirMaker executable (default: sibling
%                         segmirmaker/build_release_ifx/SegMirMaker,
%                         override via env SEGMIRMAKER_BIN)
%     workdir     ('')    scratch dir (default: fresh tempname dir; it
%                         is left in place — outputs live there)
%     param_file  ('')    macos_param.txt to use (default: parent's dir,
%                         falling back to segmirmaker/test_in)
%
%   out fields: .presc .hx .log .workdir .answers
%
%   The parent .in, macos_param.txt, and any GridFile= data files the
%   parent references are copied into the scratch dir (SegMirMaker
%   loads the parent through the real engine loader, which resolves
%   GridFile from the cwd).
%
%   See also: tSegMirMaker, segmirmaker/README.md ("Batch / scripted use").

arguments
    parent_in (1,1) string
    opts.elt (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.name (1,1) string = "segout"
    opts.dofs (1,1) double {mustBeMember(opts.dofs, [3 6])} = 6
    opts.start_elt double = []
    opts.rings (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.grid (1,1) string = "Hex"
    opts.psi double = []
    opts.segxgrid double = []
    opts.f double = []
    opts.ecc double = []
    opts.gap (1,1) double {mustBeNonnegative} = 0
    opts.standoff double = []
    opts.meas_config (1,1) double {mustBeMember(opts.meas_config, [1 2])} = 1
    opts.seg_size double = []
    opts.ap_diam double = []
    opts.binary (1,1) string = ""
    opts.workdir (1,1) string = ""
    opts.param_file (1,1) string = ""
end

if ~isfile(parent_in)
    error('macos:design:segmirmaker_run:parent', ...
        'parent prescription not found: %s', parent_in);
end
if ~isempty(opts.seg_size) && ~isempty(opts.ap_diam)
    error('macos:design:segmirmaker_run:sizeargs', ...
        'seg_size and ap_diam are mutually exclusive (dialog size option 1 vs 2)');
end

% --- locate the binary -------------------------------------------------
bin = opts.binary;
if strlength(bin) == 0
    bin = string(getenv('SEGMIRMAKER_BIN'));
end
if strlength(bin) == 0
    here = fileparts(mfilename('fullpath'));          % .../src/+macos/+design
    res_root = fileparts(fileparts(fileparts(fileparts(here))));  % MACOS_resources
    bin = fullfile(res_root, 'segmirmaker', 'build_release_ifx', 'SegMirMaker');
end
if ~isfile(bin)
    error('macos:design:segmirmaker_run:binary', ...
        ['SegMirMaker binary not found: %s\n' ...
         'Build it: cd segmirmaker && source ./makesegmirmaker.sh ' ...
         '(or set SEGMIRMAKER_BIN / the ''binary'' option).'], bin);
end

% --- scratch dir + input files -----------------------------------------
wd = opts.workdir;
if strlength(wd) == 0
    wd = string(tempname);
end
if ~isfolder(wd), mkdir(wd); end
[parent_dir, parent_base, parent_ext] = fileparts(parent_in);
if strlength(parent_dir) == 0, parent_dir = "."; end
copyfile(parent_in, fullfile(wd, parent_base + parent_ext));

pf = opts.param_file;
if strlength(pf) == 0
    pf = fullfile(parent_dir, 'macos_param.txt');
    if ~isfile(pf)
        here = fileparts(mfilename('fullpath'));
        res_root = fileparts(fileparts(fileparts(fileparts(here))));
        pf = fullfile(res_root, 'segmirmaker', 'test_in', 'macos_param.txt');
    end
end
if ~isfile(pf)
    error('macos:design:segmirmaker_run:param', ...
        'macos_param.txt not found (looked beside the parent and in segmirmaker/test_in)');
end
copyfile(pf, fullfile(wd, 'macos_param.txt'));

% Grid data files the parent references (engine resolves them from cwd).
txt = fileread(parent_in);
gf = regexp(txt, 'GridFile\s*=\s*(\S+)', 'tokens');
for k = 1:numel(gf)
    g = string(gf{k}{1});
    if strcmpi(g, "none"), continue; end
    src = fullfile(parent_dir, g);
    dst = fullfile(wd, g);
    if isfile(src)
        if ~isfile(dst), copyfile(src, dst); end
    else
        error('macos:design:segmirmaker_run:gridfile', ...
            'parent references GridFile=%s but %s does not exist', g, src);
    end
end

% --- assemble the dialog answers (order is load-bearing) ---------------
    function s = numstr(v)
        s = strtrim(sprintf('%.17g ', v));
    end
blank = "";
    function s = optnum(v)
        if isempty(v), s = blank; else, s = string(numstr(v)); end
    end
answers = [ ...
    parent_base + parent_ext;      ... % 1  parent prescription file
    string(numstr(opts.elt));      ... % 2  parent element number
    opts.name;                     ... % 3  output file base name
    string(numstr(opts.dofs));     ... % 4  segment DOFs (3 or 6)
    optnum(opts.start_elt);        ... % 5  starting segment number
    string(numstr(opts.rings));    ... % 6  number of rings
    opts.grid;                     ... % 7  ray grid type
    optnum(opts.psi);              ... % 8  principal axis (x,y,z)
    optnum(opts.segxgrid);         ... % 9  SegXgrid (x,y,z)
    optnum(opts.f);                ... % 10 focal length
    optnum(opts.ecc);              ... % 11 eccentricity
    string(numstr(opts.gap));      ... % 12 inter-segment gap
    optnum(opts.standoff);         ... % 13 standoff distance
    string(numstr(opts.meas_config)) ]; % 14 measurement configuration
if ~isempty(opts.ap_diam)
    answers(end+1) = "2";              % 15 size option: aperture diameter
    answers(end+1) = string(numstr(opts.ap_diam));
else
    answers(end+1) = "1";              % 15 size option: segment size
    answers(end+1) = optnum(opts.seg_size);
end
answers(end+1) = "Y";                  % 17 write prescription to file

ansfile = fullfile(wd, 'answers.stdin');
fid = fopen(ansfile, 'w');
fprintf(fid, '%s\n', answers);
fclose(fid);

% --- run ----------------------------------------------------------------
logfile = fullfile(wd, 'segmirmaker_run.log');
cmd = sprintf('cd "%s" && "%s" < "%s" > "%s" 2>&1', wd, bin, ansfile, logfile);
st = system(cmd);
logtxt = "";
if isfile(logfile), logtxt = string(fileread(logfile)); end

presc = fullfile(wd, opts.name + ".presc");
hx    = fullfile(wd, opts.name + "Hx.m");
if st ~= 0 || ~isfile(presc) || ~isfile(hx) || ~contains(logtxt, "Done.")
    tail = logtxt;
    if strlength(tail) > 2000, tail = extractAfter(tail, strlength(tail)-2000); end
    error('macos:design:segmirmaker_run:failed', ...
        ['SegMirMaker run failed (exit %d).  The dialog is answered ' ...
         'positionally — a mismatched answer shifts every later prompt.\n' ...
         'workdir: %s\n--- log tail ---\n%s'], st, wd, tail);
end

out = struct('presc', char(presc), 'hx', char(hx), 'log', char(logfile), ...
             'workdir', char(wd), 'answers', {cellstr(answers)});
end
