function info = write_workbook(coef, path, varargin)
%WRITE_WORKBOOK  The HabEx-form error-budget calculator, with LIVE formulas.
%   INFO = macos.budget.write_workbook(COEF, PATH) writes an .xlsx the
%   system engineer EDITS: the sigma inputs sit in white cells, every other
%   number is a formula or a tool-written coefficient, so a changed sigma
%   re-rolls the budget in Excel / LibreOffice without MATLAB.  COEF is
%   macos.budget.to_coef's struct (both stages when run).  Layout after the
%   HabEx-B WFE calculator (MACOS_sandbox/error_budgets/HabExB_WFE_budget.xlsx):
%
%   Sheet 'WFE' (the calculator).  A1 date serial, B1 title, C1 'Initial
%     Alignment', G1 'Metrology Control', K1 'Total WFE'; rows 2-3 the
%     column heads; then one row per budget line, grouped by section (the
%     section label in column A on its first row):
%        B name | C sigma  D units (INPUT) | E = C*Coefficients!D (Pre WFC)
%        F = C*Coefficients!E (Post WFC) | G drift sigma  H units (INPUT) |
%        I = G*Coefficients!G (Pre-metrology) | J = G*Coefficients!H
%        (Post-metrology) | K = SQRT(SUMSQ(F,J))
%     then 'WFE Summary': one SQRT(SUMSQ(...)) row per section, 'Everything
%     Else' (the lines outside the focus section), 'TOTAL', 'Design WFE'
%     (input C -> F, K), 'Payload WFE' = SQRT(SUMSQ(K_TOTAL, K_design)),
%     'WFE Requirement' (input), 'WFE Margin' = 1 - Payload/Requirement.
%     Every range is the generated rows' -- no hard-coded row numbers.
%   Sheet 'Coefficients'.  Same rows: B name, C units, D/E = Pre/Post WFC,
%     F drift units, G/H = Pre/Post-MET -- each a FORMULA pointing at the
%     raw block to its right, which the tool writes: J '<key>0' K value,
%     L '<key>f' M value (iSeq = 1, nm per engineering unit); O/P, Q/R the
%     same for iSeq = 2; T/U the Monte-Carlo check of the line's post-WFC /
%     post-MET WFE (nm) when the 'table' option carries MC columns.
%   Sheet 'WFE BD' (the roll-up tree): Total = Payload, Static (post-WFC)
%     and Drift (post-metrology) branches over the line rows, per-section
%     static / drift, the design residual, the Reserve against the
%     requirement, and the total in waves at an input wavelength.
%   Sheets 'PSD', 'StrTherm' -- stubs for the engineer's non-ray-trace
%     terms (pass-through rows link to them by hand); 'Log' -- the
%     generation record.
%
%   Options:
%     'title'        workbook title (default 'MACOS WFE budget')
%     'design_wfe'   design residual, nm (default 0)
%     'requirement'  WFE requirement, nm (default NaN -> blank input)
%     'focus'        the section reported against 'Everything Else'
%                    (default the first line's section)
%     'table'        macos.budget.rollup's table, for the MC check columns
%     'wavelength_nm'  for the block diagram's waves row (default 500)
%     'log'          cellstr of extra Log lines (sens artifact, SHAs, ...)
%
%   The WFS sensing line's sigma is WHITE per-sample noise (Eq. 32; Dave
%   2026-10-16), so its post-WFC coefficient is sqrt(rank/n_w); the HabEx
%   reading (alpha = 1, the sigma passes whole) is one edit -- type 1 in
%   that line's Coefficients!M cell.  The Log names the cell.
%
%   INFO: .path .rows (struct: lines [first last], summary rows by name,
%   payload cell) for gates that read the workbook back.
%
%   Written directly as SpreadsheetML parts (zip of XML) -- MATLAB's
%   writecell stores a '=' string as TEXT (measured 2026-10-15), and this
%   needs no openpyxl.  Formulas carry no cached value and the workbook
%   asks for a full recalculation on load.
%
%   See also macos.budget.to_coef, macos.budget.rollup,
%   macos.budget.write_xlsx.

p = inputParser;
p.addParameter('title', 'MACOS WFE budget', @(x) ischar(x) || isstring(x));
p.addParameter('design_wfe', 0, @isscalar);
p.addParameter('requirement', NaN, @isscalar);
p.addParameter('focus', '', @(x) ischar(x) || isstring(x));
p.addParameter('table', [], @(x) isempty(x) || istable(x));
p.addParameter('wavelength_nm', 500, @isscalar);
p.addParameter('log', {}, @iscellstr);
p.parse(varargin{:});
o = p.Results;

% ---- lines, grouped by section (first-appearance order) -----------------
keys = coef.meta.order(:);
L = cellfun(@(k) coef.(k), keys, 'UniformOutput', false);
L = [L{:}];
secs = unique({L.section}, 'stable');
ord = [];
for s = 1:numel(secs), ord = [ord; find(strcmp({L.section}, secs{s}))']; end %#ok<AGROW>
L = L(ord);  keys = keys(ord);
n = numel(L);
focus = char(o.focus);  if isempty(focus) && n > 0, focus = L(1).section; end

W = cells_();  C = cells_();
% ---- WFE sheet heads ----------------------------------------------------
W = put_(W, 1, 1, 'n', datenum_now_(), 1);
W = put_(W, 1, 2, 's', char(o.title), 1);
W = put_(W, 1, 3, 's', 'Initial Alignment', 1);
W = put_(W, 1, 7, 's', 'Metrology Control', 1);
W = put_(W, 1, 11, 's', 'Total WFE', 1);
hd2 = {2, 'Degree of Freedom'; 3, 'Inputs'; 5, 'RMS WFE (nm)'; 7, 'Drift Inputs'; ...
    9, 'RMS WFE (nm)'; 11, 'Post Metrology'};
for q = 1:size(hd2, 1), W = put_(W, 2, hd2{q, 1}, 's', hd2{q, 2}, 1); end
hd3 = {3, 'Values'; 4, 'Units'; 5, 'Pre WFC'; 6, 'Post WFC'; 7, 'Values'; 8, 'Units'; ...
    9, 'Pre-metrology'; 10, 'Post-metrology'; 11, 'RMS WFE (nm)'};
for q = 1:size(hd3, 1), W = put_(W, 3, hd3{q, 1}, 's', hd3{q, 2}, 1); end
C = put_(C, 1, 2, 's', 'Coefficients (nm RMS WFE per unit input)', 1);
C = put_(C, 1, 10, 's', 'iSeq = 1 (WFC)', 1);
C = put_(C, 1, 15, 's', 'iSeq = 2 (metrology)', 1);
C = put_(C, 1, 20, 's', 'Monte-Carlo check (nm)', 1);
hc = {2, 'Degree of Freedom'; 3, 'Units'; 4, 'Pre WFC'; 5, 'Post WFC'; 6, 'Units'; ...
    7, 'Pre-MET'; 8, 'Post-MET'; 10, 'key'; 11, 'pre'; 12, 'key'; 13, 'post'; ...
    15, 'key'; 16, 'pre'; 17, 'key'; 18, 'post'; 20, 'post-WFC'; 21, 'post-MET'};
for q = 1:size(hc, 1), C = put_(C, 3, hc{q, 1}, 's', hc{q, 2}, 1); end

% ---- one row per line ---------------------------------------------------
r0 = 4;  rows = r0 + (0:n-1);
mcw = nan(n, 1);  mcm = nan(n, 1);
T = o.table;
if ~isempty(T)
    for i = 1:n
        j = find(strcmp(T.Name, L(i).name) & ~strcmp(T.Kind, 'summary'), 1);
        if isempty(j), continue; end
        if ismember('MC_Post_WFC_nm', T.Properties.VariableNames), mcw(i) = T.MC_Post_WFC_nm(j); end
        if ismember('MC_Post_MET_nm', T.Properties.VariableNames), mcm(i) = T.MC_Post_MET_nm(j); end
    end
end
for i = 1:n
    r = rows(i);  e = L(i);  k = keys{i};
    if i == 1 || ~strcmp(L(i-1).section, e.section)
        W = put_(W, r, 1, 's', e.section, 1);
    end
    W = put_(W, r, 2, 's', e.name, 0);
    W = put_(W, r, 3, 'n', e.sigma_eng, 0);                  % INPUT
    W = put_(W, r, 4, 's', e.units, 0);
    W = put_(W, r, 5, 'f', sprintf('C%d*Coefficients!D%d', r, r), 2);
    W = put_(W, r, 6, 'f', sprintf('C%d*Coefficients!E%d', r, r), 2);
    W = put_(W, r, 7, 'n', e.sigma_drift, 0);                % INPUT
    W = put_(W, r, 8, 's', e.units_drift, 0);
    W = put_(W, r, 9, 'f', sprintf('G%d*Coefficients!G%d', r, r), 2);
    W = put_(W, r, 10, 'f', sprintf('G%d*Coefficients!H%d', r, r), 2);
    W = put_(W, r, 11, 'f', sprintf('SQRT(SUMSQ(F%d,J%d))', r, r), 2);
    % Coefficients: D..H point at the raw blocks, which the tool writes
    C = put_(C, r, 2, 's', e.name, 0);
    C = put_(C, r, 3, 's', e.units, 0);
    C = put_(C, r, 4, 'f', sprintf('K%d', r), 2);
    C = put_(C, r, 5, 'f', sprintf('M%d', r), 2);
    C = put_(C, r, 6, 's', e.units_drift, 0);
    C = put_(C, r, 7, 'f', sprintf('P%d', r), 2);
    C = put_(C, r, 8, 'f', sprintf('R%d', r), 2);
    C = put_(C, r, 10, 's', [k '0'], 0);   C = put_(C, r, 11, 'n', e.coef_pre, 0);
    C = put_(C, r, 12, 's', [k 'f'], 0);   C = put_(C, r, 13, 'n', e.coef_post, 0);
    C = put_(C, r, 15, 's', [k '0'], 0);   C = put_(C, r, 16, 'n', e.coef_met, 0);
    C = put_(C, r, 17, 's', [k 'f'], 0);   C = put_(C, r, 18, 'n', e.coef_met_post, 0);
    if isfinite(mcw(i)), C = put_(C, r, 20, 'n', mcw(i), 0); end
    if isfinite(mcm(i)), C = put_(C, r, 21, 'n', mcm(i), 0); end
end
rl = rows(end);  if n == 0, rl = r0 - 1; end

% ---- WFE Summary --------------------------------------------------------
r = rl + 2;
W = put_(W, r, 1, 's', 'WFE Summary', 1);
info.rows = struct();
info.rows.lines = [r0 rl];
sumrow = @(W, r, label, rngs) sum_row_(W, r, label, rngs);
for s = 1:numel(secs)
    in = rows(strcmp({L.section}, secs{s}));
    W = sumrow(W, r, sprintf('%s WFE', secs{s}), ranges_(in));
    info.rows.(fieldkey_(sprintf('%s WFE', secs{s}))) = r;
    r = r + 1;
end
ee = rows(~strcmp({L.section}, focus));
W = sumrow(W, r, 'Everything Else', ranges_(ee));  info.rows.Everything_Else = r;  r = r + 1;
W = sumrow(W, r, 'TOTAL', ranges_(rows));  rt = r;  info.rows.TOTAL = r;  r = r + 1;
rd = r;  info.rows.Design_WFE = r;
W = put_(W, r, 2, 's', 'Design WFE', 1);
W = put_(W, r, 3, 'n', o.design_wfe, 0);  W = put_(W, r, 4, 's', 'nm', 0);
W = put_(W, r, 6, 'f', sprintf('C%d', r), 2);
W = put_(W, r, 11, 'f', sprintf('SQRT(SUMSQ(F%d,J%d))', r, r), 2);
r = r + 1;  rp = r;  info.rows.Payload_WFE = r;
W = put_(W, r, 2, 's', 'Payload WFE', 1);
W = put_(W, r, 11, 'f', sprintf('SQRT(SUMSQ(K%d,K%d))', rt, rd), 3);
r = r + 1;  rq = r;  info.rows.WFE_Requirement = r;
W = put_(W, r, 2, 's', 'WFE Requirement', 1);
if isfinite(o.requirement), W = put_(W, r, 3, 'n', o.requirement, 0); end
W = put_(W, r, 4, 's', 'nm', 0);
W = put_(W, r, 11, 'f', sprintf('C%d', r), 2);
r = r + 1;  info.rows.WFE_Margin = r;
W = put_(W, r, 2, 's', 'WFE Margin', 1);
W = put_(W, r, 11, 'f', sprintf('IF(K%d>0,1-(K%d/K%d),"")', rq, rp, rq), 3);
info.payload_cell = sprintf('K%d', rp);

% ---- WFE BD: the roll-up tree -------------------------------------------
B = cells_();
B = put_(B, 1, 1, 's', [char(o.title) ' -- WFE block diagram'], 1);
B = put_(B, 3, 1, 's', 'Total (Payload) WFE, nm', 1);
B = put_(B, 3, 2, 'f', sprintf('WFE!K%d', rp), 3);
B = put_(B, 4, 1, 's', '  Static (post-WFC), nm', 0);
B = put_(B, 4, 2, 'f', sprintf('SQRT(SUMSQ(%s))', ranges_(rows, 'WFE!F')), 2);
B = put_(B, 5, 1, 's', '  Drift (post-metrology), nm', 0);
B = put_(B, 5, 2, 'f', sprintf('SQRT(SUMSQ(%s))', ranges_(rows, 'WFE!J')), 2);
B = put_(B, 6, 1, 's', '  Design, nm', 0);
B = put_(B, 6, 2, 'f', sprintf('WFE!K%d', rd), 2);
B = put_(B, 7, 1, 's', '  Reserve vs requirement, nm', 0);
B = put_(B, 7, 2, 'f', sprintf('IF(WFE!K%d>WFE!K%d,SQRT(WFE!K%d^2-WFE!K%d^2),0)', ...
    rq, rp, rq, rp), 2);
B = put_(B, 9, 1, 's', 'Wavelength, nm', 0);
B = put_(B, 9, 2, 'n', o.wavelength_nm, 0);
B = put_(B, 10, 1, 's', 'Total, waves', 0);
B = put_(B, 10, 2, 'f', 'B3/B9', 2);
B = put_(B, 12, 1, 's', 'Section', 1);
B = put_(B, 12, 2, 's', 'Static, nm', 1);
B = put_(B, 12, 3, 's', 'Drift, nm', 1);
B = put_(B, 12, 4, 's', 'Total, nm', 1);
for s = 1:numel(secs)
    in = rows(strcmp({L.section}, secs{s}));
    B = put_(B, 12 + s, 1, 's', secs{s}, 0);
    B = put_(B, 12 + s, 2, 'f', sprintf('SQRT(SUMSQ(%s))', ranges_(in, 'WFE!F')), 2);
    B = put_(B, 12 + s, 3, 'f', sprintf('SQRT(SUMSQ(%s))', ranges_(in, 'WFE!J')), 2);
    B = put_(B, 12 + s, 4, 'f', sprintf('SQRT(SUMSQ(%s))', ranges_(in, 'WFE!K')), 2);
end

% ---- stubs + log ---------------------------------------------------------
P = cells_();
P = put_(P, 1, 1, 's', 'PSD figure terms (engineer-owned).  Link a WFE row to H2 (pre) / H3 (post), as the HabEx sheet does.', 1);
P = put_(P, 2, 7, 's', 'pre coefficient', 0);   P = put_(P, 2, 8, 'n', 0, 0);
P = put_(P, 3, 7, 's', 'post coefficient', 0);  P = put_(P, 3, 8, 'n', 0, 0);
S = cells_();
S = put_(S, 1, 1, 's', 'Strain / thermal terms (engineer-owned).  Link WFE rows to the coefficient cells below.', 1);
S = put_(S, 7, 1, 's', 'lateral gradient pre', 0);  S = put_(S, 7, 2, 'n', 0, 0);
S = put_(S, 8, 1, 's', 'lateral gradient post', 0); S = put_(S, 8, 2, 'n', 0, 0);
S = put_(S, 26, 1, 's', 'thru-thickness pre', 0);   S = put_(S, 26, 2, 'n', 0, 0);
S = put_(S, 27, 1, 's', 'thru-thickness post', 0);  S = put_(S, 27, 2, 'n', 0, 0);
G = cells_();
lg = [{sprintf('generated %s by macos.budget.write_workbook', datestr(now, 31)), ... %#ok<TNOW1,DATST>
    sprintf('lines %d, sections %d, focus section "%s"', n, numel(secs), focus), ...
    sprintf('Total WFE (tool) %.10g nm; post-WFC %.10g; post-metrology %.10g', ...
        coef.total, coef.total_post_wfc, coef.total_post_met), ...
    sprintf('controlled %d, metrology stage %d, metric %s, cbm %g', ...
        coef.meta.controlled, coef.meta.has_met, char(coef.meta.metric), coef.meta.cbm), ...
    'method: Redding 2014 (SPIE), linearized ray-trace covariance budget'}, ...
    wfs_log_(L, rows), o.log(:)'];
for q = 1:numel(lg), G = put_(G, q, 1, 's', lg{q}, 0); end

% ---- write the package --------------------------------------------------
sheets = {'WFE', W, [6 34 12 9 12 12 12 9 13 13 14]; ...
    'Coefficients', C, [6 34 9 12 12 9 12 12 2 22 14 22 14 2 22 14 22 14 2 12 12]; ...
    'WFE BD', B, [34 14 14 14]; 'PSD', P, []; 'StrTherm', S, [24 12]; 'Log', G, 100};
write_xlsx_parts_(path, sheets);
info.path = path;
fprintf('wrote %s (%d lines, live formulas; tool Total WFE %.4g nm)\n', ...
    path, n, coef.total);
end

% =========================================================================

function lg = wfs_log_(L, rows)
% Log lines for the WFS sensing row(s): the default reading and the one
% cell that switches to the HabEx reading (Dave 2026-10-16).
lg = {};
for i = find(strcmp({L.kind}, 'wfs'))
    lg{end+1} = sprintf(['WFS: "%s" (WFE row %d) sigma %g %s is WHITE per-sample ' ...
        'noise (Redding 2014 Eq. 32, the default ruled 2026-10-16): post-WFC ' ...
        'coefficient Coefficients!M%d = %.6g = sqrt(rank/n_w), Post WFC %.4g nm.'], ...
        L(i).name, rows(i), L(i).sigma_eng, L(i).units, rows(i), L(i).coef_post, L(i).wfe); %#ok<AGROW>
    lg{end+1} = sprintf(['WFS, HabEx reading (alpha = 1, the sigma passes whole): ' ...
        'type 1 in Coefficients!M%d; the payload re-rolls.'], rows(i)); %#ok<AGROW>
end
end

function W = sum_row_(W, r, label, rngE)
% one summary row: SQRT(SUMSQ(...)) over the same line rows in E F I J K
W = put_(W, r, 2, 's', label, 1);
if isempty(rngE), return; end
for c = [5 6 9 10 11]
    L = char('A' + c - 1);
    W = put_(W, r, c, 'f', sprintf('SQRT(SUMSQ(%s))', strrep(rngE, '#', L)), 3);
end
end

function s = ranges_(rows, prefix)
% contiguous runs of ROWS as 'E4:E9,E12:E12' ('#' as the column when no
% prefix is given; PREFIX like 'WFE!F' otherwise)
if nargin < 2, prefix = '#'; end
s = '';
if isempty(rows), return; end
rows = sort(rows(:))';
br = [0, find(diff(rows) > 1), numel(rows)];
parts = cell(1, numel(br) - 1);
for q = 1:numel(br) - 1
    a = rows(br(q) + 1);  b = rows(br(q + 1));
    parts{q} = sprintf('%s%d:%s%d', prefix, a, strrep(prefix, 'WFE!', ''), b);
end
s = strjoin(parts, ',');
end

function k = fieldkey_(s)
k = regexprep(s, '[^A-Za-z0-9]', '_');
if ~isletter(k(1)), k = ['s_' k]; end
end

function d = datenum_now_()
d = floor(now) - datenum(1899, 12, 30);          %#ok<TNOW1> Excel serial
end

function S = cells_()
S = struct('r', {}, 'c', {}, 't', {}, 'v', {}, 's', {});
end

function S = put_(S, r, c, t, v, st)
% t: 's' string, 'n' number, 'f' formula; st: style 0 plain, 1 bold,
% 2 grey (computed), 3 grey bold
if strcmp(t, 'n') && ~isfinite(v), return; end
S(end+1) = struct('r', r, 'c', c, 't', t, 'v', v, 's', st);
end

function write_xlsx_parts_(path, sheets)
tmp = tempname;  mkdir(tmp);  cl = onCleanup(@() rmdir(tmp, 's'));
mkdir(fullfile(tmp, '_rels'));  mkdir(fullfile(tmp, 'xl'));
mkdir(fullfile(tmp, 'xl', '_rels'));  mkdir(fullfile(tmp, 'xl', 'worksheets'));
ns = size(sheets, 1);
main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main';
rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' ...
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' ...
    '<Default Extension="xml" ContentType="application/xml"/>' ...
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' ...
    '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'];
wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
    '<workbook xmlns="' main '" xmlns:r="' rel '"><sheets>'];
wr = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'];
for k = 1:ns
    ct = [ct sprintf(['<Override PartName="/xl/worksheets/sheet%d.xml" ContentType=' ...
        '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'], k)]; %#ok<AGROW>
    wb = [wb sprintf('<sheet name="%s" sheetId="%d" r:id="rId%d"/>', esc_(sheets{k, 1}), k, k)]; %#ok<AGROW>
    wr = [wr sprintf(['<Relationship Id="rId%d" Type="%s/worksheet" ' ...
        'Target="worksheets/sheet%d.xml"/>'], k, rel, k)]; %#ok<AGROW>
    put_file_(fullfile(tmp, 'xl', 'worksheets', sprintf('sheet%d.xml', k)), ...
        sheet_xml_(sheets{k, 2}, sheets{k, 3}, main));
end
ct = [ct '</Types>'];
wb = [wb '</sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>'];
wr = [wr sprintf('<Relationship Id="rId%d" Type="%s/styles" Target="styles.xml"/>', ...
    ns + 1, rel) '</Relationships>'];
put_file_(fullfile(tmp, '[Content_Types].xml'), ct);
put_file_(fullfile(tmp, '_rels', '.rels'), ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' ...
    '<Relationship Id="rId1" Type="' rel '/officeDocument" Target="xl/workbook.xml"/>' ...
    '</Relationships>']);
put_file_(fullfile(tmp, 'xl', 'workbook.xml'), wb);
put_file_(fullfile(tmp, 'xl', '_rels', 'workbook.xml.rels'), wr);
put_file_(fullfile(tmp, 'xl', 'styles.xml'), styles_xml_(main));
z = [tempname '.zip'];
parts = [{'[Content_Types].xml', '_rels/.rels', 'xl/workbook.xml', ...
    'xl/_rels/workbook.xml.rels', 'xl/styles.xml'}, ...
    arrayfun(@(k) sprintf('xl/worksheets/sheet%d.xml', k), 1:ns, 'UniformOutput', false)];
zip(z, parts, tmp);
if exist(path, 'file'), delete(path); end
movefile(z, path);
end

function x = sheet_xml_(S, widths, main)
x = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="' main '">'];
if ~isempty(widths)
    x = [x '<cols>'];
    for c = 1:numel(widths)
        x = [x sprintf('<col min="%d" max="%d" width="%g" customWidth="1"/>', c, c, widths(c))]; %#ok<AGROW>
    end
    x = [x '</cols>'];
end
x = [x '<sheetData>'];
if ~isempty(S)
    [~, o] = sortrows([[S.r]' [S.c]']);
    S = S(o);
    rr = [S.r];
    for r = unique(rr)
        x = [x sprintf('<row r="%d">', r)]; %#ok<AGROW>
        for e = S(rr == r)
            ref = sprintf('%s%d', col_(e.c), e.r);
            st = '';  if e.s > 0, st = sprintf(' s="%d"', e.s); end
            switch e.t
                case 's'
                    x = [x sprintf('<c r="%s"%s t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>', ...
                        ref, st, esc_(e.v))]; %#ok<AGROW>
                case 'n'
                    x = [x sprintf('<c r="%s"%s><v>%.17g</v></c>', ref, st, e.v)]; %#ok<AGROW>
                case 'f'
                    x = [x sprintf('<c r="%s"%s><f>%s</f></c>', ref, st, esc_(e.v))]; %#ok<AGROW>
            end
        end
        x = [x '</row>']; %#ok<AGROW>
    end
end
x = [x '</sheetData></worksheet>'];
end

function x = styles_xml_(main)
x = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="' main '">' ...
    '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>' ...
    '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>' ...
    '<fills count="3"><fill><patternFill patternType="none"/></fill>' ...
    '<fill><patternFill patternType="gray125"/></fill>' ...
    '<fill><patternFill patternType="solid"><fgColor rgb="FFD9D9D9"/><bgColor indexed="64"/></patternFill></fill></fills>' ...
    '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>' ...
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>' ...
    '<cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>' ...
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>' ...
    '<xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1"/>' ...
    '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>' ...
    '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'];
end

function s = col_(c)
s = '';
while c > 0
    m = mod(c - 1, 26);
    s = [char('A' + m) s]; %#ok<AGROW>
    c = (c - m - 1) / 26;
end
end

function s = esc_(s)
s = char(s);
s = strrep(s, '&', '&amp;');  s = strrep(s, '<', '&lt;');
s = strrep(s, '>', '&gt;');   s = strrep(s, '"', '&quot;');
end

function put_file_(f, txt)
fid = fopen(f, 'w', 'n', 'UTF-8');  fwrite(fid, unicode2native(txt, 'UTF-8'));  fclose(fid);
end
