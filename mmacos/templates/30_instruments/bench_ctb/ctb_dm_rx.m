function out = ctb_dm_rx(opts)
%CTB_DM_RX  Emit a DM-augmented CTB deck (grid-data DM surfaces).
%   out = CTB_DM_RX() reads the committed compact deck (ctb_dcr.in) and
%   writes ctb_dm.in beside it, with the two DM blocks rewritten as
%   grid-data surfaces so actuator figures can be applied at runtime
%   (macos.set_elt_grid / macos.elt_grid_add):
%     Surface= Flat  ->  Surface= GridData
%     + nGridMat / GridFile / GridSrfdx / pData / xData / yData / zData
%   The grid coordinate frame is built from the DM element's OWN frame
%   (pData=VptElt, xData=xObs, zData=psiElt, yData=z-cross-x) -- the
%   frame rule that makes pokes localize (the e5 "central dot" lesson:
%   a grid channel in the wrong frame paints responses about the
%   aperture center; see macos.design.grid_augment_rx).
%
%   The committed hand decks are never touched; the emitted deck is a
%   derived artifact (regenerate with this script, do not hand-edit).
%
%   Name-value:
%     'rx_in'    source deck (default ctb_dcr.in in this dir)
%     'rx_out'   output deck (default ctb_dm.in beside rx_in)
%     'dms'      element names to augment (default {'DM1','DM2'})
%     'ng'       nGridMat (default 256; model_size must be >= ng)
%     'span_mm'  grid span, base units (default 2*ApVec(1) of each DM:
%                the full element aperture; rays outside get no figure)
%     'gridfile' flat seed grid name (default 'flat<ng>.txt', written
%                beside rx_out if missing -- the engine resolves
%                GridFile= against the cwd at load time)
%
%   out: rx_out, dms, ielt (per DM), ng, gdx_mm (per DM), span_mm (per
%   DM), gridfile.
%
%   Run:  >> out = ctb_dm_rx;
%   See also: ctb_dm, macos.elt_grid_add, macos.design.grid_augment_rx.
    arguments
        opts.rx_in    (1,:) char = ''
        opts.rx_out   (1,:) char = ''
        opts.dms      (1,:) cell = {'DM1','DM2'}
        opts.ng       (1,1) double {mustBeInteger, mustBePositive} = 256
        opts.span_mm  double = []
        opts.gridfile (1,:) char = ''
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.rx_in),  opts.rx_in  = fullfile(here, 'ctb_dcr.in'); end
    if isempty(opts.rx_out), opts.rx_out = fullfile(here, 'ctb_dm.in');  end
    if isempty(opts.gridfile)
        opts.gridfile = sprintf('flat%d.txt', opts.ng);
    end
    assert(isfile(opts.rx_in), 'ctb_dm_rx: %s not found', opts.rx_in);

    GRID_KEYS = ["nGridMat" "GridFile" "GridSrfdx" "GridType" ...
                 "pData" "xData" "yData" "zData" "lData"];

    L = splitlines(string(fileread(opts.rx_in)));
    outL = strings(0, 1);
    indm = false;  cur = struct();  dmn = 0;
    ielt = zeros(1, numel(opts.dms));
    gdx_mm = zeros(1, numel(opts.dms));
    span_mm = zeros(1, numel(opts.dms));
    names = strings(1, numel(opts.dms));
    lastIElt = NaN;

    for i = 1:numel(L)
        ln = L(i);
        tl = strtrim(ln);
        if startsWith(tl, 'iElt=')
            lastIElt = sscanf(extractAfter(char(tl), '='), '%d');
            indm = false;
        end
        if startsWith(tl, 'EltName=')
            nm = strtrim(extractAfter(tl, '='));
            indm = any(strcmp(char(nm), opts.dms));
            if indm
                dmn = dmn + 1;
                names(dmn) = nm;
                ielt(dmn) = lastIElt;
                cur = struct();
            end
        end
        if indm
            % dispatch on key; grid lines are emitted before PropType=
            key = strtrim(extractBefore(tl, '='));
            if ~ismissing(key) && any(key == GRID_KEYS)
                continue;                       % stale grid line: REPLACED
            end
            for k = ["psiElt" "VptElt" "xObs" "ApVec"]
                if startsWith(tl, k + "=")
                    cur.(char(k)) = sscanf(strrep(upper( ...
                        char(extractAfter(tl, '='))), 'D', 'E'), '%g').';
                end
            end
            if startsWith(tl, 'Surface=')
                srf = strtrim(extractAfter(tl, '='));
                assert(srf == "Flat", ...
                    'ctb_dm_rx: %s Surface= %s (expected Flat)', names(dmn), srf);
                outL(end+1) = "          Surface=  GridData"; %#ok<AGROW>
                continue;
            end
            if startsWith(tl, 'PropType=')
                % all frame keys are parsed by now -- emit the grid block
                assert(all(isfield(cur, {'psiElt','VptElt','xObs','ApVec'})), ...
                    'ctb_dm_rx: %s block lacks psiElt/VptElt/xObs/ApVec', names(dmn));
                z = cur.psiElt(1:3);  z = z / norm(z);
                x = cur.xObs(1:3);    x = x / norm(x);
                y = cross(z, x);      y = y / norm(y);
                if isempty(opts.span_mm), sp = 2*cur.ApVec(1);
                else,                     sp = opts.span_mm; end
                g = sp / (opts.ng - 1);
                span_mm(dmn) = sp;  gdx_mm(dmn) = g;
                outL = [outL(:)
                    sprintf("         nGridMat=  %d", opts.ng)
                    "         GridFile=  " + string(opts.gridfile)
                    sprintf("        GridSrfdx=%.17G", g)
                    sprintf("            pData=  %.17G  %.17G  %.17G", cur.VptElt(1:3))
                    sprintf("            xData=  %.17G  %.17G  %.17G", x)
                    sprintf("            yData=  %.17G  %.17G  %.17G", y)
                    sprintf("            zData=  %.17G  %.17G  %.17G", z)]; %#ok<AGROW>
            end
        end
        outL(end+1) = ln; %#ok<AGROW>
    end
    assert(dmn == numel(opts.dms), ...
        'ctb_dm_rx: found %d of %d DM blocks', dmn, numel(opts.dms));

    fid = fopen(opts.rx_out, 'w');
    assert(fid > 0, 'ctb_dm_rx: cannot open %s', opts.rx_out);
    fprintf(fid, '%s\n', outL);
    fclose(fid);

    % flat seed grid beside the emitted deck (load-time cwd rule)
    gf = fullfile(fileparts(opts.rx_out), opts.gridfile);
    if ~isfile(gf)
        macos.write_grid_file(gf, zeros(opts.ng));
    end

    out = struct('rx_out',opts.rx_out, 'dms',{cellstr(names)}, 'ielt',ielt, ...
                 'ng',opts.ng, 'gdx_mm',gdx_mm, 'span_mm',span_mm, ...
                 'gridfile',gf);
    fprintf('[ctb_dm_rx] %s: %s at iElt [%s], ng=%d, gdx=[%s] mm\n', ...
        opts.rx_out, strjoin(cellstr(names), ','), num2str(ielt), ...
        opts.ng, num2str(gdx_mm, '%.4g '));
end
