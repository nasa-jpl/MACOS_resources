function nElt = load_rx(rx_path)
%MACOS.LOAD_RX  Load a prescription file (.in) into the macos engine.
%   N = macos.load_rx(RX_PATH) loads the prescription at RX_PATH and
%   returns the number of elements N (>0).  RX_PATH may be supplied
%   with or without the trailing '.in' — macos's OLD command appends
%   '.in' unconditionally, so we strip a trailing '.in' if present
%   (otherwise macos tries to open 'foo.in.in').
%
%   Session modes (POLARIZATION, VECTOR, CENTROID, LNEG, CIR) SURVIVE load_rx
%   (2026-10-11); load_rx warns (id mmacos:modesInForce) when any is in force.
%   Read them with macos.modes; set them per deck with macos.polarization /
%   vecdif_set / the xp_fnd mode.
arguments
    rx_path (1,:) char
end
if ~exist(rx_path, 'file')
    error('macos:load_rx:notFound', 'Rx file not found: %s', rx_path);
end
[~, ~, ext] = fileparts(rx_path);
if strcmpi(ext, '.in')
    rx_path = rx_path(1:end-3);
end
try
    nElt = mmacos('load_rx', rx_path);
catch e
    % the engine's own reason (its message buffer, PLAN_CONSOLIDATION 1.2 (ee)); left in the
    % buffer for macos.messages
    why = strtrim(macos.messages(false));
    why = why(contains(why, "**"));
    if isempty(why), rethrow(e); end
    id = e.identifier;  if isempty(id), id = 'macos:load_rx:refused'; end
    throw(MException(id, '%s', sprintf('%s: %s', e.message, strjoin(why(max(1, end-3):end), ' | '))));
end
% Session modes survive load_rx (Dave 2026-10-11, PLAN_CONSOLIDATION 4 item 2):
% say so when any is in force, as the CLI's LOAD does -- a script's log shows it,
% a test can assert or suppress it by id.  Set them per deck with
% macos.polarization / macos.vecdif / the xp_fnd mode.
md = macos.modes();
if ~isempty(md.in_force)
    warning('mmacos:modesInForce', 'load_rx: session modes in force: %s', ...
            strjoin(md.in_force, ', '));
end
end
