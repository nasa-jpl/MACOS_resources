function nElt = load_rx(rx_path)
%MACOS.LOAD_RX  Load a prescription file (.in) into the macos engine.
%   N = macos.load_rx(RX_PATH) loads the prescription at RX_PATH and
%   returns the number of elements N (>0).  RX_PATH may be supplied
%   with or without the trailing '.in' — macos's OLD command appends
%   '.in' unconditionally, so we strip a trailing '.in' if present
%   (otherwise macos tries to open 'foo.in.in').
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
nElt = mmacos('load_rx', rx_path);
end
