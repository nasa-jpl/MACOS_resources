function save_rx(rx_path)
%MACOS.SAVE_RX  Write the current macos state to a prescription file.
arguments
    rx_path (1,:) char
end
[~, ~, ext] = fileparts(rx_path);
if strcmpi(ext, '.in')
    rx_path = rx_path(1:end-3);
end
mmacos('save_rx', rx_path);
end
