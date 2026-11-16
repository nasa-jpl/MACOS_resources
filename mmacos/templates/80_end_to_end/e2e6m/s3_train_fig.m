function s3_train_fig()
%S3_TRAIN_FIG  Layout renders for the S3 back end (the layout gate).
%
%   Two figures:
%     s3_train_iso.png  the full train, s3_seg_full.in -- iso + side.
%     s3_back_iso.png   the BACK END ALONE, s3_seg_back.in.  Needed
%        because the relay mirrors are centimetre-class against a 6 m
%        primary: in a full-train render they are labels with no visible
%        body.  Same model, its own scale.
%
%   Reads nothing but the committed deck; changes no state on disk.

here = fileparts(mfilename('fullpath'));
setup_(here);

macos.init(256);
one_(here, 's3_seg_full.in',  's3_train_iso.png', ...
     'full train: segmented telescope + back end');
one_(here, 's3_seg_back.in',  's3_back_iso.png', ...
     'back end alone: four-mirror relay, telescope focus to detector');
end

function one_(here, rxname, pngname, label)
rx = fullfile(here, rxname);
macos.load_rx(rx);
n = macos.num_elt();
png = fullfile(here, pngname);
macos.view_std('front', [], 'back', [], ...
               'iso', [-35 22], 'side', [0 0], ...
               'title', sprintf('%s (%d elts)', label, n), ...
               'save', png, 'visible', false);
d = dir(png);
fprintf('s3_train_fig: %s -- %d elements -> %s (%d bytes)\n', ...
        rxname, n, pngname, d.bytes);
end

function setup_(here)
run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
end
