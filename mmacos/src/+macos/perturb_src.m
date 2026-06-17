function perturb_src(opts)
%MACOS.PERTURB_SRC  Rigid-body perturbation of the source (iElt=0).
%   Name-value pairs:
%     'rotation'    [3x1] x,y,z rotation (rad).  Default zeros.
%     'translation' [3x1] x,y,z translation in SI metres.  Converted
%                   internally to BaseUnits via 1/CBM.  Default zeros.
%
%   The (rotation, translation) frame matches the prescription's
%   source-frame setting (LOCAL when SrcLF_FLG is set in the Rx,
%   else GLOBAL).  macos has no per-call frame switch for sources --
%   the choice is baked into the prescription.
arguments
    opts.rotation    (3,1) double = zeros(3,1)
    opts.translation (3,1) double = zeros(3,1)
end
c = mmacos('base_unit_to_metres');
if c == 0.0
    error('macos:perturb_src:noCBM', ...
        'CBM unavailable (Rx not loaded?) -- cannot convert SI translation.');
end
del_base = opts.translation / c;
mmacos('perturb_src', opts.rotation, del_base);
end
