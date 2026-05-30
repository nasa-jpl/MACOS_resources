function perturb(srf, opts)
%MACOS.PERTURB  Rigid-body perturbation of a single element.
%   macos.perturb(SRF, ...) applies a rotation + translation to
%   element SRF via macos's CPERTURB_PROG (the programmatic sibling
%   of the interactive CPERTURB).  Unlike a raw set_elt_vpt, this
%   performs the full bookkeeping: position + orientation, the
%   element's local TElt frame matrix, aperture vector, HOE points,
%   Mon / pData / FF figure-error coordinate frames, metrology points,
%   and any linked-element children.
%
%   Name-value pairs:
%     'rotation'    [3x1] x,y,z rotation perturbation (radians).
%                   Default zeros.
%     'translation' [3x1] x,y,z translation in SI metres.
%                   Default zeros.  Converted internally to BaseUnits
%                   via 1/CBM.
%     'frame'       'local' (default) | 'global'.  Coordinate frame the
%                   rotation/translation are expressed in.
%
%   For multi-element bulk perturbations, see macos.perturb_many.
%   For group perturbations (GPERTURB), see macos.perturb_grp.

arguments
    srf                 (1,1) double {mustBeInteger, mustBePositive}
    opts.rotation       (3,1) double = zeros(3,1)
    opts.translation    (3,1) double = zeros(3,1)
    opts.frame          (1,:) char {mustBeMember(opts.frame, ...
                            {'local','global'})} = 'local'
end

% SI metres -> BaseUnits via CBM
c = mmacos('base_unit_to_metres');
if c == 0.0
    error('macos:perturb:noCBM', ...
        'CBM unavailable (Rx not loaded?) -- cannot convert SI translation.');
end
del_base = opts.translation / c;

use_local = strcmp(opts.frame, 'local');
mmacos('perturb_elt', srf, opts.rotation, del_base, use_local);
end
