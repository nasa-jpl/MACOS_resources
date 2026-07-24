function out = beam(type, opts)
%MACOS.BEAM  Shape the source amplitude (apodization) profile.
%   macos.beam(TYPE, ...) sets the source beam profile that macos applies
%   to the aperture amplitude before tracing.  Setting the beam resets the
%   trace, so re-trace afterwards.
%
%   TYPE (case-insensitive):
%     'uniform'   flat-top (default illumination) -- no parameters
%     'gaussian'  Gaussian waist -- 'waist' [rx ry] required
%     'cos'       cosine**power -- 'radius' R and 'power' P required
%     'dipole'    dipole pattern -- no parameters
%
%   Name-value parameters (by type):
%     'waist'  [rx ry]  Gaussian x/y waist radii, in source BaseUnits
%                       (a scalar is broadcast to [r r]).  GAUSSIAN only.
%     'radius' R        cosine beam radius, in source BaseUnits.  COS only.
%     'power'  P        cosine exponent.                          COS only.
%
%   S = macos.beam() with NO arguments returns the CURRENT profile as a
%   struct: .type (char), .waist [rx ry], .power (cosine exponent).
%
%   Examples:
%     macos.beam('uniform');
%     macos.beam('gaussian', 'waist', [12 12]);   % 12 mm waist (mm Rx)
%     macos.beam('gaussian', 'waist', 8);         % circular 8-unit waist
%     macos.beam('cos', 'radius', 10, 'power', 2);
%     s = macos.beam();                           % query
%
%   See also: macos.window, macos.spot, macos.trace.
arguments
    type    (1,:) char {mustBeMember(type, ...
                {'uniform','gaussian','cos','dipole',''})} = ''
    opts.waist  double = []
    opts.radius (1,1) double {mustBePositive} = 1
    opts.power  (1,1) double = 1
end

% ---- query mode: no type given --------------------------------------
if isempty(type)
    [code, rx, ry, cosPwr] = mmacos('beam_get');
    names = {'uniform','gaussian','cos','dipole'};
    code = round(code);
    if code >= 1 && code <= 4
        out.type = names{code};
    else
        out.type = 'unset';
    end
    out.waist = [rx ry];
    out.power = cosPwr;
    return
end

% ---- set mode --------------------------------------------------------
codes = struct('uniform', 1, 'gaussian', 2, 'cos', 3, 'dipole', 4);
code  = codes.(type);
p1 = 0; p2 = 0;

switch type
    case 'gaussian'
        if isempty(opts.waist)
            error('macos:beam:missingWaist', ...
                  'gaussian beam requires ''waist'' [rx ry].');
        end
        w = opts.waist(:).';
        if isscalar(w), w = [w w]; end
        if numel(w) ~= 2
            error('macos:beam:badWaist', ...
                  '''waist'' must be a scalar or [rx ry].');
        end
        mustBePositive(w);
        p1 = w(1); p2 = w(2);
    case 'cos'
        p1 = opts.radius; p2 = opts.power;
    % 'uniform' / 'dipole' take no parameters
end

mmacos('beam_set', code, p1, p2);
end
