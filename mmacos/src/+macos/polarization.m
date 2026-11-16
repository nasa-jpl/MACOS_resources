function out = polarization(state, opts)
%MACOS.POLARIZATION  Turn polarized ray tracing on/off + set source state.
%   macos.polarization('on', ...) enables polarized ray tracing (the
%   engine POLARIZATION command): rays carry a complex 3-vector E-field,
%   surface coatings become active, and vector diffraction is enabled
%   when the model supports it (mWF>=3, true for all stock model sizes).
%   macos.polarization('off') restores scalar tracing (NOPOLARIZATION).
%
%   The source Jones state (Ex0,Ey0) is set from name-value pairs:
%     'Ex'  [re im]  source Ex complex amplitude (default [1 0])
%     'Ey'  [re im]  source Ey complex amplitude (default [0 0])
%   Scalars are treated as the real part (imag 0).
%
%   S = macos.polarization() with NO arguments QUERIES the state and
%   returns a struct: .on (logical), .vector (logical, vector diffraction),
%   .Ex (complex), .Ey (complex).
%
%   Examples:
%     macos.polarization('on');                   % x-polarized (default)
%     macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
%     macos.polarization('on', 'Ex', [1 0], 'Ey', [0 1]);  % circular
%     macos.polarization('off');
%     s = macos.polarization();                   % query
%
%   See also: macos.vector_diffraction, macos.coating, macos.ray_field.
arguments
    state (1,:) char {mustBeMember(state, {'on','off',''})} = ''
    opts.Ex double = [1 0]
    opts.Ey double = [0 0]
end

% ---- query mode ------------------------------------------------------
if isempty(state)
    [onc, vecc, exre, exim, eyre, eyim] = mmacos('pol_get');
    out.on     = onc ~= 0;
    out.vector = vecc ~= 0;
    out.Ex     = complex(exre, exim);
    out.Ey     = complex(eyre, eyim);
    return
end

% ---- set mode --------------------------------------------------------
if strcmp(state, 'off')
    mmacos('pol_set', 0, 0, 0, 0, 0);
    return
end

ex = opts.Ex(:).';  if isscalar(ex), ex = [ex 0]; end
ey = opts.Ey(:).';  if isscalar(ey), ey = [ey 0]; end
if numel(ex) ~= 2 || numel(ey) ~= 2
    error('macos:polarization:badState', ...
        '''Ex''/''Ey'' must be [re im] (or a real scalar).');
end
mmacos('pol_set', 1, ex(1), ex(2), ey(1), ey(2));
end
