function init(model_size)
%MACOS.INIT  Initialize / resize the macos engine.
%   macos.init(MODEL_SIZE) allocates the engine for prescriptions up to
%   MODEL_SIZE elements / rays / etc.  Must be called once per MATLAB
%   session before any other macos.* call.
%
%   MODEL_SIZE must be one of macos.model_sizes() -- param_mod.F rejects
%   anything else by calling `stop`, which would take the whole MATLAB
%   process down with it, so it is screened here first.
%
%   See also: macos.model_sizes, macos.unload, macos.load_rx, macos.has_rx.
arguments
    model_size (1,1) double {mustBeInteger, mustBePositive}
end
if ~ismember(model_size, macos.model_sizes())
    error('macos:init:badModelSize', ...
        ['model_size must be one of [%s]; got %g.  The engine aborts the ' ...
         'PROCESS on any other value.'], ...
        strjoin(compose('%d', macos.model_sizes()), ' '), model_size);
end
mmacos('init', model_size);
end
