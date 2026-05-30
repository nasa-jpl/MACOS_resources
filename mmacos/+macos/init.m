function init(model_size)
%MACOS.INIT  Initialize / resize the macos engine.
%   macos.init(MODEL_SIZE) allocates the engine for prescriptions up to
%   MODEL_SIZE elements / rays / etc.  Must be called once per MATLAB
%   session before any other macos.* call.
%
%   See also: macos.load_rx, macos.has_rx.
arguments
    model_size (1,1) double {mustBeInteger, mustBePositive}
end
mmacos('init', model_size);
end
