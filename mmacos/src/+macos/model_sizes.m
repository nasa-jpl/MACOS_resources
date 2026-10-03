function s = model_sizes()
%MACOS.MODEL_SIZES  The model sizes the engine accepts.
%   s = macos.model_sizes() returns [128 256 512 1024 2048 4096 8192] --
%   the values param_mod.F's param_mod_init will accept.  ANY OTHER VALUE
%   MAKES THE ENGINE CALL `stop`, which terminates the host MATLAB
%   process; macos.init screens against this list so a typo raises a
%   MATLAB error instead.
s = [128 256 512 1024 2048 4096 8192];
end
