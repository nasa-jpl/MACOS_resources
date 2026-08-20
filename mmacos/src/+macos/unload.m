function unload()
%MACOS.UNLOAD  Release the engine's memory back to its minimum footprint.
%   macos.unload() re-initialises the engine at the smallest supported
%   model size (128), which deallocates and rebuilds every model-sized
%   array.  Use it when a big run is finished and you want the memory back
%   WITHOUT quitting MATLAB.  The engine stays usable: call macos.init(N)
%   (or construct a new macos.Session) and carry on.
%
%   Measured on this box (MATLAB R2026a, gfortran mex), FFSegDemoAll at
%   model 256, resident set from /proc/self/statm:
%       at model 256 .............. 1561 MB
%       after macos.unload() ...... 1007 MB      (554 MB returned)
%   repeatable over four cycles with no accumulation, and the OPD after
%   re-init is bit-identical to the pre-unload map.
%
%   WHAT IT DOES NOT DO -- and why you should not reach for `clear mex`.
%   `clear('mmacos')` unloads the mex DSO but does NOT free the engine's
%   Fortran module allocatables: nothing deallocates them before the
%   library goes away, so the heap blocks are orphaned.  Measured, same
%   fixture: four load / `clear mmacos` / reload cycles grew the process
%   by ~720 MB EACH (1582 -> 2303 -> 3023 -> 3742 MB) while `clear`
%   itself returned 1.6 MB.  Calling macos.unload() first cuts that to
%   ~190 MB per cycle, but it is still a leak.  Do not clear the mex in a
%   loop.  A true free needs an engine-side deallocate-all hook (there is
%   none today) -- see PLAN.md.
%
%   NO-OP AT THE MINIMUM.  The engine only rebuilds when the requested
%   size differs from the current one, so calling unload() on a session
%   that is already at model 128 does nothing -- there is nothing large to
%   release, and the loaded prescription stays loaded.
%
%   See also: macos.init, macos.Session.
macos.init(macos.model_size_min());
end
