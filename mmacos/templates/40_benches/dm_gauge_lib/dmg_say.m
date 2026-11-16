function dmg_say(rep, fmt, varargin)
%DMG_SAY  Print to console AND the report file.
fprintf(fmt, varargin{:});
fprintf(rep, fmt, varargin{:});
end
