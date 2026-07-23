function txt = ideal_lens_emit(L, iElt0, vpt_front, psi, opts)
%MACOS.DESIGN.IDEAL_LENS_EMIT  Render an ideal_lens spec to Rx element text.
%   txt = macos.design.ideal_lens_emit(L, IELT0, VPT_FRONT, PSI) returns
%   the two Refractor element blocks (front + powered, in light order)
%   for the lens spec L (from macos.design.ideal_lens), numbered starting
%   at IELT0, with the FRONT vertex at VPT_FRONT (1x3) and optical axis
%   PSI (1x3 unit vector, default [0 0 1]).  The second surface is placed
%   L.thickness along PSI from the front vertex.
%
%   Name-value:
%     'fmt'  printf conversion for reals (default '%.10E').
%
%   The returned text is a char array of complete element blocks (each
%   terminated by nECoord), ready to concatenate into an Rx between the
%   source block and the nOutCord/Tout terminator.
%
%   See also: macos.design.ideal_lens.

arguments
    L         (1,1) struct
    iElt0     (1,1) double {mustBeInteger, mustBePositive}
    vpt_front (1,3) double
    psi       (1,3) double = [0 0 1]
    opts.fmt  (1,:) char = '%.10E'
end
psi = psi(:)' / norm(psi);
F = opts.fmt;
lines = {};
for k = 1:numel(L.surf)
    s = L.surf(k);
    vpt = vpt_front(:)' + s.dz * psi;
    ie  = iElt0 + (k-1);
    lines{end+1} = sprintf('             iElt=  %d', ie);                       %#ok<AGROW>
    lines{end+1} = sprintf('          EltName=  %s', s.name);                   %#ok<AGROW>
    lines{end+1} = '          Element=  Refractor';                            %#ok<AGROW>
    lines{end+1} = sprintf('          Surface=  %s', s.surface);               %#ok<AGROW>
    lines{end+1} = sprintf(['            KrElt=  ' F], s.Kr);                   %#ok<AGROW>
    lines{end+1} = sprintf(['            KcElt=  ' F], s.Kc);                   %#ok<AGROW>
    lines{end+1} = sprintf('           psiElt=  %.10E  %.10E  %.10E', psi);     %#ok<AGROW>
    lines{end+1} = sprintf('           VptElt=  %.10E  %.10E  %.10E', vpt);     %#ok<AGROW>
    lines{end+1} = sprintf('           RptElt=  %.10E  %.10E  %.10E', vpt);     %#ok<AGROW>
    lines{end+1} = sprintf(['           IndRef=  ' F], s.indref);              %#ok<AGROW>
    lines{end+1} = '           Extinc=  0.0D+00';                              %#ok<AGROW>
    lines{end+1} = '            nCoat=  0';                                    %#ok<AGROW>
    lines{end+1} = '             xObs=  1.0D+00  0.0D+00  0.0D+00';            %#ok<AGROW>
    lines{end+1} = '             nObs=  0';                                    %#ok<AGROW>
    lines{end+1} = '           ApType=  Circular';                            %#ok<AGROW>
    lines{end+1} = sprintf(['            ApVec=  ' F '  0.0D+00  0.0D+00'], s.ap); %#ok<AGROW>
    lines{end+1} = '         PropType=  Geometric';                           %#ok<AGROW>
    lines{end+1} = '             zElt=  1.0D+22';                             %#ok<AGROW>
    lines{end+1} = '          nECoord=  -6';                                  %#ok<AGROW>
    lines{end+1} = '';                                                        %#ok<AGROW>
end
txt = strjoin(lines, newline);
end
