function stop(iElt, vpt_offset)
%MACOS.STOP  Set the system aperture stop to an element.
%   macos.stop(IELT) declares element IELT as the system Stop.  The
%   chief ray is then aimed to pass through the centre of that
%   element (VptElt(IELT) + offset).
%
%   macos.stop(IELT, [DX DY]) places the stop at a local-frame offset
%   from VptElt(IELT) -- DX along xElt, DY along yElt, both in
%   BaseUnits.  Use this when the prescription Stop point isn't the
%   element vertex.
%
%   See also: macos.stop_obj, macos.sxp.
arguments
    iElt       (1,1) double {mustBeInteger, mustBePositive}
    vpt_offset (1,2) double = [0 0]
end
mmacos('stop_info_set', iElt, vpt_offset);
end
