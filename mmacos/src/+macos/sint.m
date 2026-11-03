function r = sint()
%MACOS.SINT  Read the data of every Interpolated surface and set up its interpolation -- the CLI's SINT.
%   r = macos.sint()
%   For each element with Surface= Interpolated (SrfType 5) the engine reads
%   <deck>.srfN.bin, else .srfN.txt, where <deck> is the loaded deck's name without
%   its '.in' (Rx_TraceRewrites.srf1.txt for element 1) or, second, with it (the CLI's
%   spelling, Rx_TraceRewrites.in.srf1.txt) -- and precomputes the triangular
%   interpolation the trace uses.  Until it is set, such a surface traces
%   as its base conic.  A missing file is not prompted for: that element is not set.
%     r.n_set   how many elements were set
%     r.set     1 x nElt: 1 = set, -1 = an Interpolated surface not set, 0 = not one
%   Why an element was not set (a missing or unreadable file, too few or too many
%   points) is in macos.messages; a file shorter than its header uses the points read,
%   with a warning there.  The trace is marked modified.
%   api sint_cmd (BRIEF_to_bindings B.2, 2026-10-23).
%
%   See also: macos.messages, macos.opd.
ne = macos.num_elt();
[n, s] = mmacos('sint_cmd', zeros(1, ne), double(ne));
macos.modify();
r = struct('n_set', double(n), 'set', reshape(double(s), 1, []));
end
