function ff = find_freeform_elts()
%MACOS.FIND_FREEFORM_ELTS  Indices of FreeForm-typed elements.
%   ff = macos.find_freeform_elts() returns a column vector of 1-based
%   element ids for every FreeForm (SrfType=14) surface in the loaded
%   prescription, in element order.  Empty array if no FreeForm
%   surfaces are present.
%
%   Used by sensitivity-channel constructors to enumerate the
%   eligibility set for MonZern / FFZern channels.
%
%   See also: macos.find_zern_elts.
n = macos.num_elt();
if n <= 0
    ff = zeros(0,1);
    return
end
all_elts = (1:n).';
mask = mmacos('elt_srf_ff_fnd', all_elts, n);
ff = all_elts(logical(mask(:)));
end
