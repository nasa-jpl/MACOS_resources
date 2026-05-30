function p = rx_mask_params(tag)
%RX_MASK_PARAMS  Per-Rx test fixture for the mask test suite.
%   Mirrors pymacos/tests/test_masks.py:42 RX_PARAMS dict.
%
%   p = rx_mask_params('parabola')      returns the Rx_Mask_Parabolas
%                                       config (image-plane local frame).
%   p = rx_mask_params('parabola_glb')  returns the global-frame variant.
%
%   Returned struct fields:
%       .rx_path        absolute path to the source .in file
%       .srfs           struct keyed by surface id (S2,S4 for parabola;
%                       S3,S5 for parabola_glb) with per-surface info:
%           .dx_fact    sign of dx in ApVec (psiElt vs TElt_z convention)
%           .line_id    1-based line number of the "ApVec=" line.
%                       ApType=  lives at line_id-1.
%           .line_id_obs 1-based line number of the "nObs=" line.
%
%   Note: pymacos stores these as 0-based indices (line_id = 99 means
%   ApVec is at lines[99] in 0-based slice).  We store 1-based directly
%   (line_id = 100), matching MATLAB cellstr indexing.
arguments
    tag (1,:) char {mustBeMember(tag, {'parabola','parabola_glb'})}
end
switch tag
    case 'parabola'
        p.rx_path = rx_fixture_path('Rx_Mask_Parabolas.in');
        p.srfs.S2.dx_fact     = -1;
        p.srfs.S2.line_id     = 100;   % ApVec line; ApType at 99
        p.srfs.S2.line_id_obs = 101;   % nObs line
        p.srfs.S4.dx_fact     = +1;
        p.srfs.S4.line_id     = 149;
        p.srfs.S4.line_id_obs = 150;
    case 'parabola_glb'
        p.rx_path = rx_fixture_path('Rx_Mask_Parabolas_glb.in');
        p.srfs.S3.dx_fact     = -1;
        p.srfs.S3.line_id     = 124;
        p.srfs.S3.line_id_obs = 125;
        p.srfs.S5.dx_fact     = +1;
        p.srfs.S5.line_id     = 173;
        p.srfs.S5.line_id_obs = 174;
end
end
