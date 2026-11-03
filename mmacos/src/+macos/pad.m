function [out] = pad(in, npix_rows, npix_cols)
%MACOS.PAD  Zero-pad (or centre-crop) a matrix to a target size.
%   OUT = macos.pad(IN, NPIX)              square NPIX-by-NPIX result
%   OUT = macos.pad(IN, NPIX_ROWS, NPIX_COLS)
%
%   The input is centred on the output using the floor(n/2+1) centre
%   convention; a target smaller than the input centre-crops instead.
%
%   See also macos.mimg.


   if nargin < 3
      npix_cols = npix_rows;
   end

   out = zeros(npix_rows,npix_cols);

   [nrows, ncols]  = size(in);

   ixc = floor(ncols/2 + 1);
   iyc = floor(nrows/2 + 1);

   oxc = floor(npix_cols/2 + 1);
   oyc = floor(npix_rows/2 + 1);

   dx = npix_cols-ncols;
   dy = npix_rows-nrows;

   if dx<=0
     ix1 = ixc - floor(npix_cols/2);
     ix2 = ix1 + npix_cols - 1;
     ox1 = 1;
     ox2 = npix_cols;
   else
     ix1 = 1;
     ix2 = ncols;
     ox1 = oxc - floor(ncols/2);
     ox2 = ox1 + ncols - 1;
   end
   
   if dy<=0
     iy1 = iyc - floor(npix_rows/2);
     iy2 = iy1 + npix_rows - 1;
     oy1 = 1;
     oy2 = npix_rows;
   else
     iy1 = 1;
     iy2 = nrows;
     oy1 = oyc - floor(nrows/2);
     oy2 = oy1 + nrows - 1;
   end

   out ( oy1:oy2, ox1:ox2)  = in ( iy1:iy2, ix1:ix2);

% Uncomment for testing
%[ixc iyc iy1 iy2 ix1 ix2 ]
%[oxc oyc oy1 oy2 ox1 ox2 ]

return
