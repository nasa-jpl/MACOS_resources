function M = read_grid_file(path)
%MACOS.READ_GRID_FILE  Read a MACOS GridFile (ASCII) into GridMat orientation.
%   M = macos.read_grid_file(PATH) reads an N x N grid-data file the SAME way
%   the engine's GridInit (surfsub.F) does -- text line j fills GridMat COLUMN
%   j -- and returns M in the engine's GridMat(i,j) convention (first index =
%   +x, second index = +y).  It is the exact inverse of macos.write_grid_file,
%   and pairs with macos.elt_grid_add:
%
%       M = macos.read_grid_file('oldGridData.txt');
%       macos.elt_grid_add(iElt, M);     % == loading GridFile= oldGridData.txt
%
%   with NO manual transpose.  This resolves the recurring surprise that a raw
%   MATLAB read of a GridFile comes out TRANSPOSED: readmatrix / load read the
%   file line = matrix ROW, but the engine reads line = matrix COLUMN, so
%
%       readmatrix(PATH) == read_grid_file(PATH).'
%
%   i.e. a naive load must be transposed -- rot90(fliplr(...)), an exact
%   transpose -- before macos.elt_grid_add.  read_grid_file does that for you.
%
%   CROSS-BINDING NOTE: pymacos.elt_grid_add uses the OPPOSITE [y,x] convention
%   (it transposes internally), so pymacos users feed it a plain numpy read of
%   the file.  mmacos keeps the engine's physical [x,y] convention, so use this
%   helper (or macos.write_grid_file to emit one) rather than a bare load.
%
%   See also: macos.write_grid_file, macos.elt_grid_add,
%             macos.zernike_grid_basis.
arguments
    path (1,:) char
end
fid = fopen(path, 'r');
if fid < 0
    error('macos:read_grid_file:open', 'cannot open %s for reading', path);
end
closer = onCleanup(@() fclose(fid));
raw = fscanf(fid, '%g');                  % all numbers, file (row-major) order
n2 = numel(raw);
N  = round(sqrt(n2));
if N*N ~= n2
    error('macos:read_grid_file:notSquare', ...
          '%s holds %d values, not a perfect square (N x N)', path, n2);
end
% raw is the file in row-major order; a column-major reshape of a row-major
% stream is exactly the transpose of the naive [line=row] matrix -- which is
% the engine's [line=column] GridMat(i,j) layout.
M = reshape(raw, N, N);
end
