function write_grid_file(path, M)
%MACOS.WRITE_GRID_FILE  Write an N x N grid to a MACOS GridFile (ASCII).
%   macos.write_grid_file(PATH, M) writes the [N x N] matrix M to a formatted
%   grid-data file that the engine's GridInit (surfsub.F) reads back into
%   GridMat with NO transpose.  The reader is
%       DO j=1,N:  READ (GridMat(i,j), i=1,N)
%   i.e. file line j is GridMat column j.  fprintf serialises M column-major,
%   so writing N values per line reproduces exactly that ordering -- the engine
%   reads GridMat == M, in the (first index = +x, second index = +y) convention
%   of macos.zernike_grid_basis / macos.elt_grid_add.  Point a segment's
%   GridFile= at this file to load M as that segment's grid figure.
%
%   See also: macos.segment_grid_basis, macos.elt_grid_add,
%             macos.zernike_grid_basis.
arguments
    path (1,:) char
    M    (:,:) double {mustBeReal, mustBeFinite}
end
[nr, nc] = size(M);
if nr ~= nc
    error('macos:write_grid_file:notSquare', ...
          'M must be square [N x N] (got [%d x %d])', nr, nc);
end
fid = fopen(path, 'w');
if fid < 0
    error('macos:write_grid_file:open', 'cannot open %s for writing', path);
end
closer = onCleanup(@() fclose(fid));
fmt = [repmat('%16.8E', 1, nc), '\n'];   % N values per line; line j == column j
fprintf(fid, fmt, M);                    % fprintf walks M column-major -> GridMat==M
end
