classdef tReadGridFile < matlab.unittest.TestCase
%TREADGRIDFILE  macos.read_grid_file reproduces the engine GridFile= reader.
%   Luis (2026-07-16): GridData applied via `GridFile= f.txt` disagreed with
%   the same data fed to mmacos.elt_grid_add unless the array was pre-mangled
%   with rot90(fliplr(...)) -- which is an exact TRANSPOSE.  Root cause: the
%   engine's GridInit reads a text file line = GridMat COLUMN, whereas a bare
%   MATLAB readmatrix/load reads line = ROW, so the two differ by one
%   transpose.  macos.read_grid_file reads the file the engine's way, so
%   elt_grid_add(read_grid_file(f)) == GridFile= f with no manual transpose.
%
%   Engine leg uses FFSegDemoAll.in (ModelSize=256, its segments carry a
%   FreeForm grid loaded from a GridFile=), so this class groups with the
%   256 suite (SUITE_FREEFORM).

    methods (Test)

        function test_write_read_roundtrip(testCase)
            % read_grid_file must exactly invert write_grid_file.
            N = 17;
            M = 1e-3 * reshape(1:N*N, N, N);        % asymmetric, distinct
            testCase.assertNotEqual(M, M.', 'M must be asymmetric');
            f = [tempname '.txt'];
            c = onCleanup(@() delete(f)); %#ok<NASGU>
            macos.write_grid_file(f, M);
            Mr = macos.read_grid_file(f);
            testCase.verifyEqual(Mr, M, 'AbsTol', 1e-7, ...
                'read_grid_file must invert write_grid_file');
        end

        function test_read_is_transpose_of_naive(testCase)
            % A file laid out line = ROW (a hand-made GridData.txt): the
            % engine-convention read is the TRANSPOSE of a naive readmatrix,
            % == Luis's rot90(fliplr(...)) workaround.
            N = 13;
            M = 1e-3 * reshape(1:N*N, N, N);        % asymmetric
            f = [tempname '.txt'];
            c = onCleanup(@() delete(f)); %#ok<NASGU>
            write_naive_(f, M);                     % line r == matrix row r
            Mr = macos.read_grid_file(f);
            Mn = readmatrix(f, 'FileType', 'text');
            testCase.verifyEqual(Mr, Mn.', 'AbsTol', 1e-12, ...
                'read_grid_file == transpose of a naive readmatrix');
            testCase.verifyEqual(Mr, rot90(fliplr(Mn)), 'AbsTol', 1e-12, ...
                'read_grid_file == rot90(fliplr(naive read))');
        end

        function test_engine_gridfile_equivalence(testCase)
            % THE claim: read_grid_file reproduces what `GridFile= f` loads.
            % Stage the FFSegDemoAll fixture + our own asymmetric data file
            % (under the name its Rx references) in a scratch cwd, load, and
            % read the live grid back -- that IS the engine's GridFile
            % ingestion (ground truth).
            N = 256;                                % FFSegDemoAll nGridMat
            M = 1e-8 * ((1:N).' * 1000 + (1:N));    % M(r,c)=1e-8*(1000r+c)
            testCase.assertNotEqual(M, M.', 'M must be asymmetric');

            wd = tempname; mkdir(wd);
            cwd0 = cd(wd);
            cRestore = onCleanup(@() cleanup_(cwd0, wd)); %#ok<NASGU>
            copyfile(rx_fixture_path('FFSegDemoAll.in'), ...
                     fullfile(wd, 'FFSegDemoAll.in'));
            % write M in the naive line = ROW layout of a hand-made file
            write_naive_(fullfile(wd, 'zern41em5z155em3.txt'), M);

            m = macos.Session(256);
            m.load_rx('FFSegDemoAll.in');           % engine reads GridFile= here
            ff = m.find_freeform_elts();
            testCase.assertNotEmpty(ff, 'fixture must have a FreeForm grid elt');
            Geng = m.zrn_freeform(ff(1)).grid.mat;  % engine's ingestion

            Mrgf = macos.read_grid_file('zern41em5z155em3.txt');
            testCase.verifyEqual(Mrgf, Geng, 'AbsTol', 1e-12, ...
                'read_grid_file must reproduce the engine GridFile= ingestion');

            % and confirm the naive read is exactly the transpose (the bug)
            Mn = readmatrix('zern41em5z155em3.txt', 'FileType', 'text');
            testCase.verifyEqual(Mn, Geng.', 'AbsTol', 1e-12, ...
                'a naive readmatrix is the transpose of the engine ingestion');
        end

        function test_not_square_errors(testCase)
            f = [tempname '.txt'];
            c = onCleanup(@() delete(f)); %#ok<NASGU>
            fid = fopen(f, 'w'); fprintf(fid, '1 2 3\n4 5 6\n'); fclose(fid);
            testCase.verifyError(@() macos.read_grid_file(f), ...
                'macos:read_grid_file:notSquare');
        end

        function test_missing_file_errors(testCase)
            testCase.verifyError( ...
                @() macos.read_grid_file('/no/such/read_grid_file_xyz.txt'), ...
                'macos:read_grid_file:open');
        end
    end
end

% ---------------------------------------------------------------------------
function write_naive_(path, M)
%WRITE_NAIVE_  Write M so text line r == matrix row r (a hand-made file).
N = size(M, 2);
fid = fopen(path, 'w');
assert(fid > 0, 'cannot open %s', path);
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fmt = [strjoin(repmat({'%.15e'}, 1, N), ' '), '\n'];
fprintf(fid, fmt, M.');     % M.' walks col-major == M row-major -> line=row
end

function cleanup_(cwd0, wd)
cd(cwd0);
if exist(wd, 'dir'), rmdir(wd, 's'); end
end
