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

        function test_elt_grid_add_matches_gridfile_load(testCase)
        % Luis (2026-08-19) reported that a figure fed through
        % macos.elt_grid_add came out 90-degrees-rotated relative to the
        % same data loaded with `GridFile=`.  ONE physical map, TWO input
        % paths, compared on the OPD RESPONSE (not on the stored array):
        %   (i)  the deck's own GridFile= ingestion
        %   (ii) elt_grid_add(read_grid_file(THE SAME FILE)) on a deck
        %        whose GridFile= holds zeros
        % These must be identical.  The map is an off-axis bump plus an
        % L-shaped ridge -- deliberately NOT transpose-symmetric, because
        % a symmetric map cannot see the defect being tested.  Leg (iii)
        % feeds the TRANSPOSE and asserts it does NOT match, so a pass
        % cannot come from a figure that does nothing.
            N = 256;                                 % FFSegDemoAll nGridMat
            [I, J] = ndgrid(1:N, 1:N);               % I = +x, J = +y
            M = 2.0e-5 * exp(-(((I-90).^2 + (J-170).^2) / (2*22^2)));
            M(40:44,  30:110) = M(40:44,  30:110) + 1.0e-5;  % arm along +y
            M(40:110, 30:34 ) = M(40:110, 30:34 ) + 1.0e-5;  % arm along +x
            testCase.assertNotEqual(M, M.', 'map must be asymmetric');

            wd = tempname; mkdir(wd); cwd0 = cd(wd);
            cRestore = onCleanup(@() cleanup_(cwd0, wd)); %#ok<NASGU>
            copyfile(rx_fixture_path('FFSegDemoAll.in'), fullfile(wd,'seg.in'));
            GN = 'zern41em5z155em3.txt';              % the name the deck uses
            macos.write_grid_file(fullfile(wd,'Mref.txt'), M);
            macos.write_grid_file(fullfile(wd,'Zero.txt'), zeros(N));

            m = macos.Session(256);

            % (i) the deck loads the map itself
            copyfile(fullfile(wd,'Mref.txt'), fullfile(wd,GN));
            W_file = trace_opd_(m, 'seg.in');
            ff = m.find_freeform_elts();
            testCase.assertNotEmpty(ff, 'fixture must have FreeForm grid elts');

            % flat control -- the figure must actually do something
            copyfile(fullfile(wd,'Zero.txt'), fullfile(wd,GN));
            W_flat = trace_opd_(m, 'seg.in');
            effect = max(abs(W_file(:) - W_flat(:)));
            testCase.assertGreaterThan(effect, 1e-9, ...
                'fixture grid has no effect -- this test would be vacuous');

            % (ii) the SAME file, via read_grid_file + elt_grid_add
            Mr = macos.read_grid_file(fullfile(wd,'Mref.txt'));
            W_add  = add_and_trace_(m, 'seg.in', ff, Mr);
            % (iii) non-vacuity: the transpose must NOT match
            W_addT = add_and_trace_(m, 'seg.in', ff, Mr.');

            testCase.verifyEqual(W_add, W_file, 'AbsTol', 0, ...
                'elt_grid_add(read_grid_file(f)) must equal GridFile= f');
            testCase.verifyGreaterThan( ...
                max(abs(W_addT(:) - W_file(:))), 0.1 * effect, ...
                'a transposed feed must be detectable (test is non-vacuous)');
        end

        function test_write_grid_file_round_trips_through_gridfile(testCase)
        % The emitter half: a file written by macos.write_grid_file and
        % then ingested by the engine as `GridFile=` must come back as the
        % array that was written.
            N = 256;
            [I, J] = ndgrid(1:N, 1:N);
            M = 1e-6 * (I + 3*J) + 2e-5 * (I > 200 & J < 60);   % asymmetric
            testCase.assertNotEqual(M, M.', 'map must be asymmetric');

            wd = tempname; mkdir(wd); cwd0 = cd(wd);
            cRestore = onCleanup(@() cleanup_(cwd0, wd)); %#ok<NASGU>
            copyfile(rx_fixture_path('FFSegDemoAll.in'), fullfile(wd,'seg.in'));
            macos.write_grid_file(fullfile(wd,'zern41em5z155em3.txt'), M);

            m = macos.Session(256);
            m.load_rx('seg.in');
            ff = m.find_freeform_elts();
            G = macos.get_elt_grid(ff(1)).mat;
            testCase.verifyEqual(G, M, 'AbsTol', 1e-12, ...
                'write_grid_file -> GridFile= must round-trip the array');
        end

        function test_gridfile_before_ngridmat_still_loads(testCase)
        % GridFile= parsed BEFORE nGridMat= must load the same grid.
        %
        % GridInit sizes its read with DO j=1,nGridMat, so when GridFile=
        % came first the engine read ZERO rows, printed "dimension 0 by 0",
        % returned success, and the trace sampled an all-zero grid -- the
        % surface was SILENTLY FLAT.  Luis hit this on a GridData deck.
        % The parser now defers the read until nGridMat= is known.
        %
        % Asserts on the OPD, not on "load succeeded": the swapped deck must
        % reproduce the committed-order OPD, AND both must differ from the
        % same deck with an all-zero grid -- otherwise the comparison would
        % pass on a fixture whose figure does nothing.
            wd = tempname; mkdir(wd);
            cwd0 = cd(wd);
            cRestore = onCleanup(@() cleanup_(cwd0, wd)); %#ok<NASGU>
            copyfile(rx_fixture_path('FFSegDemoAll.in'), fullfile(wd, 'good.in'));

            N = 256;                                 % FFSegDemoAll nGridMat
            M = 1e-5 * ((1:N).' * 1000 + (1:N));     % asymmetric, non-trivial
            write_naive_(fullfile(wd, 'zern41em5z155em3.txt'), M);
            swap_keys_(fullfile(wd, 'good.in'), fullfile(wd, 'swapped.in'));

            m = macos.Session(256);
            Wgood = trace_opd_(m, 'good.in');
            Wswap = trace_opd_(m, 'swapped.in');

            % the figure must do something, or the comparison is vacuous
            write_naive_(fullfile(wd, 'zern41em5z155em3.txt'), zeros(N));
            Wflat = trace_opd_(m, 'good.in');
            testCase.assertGreaterThan(max(abs(Wgood(:) - Wflat(:))), 1e-9, ...
                'fixture grid has no effect -- this test would be vacuous');

            testCase.verifyEqual(Wswap, Wgood, 'AbsTol', 1e-12, ...
                'GridFile= before nGridMat= must load the same grid');
        end

        function test_gridfile_without_ngridmat_fails_loudly(testCase)
        % GridFile= with no nGridMat= anywhere in the element block has no
        % size to read with.  It must FAIL the load with a targeted message,
        % never silently install a 0x0 (flat) grid.
            wd = tempname; mkdir(wd);
            cwd0 = cd(wd);
            cRestore = onCleanup(@() cleanup_(cwd0, wd)); %#ok<NASGU>
            copyfile(rx_fixture_path('FFSegDemoAll.in'), fullfile(wd, 'good.in'));
            N = 256;
            write_naive_(fullfile(wd, 'zern41em5z155em3.txt'), ...
                         1e-5 * ((1:N).' * 1000 + (1:N)));
            drop_ngridmat_(fullfile(wd, 'good.in'), fullfile(wd, 'nong.in'));

            m = macos.Session(256);
            loaded = true;
            try
                nE = m.load_rx('nong.in');
                loaded = nE > 0;
            catch
                loaded = false;
            end
            % The load must FAIL, not warn-and-continue: that is the whole
            % point -- the old engine "succeeded" with a 0x0 grid and left a
            % traceable, silently flat deck.
            testCase.verifyFalse(loaded, ...
                'a GridFile= with no nGridMat= must not load');
            % ... and it must leave NO usable Rx behind to trace.
            testCase.verifyFalse(macos.has_rx(), ...
                'a failed load must not leave a usable (silently flat) Rx');
            % The engine's targeted message ("GridFile= given but nGridMat=
            % never set, iElt=") is written to process stdout by Fortran, so
            % evalc cannot capture it here; its wording is covered by the CLI
            % pty spot-check recorded in the commit message.
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

function swap_keys_(src, dst)
%SWAP_KEYS_  Copy SRC to DST with every adjacent nGridMat=/GridFile= pair
%   swapped, i.e. GridFile= comes FIRST.  Generated here rather than
%   committed so the variant cannot drift from its parent fixture.
t = strsplit(fileread(src), newline);
for k = 1:numel(t)-1
    if is_key_(t{k}, 'nGridMat') && is_key_(t{k+1}, 'GridFile')
        tmp = t{k};  t{k} = t{k+1};  t{k+1} = tmp;
    end
end
assert(any(cellfun(@(L) is_key_(L,'GridFile'), t)), 'no GridFile= in %s', src);
write_lines_(dst, t);
end

function drop_ngridmat_(src, dst)
%DROP_NGRIDMAT_  Copy SRC to DST with every nGridMat= line removed.
t = strsplit(fileread(src), newline);
t = t(~cellfun(@(L) is_key_(L, 'nGridMat'), t));
write_lines_(dst, t);
end

function tf = is_key_(line, key)
tf = ~isempty(regexp(strtrim(line), ['^' key '\s*='], 'once'));
end

function write_lines_(path, t)
if ~isempty(t) && isempty(strtrim(t{end})), t(end) = []; end
fid = fopen(path, 'w');
assert(fid > 0, 'cannot open %s', path);
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', t{:});
end

function W = add_and_trace_(m, rx, elts, G)
%ADD_AND_TRACE_  Load RX (whose GridFile= holds zeros), add G to every
%   grid element through the elt_grid_add path, and return the OPD.
m.load_rx(rx);
for e = elts(:).'
    macos.elt_grid_add(e, G);
end
n = m.num_elt();
m.trace(n - 1);
W = macos.opd();
end

function W = trace_opd_(m, rx)
%TRACE_OPD_  Load RX and return the OPD at the exit-pupil element.
m.load_rx(rx);
n = m.num_elt();
m.trace(n - 1);
W = macos.opd();
end
