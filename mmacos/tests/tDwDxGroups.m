classdef tDwDxGroups < matlab.unittest.TestCase
%TDWDXGROUPS  Regression tests for group channels in dw_dx.

    properties (Constant)
        ModelSize  = 128
        RxName     = 'e5hex1.in'
        TestGroup  = [9; 10; 12; 13]
        % Two SEGMENTS of the primary.  The multi-field and
        % superposition cases below use this pair, not TestGroup, for
        % two reasons.  (1) GroupedRigidBodyChannel's own header says a
        % group spanning the exit-pupil Return and the focal plane does
        % NOT superimpose linearly -- that non-superposition is why the
        % group channel exists at all -- so TestGroup is the wrong group
        % to gate linearity on.  (2) Superposition is only MEANINGFUL
        % when the frames agree: the per-element channels perturb in
        % each element's OWN local (TElt) frame, so the members must
        % share an orientation with the group's reference element.
        % Seg1 and Seg2 carry the SAME psiElt in this deck (they are
        % segments of one parent surface), so their local triads are
        % parallel and a group motion in ref-elt-local coords IS the
        % same motion each member gets.  Elts 9/10 -- the two lens
        % surfaces -- have OPPOSITE normals and would not superimpose in
        % any single frame.
        SegPair    = [1; 2]
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (Access = private)
        function g = seg_map(testCase)
            g = containers.Map('KeyType','char','ValueType','any');
            g('SegPair') = testCase.SegPair;
        end

        function b = multi_base(~)
            % A deliberately tiny multi-field harvest: two optics, a
            % coarse ray grid.  These cases are about COLUMN BOOKKEEPING,
            % not optics.
            b = {'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                 'ngridpts', 15, 'elts', [1; 2], 'dofs', (0:5).', ...
                 'delta', 1e-8};
        end

        function pred = member_sum(testCase, oe, T, j, kind)
            % The frame-resolved sum of the members' columns for a GLOBAL
            % axis j: a global unit motion along e_j moves member k by
            % T_k(j,i) along its own local axis i (get_elt_csys returns
            % the triad with the local axes as COLUMNS in global coords).
            % kind 'trans' (default) uses the members' Tx/Ty/Tz columns,
            % 'rot' their Rx/Ry/Rz.
            if nargin < 5, kind = 'trans'; end
            off = 3;  if strcmp(kind, 'rot'), off = 0; end
            pred = zeros(size(oe.dwdx, 1), 1);
            for k = 1:numel(testCase.SegPair)
                base = (k - 1) * 6;          % member k's 6-DOF block
                for i = 1:3
                    pred = pred + T{k}(j, i) * oe.dwdx(:, base + off + i);
                end
            end
        end

        function T = member_triads(testCase, session)
            % TElt triads of the group members, local axes as COLUMNS in
            % global coordinates (macos.get_elt_csys returns a 6x6 whose
            % upper-left 3x3 is that triad -- column 3 is psiElt).
            cs = session.get_elt_csys(testCase.SegPair);
            T = cell(numel(testCase.SegPair), 1);
            for k = 1:numel(T)
                T{k} = cs.csys(1:3, 1:3, k);
            end
        end

        function p = rx_with_eltgrp(testCase, wd)
            % A copy of the fixture carrying "EltGrp= 2 1 2" in BOTH
            % member blocks -- macos's own convention, which is what
            % parse_rx_groups dedups.  Written to a temp dir; the deck's
            % GridFile= flat.txt is unresolvable from either cwd, so the
            % copy loads exactly as the original does.
            L = splitlines(string(fileread(testCase.rx_path)));
            out = strings(0, 1);
            for k = 1:numel(L)
                out(end+1, 1) = L(k); %#ok<AGROW>
                t = strtrim(L(k));
                if startsWith(t, "iElt=")
                    v = sscanf(char(extractAfter(t, "=")), '%d', 1);
                    if ~isempty(v) && any(v == testCase.SegPair)
                        out(end+1, 1) = "         EltGrp=  2 1 2"; %#ok<AGROW>
                    end
                end
            end
            p = fullfile(wd, 'e5hex1_grp.in');
            fid = fopen(p, 'w');  fprintf(fid, '%s\n', out);  fclose(fid);
        end
    end

    methods (Test)
        function test_parse_rx_groups_empty(testCase)
            % e5hex1.in has no EltGrp= declarations.
            g = macos.channels.parse_rx_groups(testCase.rx_path);
            testCase.verifyEqual(g.Count, uint64(0), ...
                'parse_rx_groups should return empty Map for e5hex1');
        end

        function test_group_channel_builder(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            groups = containers.Map('KeyType','char','ValueType','any');
            groups('Cam') = testCase.TestGroup;
            chs = macos.channels.grouped_rigid_body_channels(m, ...
                groups, 'rx_path', testCase.rx_path);
            testCase.verifyEqual(numel(chs), 6, ...
                'Should build 6 channels (one per DOF)');
            for k = 1:numel(chs)
                testCase.verifyEqual(chs{k}.kind(), 'Group');
                testCase.verifyEqual(chs{k}.ref_elt, ...
                    testCase.TestGroup(1));
                testCase.verifyEqual(chs{k}.fp_elt, 13, ...
                    'Cam fp_elt should auto-detect to Elt 13 (FocalPlane)');
                testCase.verifyEqual(chs{k}.fp_mode, 'sxp', ...
                    'auto fp_mode -> sxp when FP in group');
            end
        end

        function test_eltgrp_install_restore(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            ref = testCase.TestGroup(1);
            % Initial: no group installed.
            testCase.verifyEmpty(m.get_elt_grp(ref));
            % After apply -> install
            ch = macos.channels.GroupedRigidBodyChannel(m, ...
                testCase.TestGroup, 3, ...
                'fp_elt', 13, 'fp_mode', 'sxp');
            ch.apply(1e-8);
            members = m.get_elt_grp(ref);
            testCase.verifyEqual(sort(members), ...
                sort(testCase.TestGroup), ...
                'apply should install the desired EltGrp');
            % After restore -> uninstall back to empty
            ch.restore();
            testCase.verifyEmpty(m.get_elt_grp(ref), ...
                'restore should release the EltGrp install');
        end

        function test_dw_dx_groups_runs_clean(testCase)
            m = macos.Session(testCase.ModelSize);
            groups = containers.Map('KeyType','char','ValueType','any');
            groups('Cam') = testCase.TestGroup;
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3; 4; 5], ...
                'groups', groups, ...
                'delta', 1e-8);
            % 11 actual optics * 3 DOFs = 33 per-element + 3 group DOFs = 36
            testCase.verifyEqual(numel(out.channel_names), 33 + 3);
            % Last three are the group channels.
            for k = 34:36
                testCase.verifyEqual(out.kind{k}, 'Group');
            end
            for k = 1:33
                testCase.verifyEqual(out.kind{k}, 'RigidBody');
            end
            % Group columns should be non-trivial.
            grp_max = max(max(abs(out.dwdx(:, 34:36))));
            testCase.verifyGreaterThan(grp_max, 0, ...
                'group columns should be non-zero');
        end
        % =============================================================
        % MULTI-FIELD supervisor -- macos.dw_dx_multi 'groups'
        % =============================================================

        function test_multi_center_block_matches_single_field(testCase)
            % GATE 1.  The supervisor builds no channels itself: per
            % field it calls dw_dx.  So the group columns of the CENTER
            % block must reproduce dw_dx's group columns for the same
            % state.  reset_xp is off so the two runs reference the same
            % (prescription) exit pupil -- with it on the supervisor
            % re-writes elt nElt-1 and the comparison is not like-for-like.
            g = testCase.seg_map();
            b = testCase.multi_base();
            mm = macos.Session(testCase.ModelSize);
            om = macos.dw_dx_multi(mm, testCase.rx_path, b{:}, ...
                'grid', '1x1', 'reset_xp', false, 'groups', g);
            ms = macos.Session(testCase.ModelSize);
            os = macos.dw_dx(ms, testCase.rx_path, ...
                'ngridpts', 15, 'elts', [1; 2], 'dofs', (0:5).', ...
                'delta', 1e-8, 'groups', g);
            testCase.verifyEqual(om.channel_names, os.channel_names, ...
                'multi must expose dw_dx''s channel list verbatim');
            % 2 optics x 6 DOFs = 12 per-element, then 6 group columns
            testCase.verifyEqual(numel(om.channel_names), 18);
            gc = 13:18;
            A = om.per_field_dwdx{1}(:, gc);
            B = os.dwdx(:, gc);
            scale = max(abs(B), [], 'all');
            testCase.verifyGreaterThan(scale, 0, ...
                'non-vacuity: the single-field group columns are zero');
            testCase.verifyLessThan( ...
                max(abs(A - B), [], 'all') / scale, 1e-10, ...
                'center-field group columns must match dw_dx''s');
            % and the stacked Jacobian carries the same block
            testCase.verifyEqual(size(om.dwdxall, 2), 18);
        end

        function test_multi_channel_identity_across_fields(testCase)
            % GATE 2.  The stack assumes every per-field block has the
            % SAME channels in the SAME order -- group channels included.
            % The supervisor asserts channel_names equality internally
            % (it ERRORS on a mismatch), so a run that returns at all has
            % already passed that; here we pin the shape, the tail
            % ordering (groups append AFTER the per-element block) and
            % the bookkeeping arrays, then check the blocks are genuinely
            % DIFFERENT harvests and not copies of one another.
            g = testCase.seg_map();
            b = testCase.multi_base();
            m = macos.Session(testCase.ModelSize);
            om = macos.dw_dx_multi(m, testCase.rx_path, b{:}, 'groups', g);
            nf = numel(om.field_names);
            testCase.verifyEqual(nf, 5, 'default field set is C + 4 corners');
            testCase.verifyEqual(numel(om.channel_names), 18);
            for k = 1:nf
                testCase.verifyEqual(size(om.per_field_dwdx{k}, 2), 18, ...
                    sprintf('field %s has a different channel count', ...
                        om.field_names{k}));
            end
            % group channels come LAST, are labelled, and carry no
            % element id (iElt 0, kind 'Group') -- section on kind
            for k = 1:12
                testCase.verifyEqual(om.kind{k}, 'RigidBody');
            end
            for k = 13:18
                testCase.verifyEqual(om.kind{k}, 'Group');
                testCase.verifyEqual(om.iElt(k), 0);
                testCase.verifyTrue( ...
                    startsWith(om.channel_names{k}, 'Grp[SegPair]'));
            end
            % dof_idx survives for the group block (the rot_output
            % scaling keys off it -- gated below)
            testCase.verifyEqual(om.dof_idx(13:18), (0:5).');
            % NON-VACUITY: the per-field blocks must be distinct
            % harvests, not one block replicated
            ctr = find(strcmp(om.field_names, 'C'), 1);
            oth = find((1:nf) ~= ctr, 1);
            d = max(abs(om.per_field_dwdx{ctr}(:, 13:18) - ...
                        om.per_field_dwdx{oth}(:, 13:18)), [], 'all');
            testCase.verifyGreaterThan(d, 0, ...
                'group columns identical at two fields -- blocks copied?');
        end

        function test_group_translation_is_the_member_sum(testCase)
            % GATE 3, first half.  A rigid TRANSLATION of the group is
            % exactly the two members translating together, so to first
            % order the group column is the sum of the member columns.
            %
            % ONE 'delta' now means one PHYSICAL poke on both sides:
            % GroupedRigidBodyChannel converts SI metres -> BaseUnits for
            % prb_grp, exactly as macos.perturb does for the per-element
            % channel, so there is no CBM factor to carry here.  (It used
            % to pass the increment straight through, which made this
            % comparison a two-delta, CBM-juggling affair -- and drove
            % the artifact gated by
            % test_scalar_delta_matches_the_split_step below.)
            %
            % FRAMES -- the part that makes this a real gate rather than
            % a coincidence.  The per-element channels perturb in each
            % element's OWN local (TElt) frame, while the group channel
            % here perturbs in GLOBAL coords (the default).  On this deck
            % Seg1 and Seg2's triads differ by about 3 deg, and a
            % segment's PISTON response is tens of times its lateral one,
            % so ignoring the frames leaks Tz into Tx at O(1) -- a naive
            % column-vs-column comparison misses by 155%, which reads as
            % "groups are broken" and is really "wrong frame".  So the
            % member sum is assembled properly: a global unit
            % displacement e_j moves member k by (T_k(j,i)) along its own
            % local axis i, T_k being the element's TElt triad
            % (get_elt_csys returns it with the local axes as COLUMNS in
            % global coords).
            g = testCase.seg_map();
            me = macos.Session(testCase.ModelSize);
            oe = macos.dw_dx(me, testCase.rx_path, 'ngridpts', 15, ...
                'elts', testCase.SegPair, 'dofs', (0:5).', 'delta', 1e-8);
            mg = macos.Session(testCase.ModelSize);
            og = macos.dw_dx(mg, testCase.rx_path, 'ngridpts', 15, ...
                'elts', 1, 'dofs', (0:5).', 'delta', 1e-8, 'groups', g, ...
                'group_coords', 'global');
            T = testCase.member_triads(mg);
            for j = 1:3      % global Tx Ty Tz
                pred = testCase.member_sum(oe, T, j);
                grp  = og.dwdx(:, 6 + 3 + j);
                sc = max(abs(pred));
                testCase.verifyGreaterThan(sc, 0, ...
                    'non-vacuity: the member translation columns are zero');
                testCase.verifyLessThan( ...
                    max(abs(grp - pred)) / sc, 1e-2, ...
                    sprintf(['group translation along global axis %d must ' ...
                             'equal the frame-resolved member sum'], j));
            end
        end

        function test_scalar_delta_matches_the_split_step(testCase)
            % THE regression gate for the unit asymmetry.
            %
            % prb_grp's signature is BaseUnits for translations;
            % macos.perturb's is SI metres.  While the group channel
            % passed its increment STRAIGHT THROUGH, one scalar 'delta'
            % meant two different physical pokes -- 1/CBM apart, 10 pm
            % against 10 nm on this millimetre deck -- and the group
            % columns fell toward the finite-difference floor.  The tell
            % was that the error GREW as the step shrank: group column
            % over frame-resolved member sum ran 1.0000 (delta 1e-5) ->
            % 1.0005 (1e-6) -> 1.012 (1e-7) -> 1.657 (1e-8).  It reads as
            % physics -- an "intra-group compensation factor" -- which is
            % how it nearly shipped in a template exhibit.
            %
            % Two assertions, in order of sharpness.  (1) At the SMALLEST
            % step, the group translation columns still reproduce the
            % member sum: pre-fix Tx misses by 66% and Tz by 200%, so
            % this is the assertion that actually fails on the old
            % channel.  (2) The scalar and the split-step (1,6) forms --
            % the two things the templates might be run with -- agree, so
            % nobody can read a units artifact as a physical ratio.
            g = testCase.seg_map();
            me = macos.Session(testCase.ModelSize);
            oe = macos.dw_dx(me, testCase.rx_path, 'ngridpts', 15, ...
                'elts', testCase.SegPair, 'dofs', (0:5).', 'delta', 1e-8);
            T = testCase.member_triads(me);

            ms = macos.Session(testCase.ModelSize);
            sc_ = macos.dw_dx(ms, testCase.rx_path, 'ngridpts', 15, ...
                'elts', 1, 'dofs', (0:5).', 'delta', 1e-8, 'groups', g);
            mv = macos.Session(testCase.ModelSize);
            sp = macos.dw_dx(mv, testCase.rx_path, 'ngridpts', 15, ...
                'elts', 1, 'dofs', (0:5).', 'groups', g, ...
                'delta', [1e-8 1e-8 1e-8 1e-6 1e-6 1e-6]);

            for j = 1:3
                pred = testCase.member_sum(oe, T, j);
                col  = 6 + 3 + j;
                r = rms_(sc_.dwdx(:, col)) / rms_(pred);
                testCase.verifyLessThan(abs(r - 1), 1e-2, sprintf( ...
                    ['scalar-delta group translation %d is %.4f of the ' ...
                     'member sum -- the BaseUnit/metre asymmetry is back'], ...
                    j, r));
                % the two step choices must not disagree about a column
                % whose units are now the same on both sides.  ONLY the
                % translations: the split form leaves the rotation step
                % alone, so a rotation column differs only by ordinary
                % FD noise (Rz on this cell is near-inert and runs 2e-2).
                d = max(abs(sc_.dwdx(:, col) - sp.dwdx(:, col))) ...
                    / max(abs(sp.dwdx(:, col)));
                testCase.verifyLessThan(d, 1e-2, sprintf( ...
                    'scalar vs split-step group translation %d differs', j));
            end
        end

        function test_group_rotation_is_not_the_member_sum(testCase)
            % GATE 3, second half -- THE non-vacuity check for the whole
            % group machinery.  A group ROTATION pivots BOTH members
            % about the GROUP frame (the reference element), which for an
            % off-pivot member is a rotation PLUS a lever-arm
            % translation.  Summing each member's own about-its-own-point
            % rotation cannot reproduce that -- and this is asserted with
            % the SAME frame resolution the translation gate uses, so the
            % difference cannot be dismissed as a frame artifact.  If
            % these matched, the "group" channel would be a bookkeeping
            % sum and GPERTURB would be doing nothing a caller could not
            % do in MATLAB.
            g = testCase.seg_map();
            me = macos.Session(testCase.ModelSize);
            oe = macos.dw_dx(me, testCase.rx_path, 'ngridpts', 15, ...
                'elts', testCase.SegPair, 'dofs', (0:5).', 'delta', 1e-8);
            mg = macos.Session(testCase.ModelSize);
            og = macos.dw_dx(mg, testCase.rx_path, 'ngridpts', 15, ...
                'elts', 1, 'dofs', (0:5).', 'delta', 1e-8, 'groups', g, ...
                'group_coords', 'global');
            T = testCase.member_triads(mg);
            % rotations are rad on both sides and always were -- the
            % SI-metres conversion the channel now does applies to
            % translations only
            worst = 0;
            for j = 1:3      % global Rx Ry Rz
                pred = testCase.member_sum(oe, T, j, 'rot');
                grp = og.dwdx(:, 6 + j);
                sc = max(abs(pred));
                if sc == 0, continue; end
                worst = max(worst, max(abs(grp - pred)) / sc);
            end
            testCase.verifyGreaterThan(worst, 1e-2, ...
                ['group rotations must NOT be the member sum -- they ' ...
                 'pivot about the group frame, not each element''s own']);
        end

        function test_jacobian_emits_base_units_rot_output_noop(testCase)
            % Convention (Dave 2026-08-25): the Jacobian's OPD numerator
            % emits in the deck's BaseUnits -- the same units as opd()
            % and the dwdz/dwdsurf/dwdgrid rungs -- so wall = dwdx*x + w0
            % is unit-consistent on any deck.  'rot_output' is a retained
            % NO-OP (it existed to un-CBM the rotations of the old
            % OPD-metres emitter).  Gated two ways on this mm fixture:
            % the two settings are bit-identical, and a HAND finite
            % difference in raw opd() units matches the emitted column
            % (the old emitter would differ by 1/CBM = 1000x here).
            g = testCase.seg_map();
            m1 = macos.Session(testCase.ModelSize);
            nat = macos.dw_dx(m1, testCase.rx_path, 'ngridpts', 15, ...
                'elts', 1, 'dofs', (0:5).', 'delta', 1e-8, 'groups', g);
            m2 = macos.Session(testCase.ModelSize);
            bpr = macos.dw_dx(m2, testCase.rx_path, 'ngridpts', 15, ...
                'elts', 1, 'dofs', (0:5).', 'delta', 1e-8, 'groups', g, ...
                'rot_output', 'base-per-rad');
            testCase.verifyLessThan(abs(nat.cbm - 1), 1, ...
                'fixture must NOT be a metre-unit deck or this is vacuous');
            testCase.verifyEqual(nat.dwdx, bpr.dwdx, ...
                'rot_output must be a no-op -- BaseUnits either way');
            % hand FD of elt 1 Tz (dof 5 -> column 6), raw opd() units
            m3 = macos.Session(testCase.ModelSize);
            m3.load_rx(testCase.rx_path);
            m3.set_src_sampling(15);
            m3.modify();
            nE = m3.num_elt();
            d = 1e-8;                                     % SI metres
            macos.perturb(1, 'translation', [0; 0; +d]);
            m3.modify();  m3.trace(nE - 1);  Wp = m3.opd();
            macos.perturb(1, 'translation', [0; 0; -2*d]);
            m3.modify();  m3.trace(nE - 1);  Wm = m3.opd();
            macos.perturb(1, 'translation', [0; 0; +d]);  % restore
            m3.modify();
            v = (Wp ~= 0) & (Wm ~= 0);
            hand = max(abs(Wp(v) - Wm(v))) / (2 * d);
            col  = max(abs(nat.dwdx(:, 6)));
            testCase.verifyEqual(col, hand, 'RelTol', 1e-6, ...
                ['emitted column scale must match a hand FD in raw ' ...
                 'opd() units -- a CBM-scaled emitter is 1000x off here']);
        end

        function test_groups_auto_and_the_explicit_map_merge(testCase)
            % The parse-once hoist in dw_dx_multi must reproduce dw_dx's
            % merge semantics: auto first, an explicit entry OVERRIDING an
            % auto one of the same name, the union otherwise.
            wd = tempname;  mkdir(wd);
            c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_with_eltgrp(wd);
            b = {'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '1x1', ...
                 'ngridpts', 15, 'elts', 1, 'dofs', (0:5).', 'delta', 1e-8};

            % (a) auto alone: EltGrp= 2 9 10 in both member blocks is ONE
            %     group, named by its member span
            m1 = macos.Session(testCase.ModelSize);
            a1 = macos.dw_dx_multi(m1, rx, b{:}, 'groups_auto', true);
            testCase.verifyEqual(numel(a1.channel_names), 12);
            for k = 7:12
                testCase.verifyEqual(a1.kind{k}, 'Group');
                testCase.verifyTrue(startsWith(a1.channel_names{k}, 'Grp[1-2]'));
            end

            % (b) union: an explicit group under a DIFFERENT name adds 6
            m2 = macos.Session(testCase.ModelSize);
            gx = containers.Map('KeyType','char','ValueType','any');
            gx('SegPair') = testCase.SegPair;
            a2 = macos.dw_dx_multi(m2, rx, b{:}, 'groups_auto', true, ...
                'groups', gx);
            testCase.verifyEqual(numel(a2.channel_names), 18);
            testCase.verifyEqual(nnz(strcmp(a2.kind, 'Group')), 12);

            % (c) override: the SAME name with different members keeps the
            %     count at 6 but changes the motion -- so the columns must
            %     differ from (a)
            m3 = macos.Session(testCase.ModelSize);
            go = containers.Map('KeyType','char','ValueType','any');
            go('1-2') = [1; 2; 3];
            a3 = macos.dw_dx_multi(m3, rx, b{:}, 'groups_auto', true, ...
                'groups', go);
            testCase.verifyEqual(numel(a3.channel_names), 12);
            testCase.verifyEqual(nnz(strcmp(a3.kind, 'Group')), 6);
            d = max(abs(a3.dwdxall(:, 7:12) - a1.dwdxall(:, 7:12)), [], 'all');
            testCase.verifyGreaterThan(d, 0, ...
                'an explicit entry must OVERRIDE the auto group of the same name');
        end

        function test_no_groups_is_the_preserved_surface(testCase)
            % The six new opts must be inert when unused: a run with no
            % groups is byte-identical to the same call written without
            % them.
            b = testCase.multi_base();
            m1 = macos.Session(testCase.ModelSize);
            a = macos.dw_dx_multi(m1, testCase.rx_path, b{:}, 'grid', '1x1');
            m2 = macos.Session(testCase.ModelSize);
            e = macos.dw_dx_multi(m2, testCase.rx_path, b{:}, 'grid', '1x1', ...
                'groups', [], 'groups_auto', false, ...
                'group_coords', 'global', 'group_fp_mode', 'auto', ...
                'group_stop_mode', 'obj', 'group_stop_pos', [0 0 0]);
            testCase.verifyTrue(isequal(a.dwdxall, e.dwdxall));
            testCase.verifyEqual(numel(a.channel_names), 12);
            testCase.verifyFalse(any(strcmp(a.kind, 'Group')));
        end
    end
end


function r = rms_(v)
r = sqrt(mean(v(isfinite(v)).^2));
end
