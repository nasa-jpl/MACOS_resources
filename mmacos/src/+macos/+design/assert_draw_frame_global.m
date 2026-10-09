function assert_draw_frame_global(plane, caller)
%ASSERT_DRAW_FRAME_GLOBAL  Refuse to read draw_rays' U/V as global x,y,z
%   unless the deck's source frame actually is the global frame.
%
%   macos.draw_rays does NOT return global coordinates.  It projects
%   each crossing onto the SOURCE / RAY-GRID basis -- DRAW sets
%   xDraw/yDraw from xGrid/yGrid/zGrid (macos_cmd_loop.inc), so
%   b.U = RayPos.xDraw and b.V = RayPos.yDraw:
%
%       plane 'XY'  ->  U along xGrid,  V along yGrid
%       plane 'XZ'  ->  U along zGrid,  V along xGrid
%       plane 'YZ'  ->  U along zGrid,  V along yGrid
%
%   The design layer pins its own emission to xGrid=(+1,0,0),
%   yGrid=(0,1,0) (see the NOTE in emit_) precisely so that its
%   consumers may treat U/V as global.  A HERITAGE / SegMirMaker deck
%   breaks that -- the whole e5mono and e2e corpus carries
%   xGrid=(-1,0,0) -- and so does macos.design.segment_rx output,
%   which deliberately flips the merged deck's xGrid to -1 to meet the
%   SegMirMaker <-> engine tiling contract (segment_rx.m ~212).
%
%   Under xGrid=(-1,0,0), reading b.U as a global X MIRRORS the pupil
%   about the Y axis.  On any pair of features straddling the X axis a
%   mirror is indistinguishable from a 180-deg rotation -- that is
%   exactly the trap that inverted round 1 of the segment-routing
%   audit (segmirmaker/SEG_AUDIT_STATUS.md, T2), and once before that
%   it sent center_focal_plane a metre off.
%
%   The test is on ORIENTATION (dot > cos 8.1 deg), not equality: the engine
%   re-derives the grid frame from ChfRayDir, so an off-axis field point
%   tilts it by the field angle.  That is normal.  A sign flip or an axis
%   permutation is not, and neither survives the threshold.
%
%   This ERRORS rather than transforming.  The projection is lossy:
%   U = RayPos.xGrid folds in the z component, so there is no general
%   inverse back to global x,y.  For a caller that genuinely needs
%   global positions the fix is macos.draw_rays3d (true global x,y,z)
%   or macos.get_ray_info, not a sign flip here.
%
%   An axis whose source vector is still all-zero is SKIPPED: zGrid is
%   derived from ChfRayDir and is unpopulated until a trace establishes the
%   ray grid.  Once populated it IS checked -- a deck propagating along -z
%   makes draw's 'XZ'/'YZ' U axis minus global Z, which the callers' cU=3
%   mapping would read with the wrong sign, so tripping there is a real
%   finding rather than a false alarm.
    s = macos.get_src_csys();
    switch upper(plane)
        case 'XY', uv = {s.xDir, 1, s.yDir, 2};
        case 'XZ', uv = {s.zDir, 3, s.xDir, 1};
        case 'YZ', uv = {s.zDir, 3, s.yDir, 2};
        otherwise
            error('macos:design:drawFrame:plane', ...
                  'plane must be YZ, XZ or XY (got %s).', plane);
    end
    axn = 'XYZ';
    uvn = 'UV';
    for q = [1 3]
        v = uv{q}(:);  c = uv{q+1};
        nm = uvn((q+1)/2);
        want = zeros(3,1);  want(c) = 1;
        % zGrid is DERIVED (from ChfRayDir) and is all-zero until a trace
        % establishes the ray grid, so an axis that is not populated yet is
        % skipped rather than failed -- DRAW establishes it, and the
        % handedness this guard exists for lives in the DECLARED xGrid/yGrid.
        if norm(v) < 1e-9, continue; end
        % Test ORIENTATION, not equality.  The engine re-derives the grid
        % frame from ChfRayDir, so an off-axis field point tilts it by the
        % field angle (a 2-arcmin sweep gives yGrid=(0,1,-5.8e-4)) -- normal
        % and harmless here.  What must never pass is a gross
        % mis-orientation: a sign flip (dot = -1, the heritage xGrid=-1 case)
        % or an axis permutation (dot = 0).  cos 8.1 deg splits those cleanly
        % from any realistic field tilt.
        if dot(v/norm(v), want) < 0.99
            error('macos:design:drawFrame:notGlobal', ...
                ['%s reads macos.draw_rays'' %c axis as global %c, but this ' ...
                 'deck''s source frame is not the global frame:\n' ...
                 '  draw ''%s'' %c-axis = [%+.6g %+.6g %+.6g], expected [%+.6g %+.6g %+.6g] (%.2f deg off)\n' ...
                 '  (xGrid=[%+.6g %+.6g %+.6g]  yGrid=[%+.6g %+.6g %+.6g]  zGrid=[%+.6g %+.6g %+.6g])\n' ...
                 'draw_rays projects onto the SOURCE/RAY-GRID basis, not the global one, so ' ...
                 'U/V would be silently mirrored here.  Heritage/SegMirMaker decks and ' ...
                 'macos.design.segment_rx output carry xGrid=(-1,0,0).  Use ' ...
                 'macos.draw_rays3d (global x,y,z) instead.'], ...
                caller, nm, axn(c), upper(plane), nm, ...
                v(1), v(2), v(3), want(1), want(2), want(3), ...
                acosd(max(-1,min(1,dot(v/norm(v),want)))), ...
                s.xDir(1), s.xDir(2), s.xDir(3), ...
                s.yDir(1), s.yDir(2), s.yDir(3), ...
                s.zDir(1), s.zDir(2), s.zDir(3));
        end
    end
end
