function out = clear_build(P, D, deck, opts)
%CLEAR_BUILD  One evaluable CLEARED afocal4 design: close it, emit it, swing it.
%
%   OUT = CLEAR_BUILD(P, D, DECK) is AFOCAL4_BUILD followed by CLEAR_TILT.
%   The design struct D carries one extra field, D.tilt_deg, the extraction
%   tilt applied to the field mirror; everything else is exactly the S4b
%   design struct and is closed, emitted and posed by AFOCAL4_BUILD
%   unchanged.  With D.tilt_deg = 0 the two functions are BIT-IDENTICAL --
%   asserted, not assumed, by AFOCAL4_CLEARING's null section.
%
%   WHY THE TILT IS APPLIED TO THE EMITTED DECK AND NOT INSIDE THE CLOSURE.
%   The closure is a PARAXIAL, coaxial statement -- spacings along one axis
%   -- and a tilt is exactly the thing it cannot express.  Applying it to
%   the committed artifact instead means (a) the first-order closure that
%   AFOCAL4_BUILD guarantees is untouched, (b) the tilt is an exact rigid
%   motion of the traced chief ray (CLEAR_TILT reads the chief and the local
%   surface normal from the engine), and (c) the deck this function writes
%   is the deck every scorer and every gate reads.  The cost is that the
%   first-order identities hold for the CHIEF exactly and for the real beam
%   only to the extent the tilted surface still collimates -- so the
%   traced M and collimation are re-verified here rather than inherited.
%
%   D.tilt_deg is the CLEARING STAGE'S OPERATING PARAMETER, in the same
%   sense P.iface is the S4 ruling's: it is not optimised to a value and
%   quoted, it is swept and reported (CLEAR_PRICE), because what it buys is
%   clearance and what it costs is pupil quality and the exchange rate is
%   the deliverable.
%
%   THE UNION WALL IS APPLIED HERE, AFTER THE SWING, AND THAT IS THE WHOLE
%   POINT OF DEFERRING IT.  AFOCAL4_BUILD carries the union body-in-beam
%   floor as a wall (P.pack.union_enforce, default off), but it emits the
%   UNTILTED train -- which on this family is the design that fails at
%   -79.9 mm.  A wall applied there would reject every iterate before the
%   tilt had a chance to clear the beam: a cage, not a wall.  So the build
%   is told to defer, the tilt is applied, and AFOCAL4_UNION_WALL then
%   judges the deck that will actually be scored.  With the wall off (the
%   default) neither call traces a ray and this file is unchanged.
%
%   Name-value:
%     'axis'   1x3 tilt axis, global (default [1 0 0] -- swings the beam in
%              the y-z plane the field bias lives in, which CLEAR_SCAN
%              measures to be the cheaper of the two axes per mm of
%              clearance won)
%     'elt'    which mirror to swing (default 'FM')
%     'verify' re-trace and report M / collimation (true)
%     'quiet'  (true)
%
%   Returns AFOCAL4_BUILD's struct plus .tilt (the CLEAR_TILT record),
%   .tilt_deg, .union (the wall's measurement) and, with 'verify', a
%   re-measured .traced.
%
%   See also AFOCAL4_BUILD, CLEAR_TILT, CLEAR_SOLVE, CLEAR_SEED,
%   AFOCAL4_UNION, AFOCAL4_UNION_WALL.

    arguments
        P (1,1) struct
        D (1,1) struct
        deck (1,:) char
        opts.axis   (1,3) double = [1 0 0]
        opts.elt    = 'FM'
        opts.verify (1,1) logical = true
        opts.quiet  (1,1) logical = true
    end

    if ~isfield(D,'tilt_deg'), D.tilt_deg = 0; end

    b = afocal4_build(P, D, deck, 'verify',false, 'quiet',true, ...
                      'defer_union',true);
    out = b;
    out.tilt_deg = D.tilt_deg;
    out.tilt = [];

    if D.tilt_deg ~= 0
        tmp = [tempname '.in'];
        cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
        copyfile(deck, tmp);
        out.tilt = clear_tilt(tmp, struct('elt',opts.elt, ...
                        'alpha',deg2rad(D.tilt_deg), 'axis',opts.axis), deck);
    end

    % the wall, on the deck that will actually be scored
    out.union = afocal4_union_wall(P, deck, 'quiet',opts.quiet);

    if opts.verify
        macos.load_rx(deck);
        if ~macos.has_rx()
            error('macos:design:clear_build:load', 'deck failed to load: %s', deck);
        end
        out.traced = traced_(macos.num_elt(), P.D);
        out.paraxial_ok = abs(out.traced.mag/P.M - 1) < 0.05;
        if ~opts.quiet
            fprintf(['  clear_build %s: tilt %+.3f deg, traced M %.5fx, ' ...
                     'exit %.3f mm, collimation %.2f urad\n'], deck, ...
                    D.tilt_deg, out.traced.mag, out.traced.exit_dia*1e3, ...
                    out.traced.collimation_urad);
        end
    end
end

% =====================================================================
function s = traced_(nE, Dap)
%TRACED_  Exit beam and collimation of the deck currently loaded.  Same
%   construction AFOCAL4_BUILD uses, repeated here because the tilt is
%   applied AFTER that function has returned and its own verification would
%   report the untilted train.
    tr = macos.trace(nE);   ri = macos.get_ray_info(tr.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
    dd = ri.dir(:,ok);   dd = dd ./ vecnorm(dd);
    dm = mean(dd,2);     dm = dm/norm(dm);
    q  = ri.pos(:,ok) - mean(ri.pos(:,ok),2);
    q  = q - dm*(dm.'*q);
    dia = 2*max(vecnorm(q));
    s = struct('exit_dia',dia, 'mag',Dap/max(dia,realmin), ...
               'collimation_urad', max(acos(min(1, dm.'*dd)))*1e6, ...
               'nrays', nnz(ok));
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
