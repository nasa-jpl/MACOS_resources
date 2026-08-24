function info = append_rx(base_in, add_in, out_in, opts)
%APPEND_RX  Splice one prescription's elements onto the end of another's.
%
%   info = MACOS.DESIGN.APPEND_RX(BASE_IN, ADD_IN, OUT_IN) writes OUT_IN:
%   the BASE deck's header and source, then the BASE deck's elements, then
%   the ADD deck's elements renumbered to follow them, then the standard
%   `nOutCord` terminator.  ADD's own header and source are discarded --
%   the spliced train has ONE source, the base's.
%
%   This is what a back end costs to attach: a `Bench` (or any other
%   builder) lays out the instrument from a chief-ray state, and the
%   result has to become MORE ELEMENTS of the telescope's train, not a
%   second deck traced separately.  Only one train can carry a telescope
%   perturbation through to a coronagraph contrast number.
%
%   THE TWO THINGS THAT MAKE THIS SAFE, both checked:
%     UNITS.  Both decks must declare the same `BaseUnits`.  Element
%       coordinates are raw numbers; splicing a millimetre bench onto a
%       metre telescope silently shrinks it by 1000 and everything still
%       "traces".  Build the add-on in the base's units (Bench takes a
%       'baseunits' option) -- APPEND_RX refuses to convert, because a
%       conversion has to touch lengths but not angles, indices, mode
%       numbers or Zernike orders, and a blanket scale gets that wrong.
%     GEOMETRY.  The add-on's first element must sit on the base's exit
%       chief ray.  APPEND_RX does not check this (it is text, not a
%       trace) -- the CALLER traces the result and verifies the ray count,
%       which is the only check that means anything.  Always do it.
%
%   Name-value:
%     'drop_base_tail'  how many of the BASE's trailing elements to drop
%                       (default 0).  A telescope deck ends in its focal
%                       plane, and a back end that re-images that focus
%                       usually wants it gone -- but `add_pupil`'s
%                       terminal quartet ends in FP_return / ExitPupil /
%                       FP, so dropping ONE leaves a dangling pupil.
%                       State the number; it is not guessable.
%     'drop_add_head'   how many of the ADD deck's leading elements to
%                       drop (default 0), e.g. a bench's entrance baffle
%                       that duplicates the telescope's stop.
%     'rename'          prefix applied to every appended EltName (default
%                       '' = keep).  Element names collide across
%                       builders; a prefix keeps a report readable.
%
%   Returns info with .n_base .n_add .n_out .baseunits .out.
%
%   See also macos.design.Bench, macos.design.Telescope.

    arguments
        base_in (1,:) char
        add_in  (1,:) char
        out_in  (1,:) char
        opts.drop_base_tail (1,1) double {mustBeInteger,mustBeNonnegative} = 0
        opts.drop_add_head  (1,1) double {mustBeInteger,mustBeNonnegative} = 0
        opts.rename         (1,:) char = ''
    end
    assert(isfile(base_in), 'append_rx: %s not found', base_in);
    assert(isfile(add_in),  'append_rx: %s not found', add_in);

    B = read_rx_(base_in);
    A = read_rx_(add_in);

    if ~strcmpi(B.units, A.units)
        error('macos:design:append_rx:units', ...
            ['BaseUnits differ: %s declares "%s", %s declares "%s".  Build ' ...
             'the add-on in the base''s units (Bench takes ''baseunits''); ' ...
             'append_rx will not convert.'], ...
            base_in, B.units, add_in, A.units);
    end

    eb = B.elts;  ea = A.elts;
    if opts.drop_base_tail > 0
        assert(numel(eb) > opts.drop_base_tail, ...
            'append_rx: drop_base_tail %d >= base element count %d', ...
            opts.drop_base_tail, numel(eb));
        eb = eb(1:end-opts.drop_base_tail);
    end
    if opts.drop_add_head > 0
        assert(numel(ea) > opts.drop_add_head, ...
            'append_rx: drop_add_head %d >= add element count %d', ...
            opts.drop_add_head, numel(ea));
        ea = ea(opts.drop_add_head+1:end);
    end
    if ~isempty(opts.rename)
        for k = 1:numel(ea)
            ea{k} = regexprep(ea{k}, '(?m)^(\s*EltName=\s*)', ...
                              ['$1' opts.rename '_'], 'once');
        end
    end

    all_e = [eb, ea];
    n = numel(all_e);
    for k = 1:n
        all_e{k} = regexprep(all_e{k}, '(?m)^(\s*iElt=\s*)\S+', ...
                             sprintf('$1 %d', k), 'once');
    end

    hdr = regexprep(B.header, '(?m)^(\s*nElt=\s*)\S+', sprintf('$1 %d', n), 'once');
    txt = [hdr sprintf('\n') strjoin(all_e, sprintf('\n')) sprintf('\n') B.tail];
    fid = fopen(out_in, 'w');
    assert(fid > 0, 'append_rx: cannot write %s', out_in);
    fprintf(fid, '%s', txt);  fclose(fid);

    info = struct('n_base', numel(eb), 'n_add', numel(ea), 'n_out', n, ...
                  'baseunits', B.units, 'out', out_in);
end

% =========================================================================
function R = read_rx_(f)
%READ_RX_  Split a prescription into header / element blocks / terminator.
%   Element blocks are delimited by the `iElt=` lines; the terminator is
%   the `% Output Coordinate System Definition` block the parser needs as
%   the element-list end marker (without it SMACOS loads nElt = 0).
    txt = fileread(f);
    u = regexp(txt, '(?m)^\s*BaseUnits=\s*(\S+)', 'tokens', 'once');
    R.units = '';  if ~isempty(u), R.units = u{1}; end

    iTail = regexp(txt, '(?m)^%\s*Output Coordinate System', 'once');
    if isempty(iTail)
        iTail = regexp(txt, '(?m)^\s*nOutCord=', 'once');
    end
    if isempty(iTail)
        error('macos:design:append_rx:terminator', ...
            ['append_rx: %s has no nOutCord terminator block -- the parser ' ...
             'needs it as the element-list end marker (without it SMACOS ' ...
             'loads nElt = 0).'], f);
    end
    body = txt(1:iTail-1);
    R.tail = txt(iTail:end);

    st = regexp(body, '(?m)^\s*iElt=', 'start');
    if isempty(st)
        error('macos:design:append_rx:noelts', ...
              'append_rx: %s declares no elements', f);
    end
    R.header = strip_trailing_(body(1:st(1)-1));
    R.elts = cell(1, numel(st));
    for k = 1:numel(st)
        if k < numel(st), e = body(st(k):st(k+1)-1); else, e = body(st(k):end); end
        R.elts{k} = strip_trailing_(e);
    end
end

function s = strip_trailing_(s)
    s = regexprep(s, '\s+$', '');
end
