function out = clear_tilt(deck_in, spec, deck_out, opts)
%CLEAR_TILT  Extraction tilt: swing one mirror, carry the train with it.
%
%   OUT = CLEAR_TILT(DECK_IN, SPEC, DECK_OUT) tilts ONE powered mirror of a
%   committed prescription by SPEC.alpha and re-poses every element
%   downstream of it by the rigid motion that keeps the CHIEF RAY's path
%   exactly what it was.  Nothing else in the deck changes: conics, radii,
%   spacings, the source and every upstream element are carried across
%   byte-for-byte.
%
%   WHY THIS MOVE EXISTS.  The collimator of the afocal4 family-2 design
%   stands inside its own M2->field-mirror feed beam, and CLEAR_SCAN shows
%   the reason is a RATIO LAW: the two bundles' footprints are both scaled
%   copies of the same off-axis field box, so they can only come apart if
%   their scales differ by more than (bias + half) / (bias - half) = 2.43,
%   which this topology never reaches.  Every FIELD-PROPORTIONAL remedy --
%   a different collimator station, a different interface standoff, a flat
%   fold anywhere in either leg -- is bound by that law.  A TILT is not: it
%   separates the two bundles by 2*alpha*d, a length that does not know what
%   field it is carrying.  That is the whole argument for spending
%   aberration on it.
%
%   THE TILT IS EXACT FOR THE CHIEF, AND MEASURED, NOT ASSUMED.  The mirror
%   turns about the point Q where the chief ray actually strikes it -- taken
%   from the traced ray history, never from the vertex -- so Q stays on the
%   surface and the incoming chief still lands there.  The new outgoing
%   chief is then computed from the ROTATED SURFACE NORMAL at Q, and the
%   downstream isometry is the rotation about Q that carries the old
%   outgoing chief onto the new one.  The surface normal at Q is itself
%   engine truth: N = unit(d_in - d_out) from the traced chief legs, so no
%   conic sag algebra is repeated here and a mirror whose vertex normal is
%   not its local normal (which is every one of these, the chief lands
%   184 mm off the field mirror's vertex) is handled correctly.
%
%   WHAT IT COSTS, AND WHY IT IS NOT A FOLD.  A flat fold is an isometry of
%   the light; this is not.  The rotated surface differs from the original
%   one everywhere except at Q, so every ray that is not the chief sees a
%   different mirror.  On a mirror at the field conjugate the leading term
%   is a PUPIL shift, which is exactly the quantity the fourth mirror was
%   added to control -- so the price must be read off AFOCAL4_SCORE's pupil
%   ladder and not just off the wavefront.  CLEAR_PRICE measures it.
%
%   SPEC fields:
%     .elt     element to tilt: index, or its EltName ('FM')
%     .alpha   tilt angle, rad (signed)
%     .axis    1x3 rotation axis, global (default [1 0 0] -- the x axis,
%              which swings the beam in the y-z plane the field bias lives
%              in).  Need not be normalised.
%     .about   'chief' (default) | 'vertex'.  'vertex' is the ordinary
%              rigid-body convention and does NOT preserve the chief path;
%              it is here so the two can be compared.
%
%   Name-value:
%     'quiet'  (true)
%
%   Returns OUT with .deck .elt .alpha .Q (the pivot) .din .dout .dout_new
%   .Rm (the mirror's own rotation) .Rd (the downstream rotation) .turn_deg
%   (the angle the beam was actually turned through) .names .vpt .psi.
%
%   See also CLEAR_SCAN, CLEAR_PRICE, AFOCAL4_UNION, PACK_FOLD.

    arguments
        deck_in  (1,:) char
        spec (1,1) struct
        deck_out (1,:) char
        opts.quiet (1,1) logical = true
    end

    if ~isfield(spec,'axis')  || isempty(spec.axis),  spec.axis  = [1 0 0]; end
    if ~isfield(spec,'about') || isempty(spec.about), spec.about = 'chief'; end
    ax = spec.axis(:)/norm(spec.axis);

    txt   = fileread(deck_in);
    names = grab_names_(txt);
    Vs    = grab_all3_(txt,'VptElt');
    Ps    = grab_all3_(txt,'psiElt');
    nE    = size(Vs,2);
    if numel(names) ~= nE
        error('clear_tilt:names', ...
              '%s: %d EltName lines but %d VptElt blocks.', deck_in, ...
              numel(names), nE);
    end
    k = elt_index_(names, spec.elt);
    if k >= nE
        error('clear_tilt:last', 'cannot tilt the last element (%s).', names{k});
    end

    % ---- the chief ray at that mirror, from the engine -------------------
    macos.load_rx(deck_in);
    macos.ray_hist('on');
    tr = macos.trace(nE);
    h  = macos.ray_hist(tr.nRays);
    macos.ray_hist('off');
    Pc  = squeeze(h.P(:,1,:));            % chief polyline; column k+off = elt k
    off = size(h.P,3) - nE;
    Q    = Pc(:, k+off);
    din  = Q - Pc(:, k+off-1);   din  = din/norm(din);
    dout = Pc(:, k+off+1) - Q;   dout = dout/norm(dout);

    % The LOCAL surface normal at Q, from the two traced legs.  A reflection
    % obeys d_out = d_in - 2(d_in.N)N, so N is parallel to d_in - d_out --
    % and that is true whatever the surface is.  Reading it from the trace
    % instead of from KcElt/KrElt keeps this routine surface-agnostic and
    % keeps it honest about a mirror used far off its own vertex.
    N = din - dout;
    if norm(N) < 1e-12
        error('clear_tilt:normal', ...
              'element %s does not deflect the chief ray -- nothing to tilt.', names{k});
    end
    N = N/norm(N);

    Rm = rot_(ax, spec.alpha);            % the mirror's own rotation
    if strcmp(spec.about,'vertex'), Qp = Vs(:,k); else, Qp = Q; end

    % ---- where the chief goes now ----------------------------------------
    Nn    = Rm*N;
    dnew  = din - 2*(din.'*Nn)*Nn;   dnew = dnew/norm(dnew);
    % the downstream rigid motion: the rotation about Qp carrying dout to dnew
    Rd    = rot_between_(dout, dnew);
    turn  = rad2deg(acos(max(-1,min(1, dout.'*dnew))));

    % ---- apply ------------------------------------------------------------
    V = Vs;  Psi = Ps;
    V(:,k)   = Qp + Rm*(Vs(:,k) - Qp);
    Psi(:,k) = Rm*Ps(:,k);
    for j = k+1:nE
        V(:,j)   = Qp + Rd*(Vs(:,j) - Qp);
        Psi(:,j) = Rd*Ps(:,j);
    end

    txt2 = txt;
    for j = k:nE
        txt2 = set_elt_pose_(txt2, j, Psi(:,j), V(:,j));
    end
    % zElt IS DELIBERATELY LEFT ALONE.  In these decks it is the EMITTER's
    % declared design spacing, not a measured vertex distance -- on the
    % committed 343 mm deck the last mirror's zElt reads 0.343 while its
    % vertex is 0.359 m from the interface plane's, because the interface is
    % posed on the traced chief.  The tilt changes the POSE, not the design,
    % and the trace reads VptElt/psiElt only.  Re-deriving zElt as a vertex
    % distance would look tidier and would quietly turn the declared
    % standoff into a different number -- which is exactly what a later
    % reader (or AFOCAL4_CLEARING's own design recovery) would take it for.
    write_(deck_out, txt2);

    out = struct('deck',deck_out, 'elt',k, 'name',names{k}, ...
                 'alpha',spec.alpha, 'axis',ax.', 'about',spec.about, ...
                 'Q',Q.', 'pivot',Qp.', 'N',N.', 'din',din.', 'dout',dout.', ...
                 'dout_new',dnew.', 'Rm',Rm, 'Rd',Rd, 'turn_deg',turn, ...
                 'names',{names}, 'vpt',V, 'psi',Psi);
    if ~opts.quiet
        fprintf(['  clear_tilt: %s tilted %.4f deg about [%.3f %.3f %.3f] ' ...
                 'at the chief point [%+.4f %+.4f %+.4f]\n'], names{k}, ...
                rad2deg(spec.alpha), ax, Q);
        fprintf('    beam turned %.4f deg; %d downstream element(s) re-posed\n', ...
                turn, nE-k);
    end
end

% =====================================================================
function R = rot_(ax, th)
    a = ax(:)/norm(ax);
    K = [0 -a(3) a(2); a(3) 0 -a(1); -a(2) a(1) 0];
    R = eye(3) + sin(th)*K + (1-cos(th))*(K*K);
end

function R = rot_between_(u, v)
%ROT_BETWEEN_  The rotation carrying unit U onto unit V about U x V.  For
%   the anti-parallel case there is no unique answer, and this study never
%   asks for one -- a tilt that reverses the chief is not a tilt.
    u = u(:)/norm(u);   v = v(:)/norm(v);
    c = max(-1, min(1, u.'*v));
    w = cross(u, v);
    if norm(w) < 1e-14
        if c > 0, R = eye(3);
        else, error('clear_tilt:reverse', 'the tilt reverses the chief ray.');
        end
        return;
    end
    R = rot_(w/norm(w), acos(c));
end

function k = elt_index_(names, who)
    if isnumeric(who), k = who;  return; end
    k = find(strcmp(names, who), 1);
    if isempty(k)
        error('clear_tilt:elt', 'element ''%s'' not found (have: %s)', ...
              who, strjoin(names,' '));
    end
end

function txt = set_elt_pose_(txt, k, psi, Vpt)
%SET_ELT_POSE_  Rewrite element K's psiElt / VptElt / RptElt and its TElt
%   frame block.  Line-based, block-delimited by iElt= -- a global regexp
%   would hit every element at once.  The TElt frame is REBUILT from psi in
%   the emitter's convention rather than transformed, so a deck this routine
%   edits stays consistent with one it did not (the PACK_FOLD rule).
    psi = psi(:)/norm(psi);   Vpt = Vpt(:);
    L   = strsplit(txt, newline, 'CollapseDelimiters', false);
    ie  = find(~cellfun('isempty', regexp(L, '^\s*iElt=', 'once')));
    if k > numel(ie)
        error('clear_tilt:elt', 'element %d requested, deck has %d.', k, numel(ie));
    end
    lo = ie(k);
    if k < numel(ie), hi = ie(k+1) - 1; else, hi = numel(L); end
    R  = surf_frame_(psi);
    v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
    v6 = @(u,w) sprintf('%.16E  %.16E  %.16E  %.16E  %.16E  %.16E', ...
                        u(1),u(2),u(3),w(1),w(2),w(3));
    for i = lo:hi
        s = L{i};
        if     ~isempty(regexp(s,'^\s*psiElt=','once'))
            L{i} = ['           psiElt=  ' v3(psi)];
        elseif ~isempty(regexp(s,'^\s*VptElt=','once'))
            L{i} = ['           VptElt=  ' v3(Vpt)];
        elseif ~isempty(regexp(s,'^\s*RptElt=','once'))
            L{i} = ['           RptElt=  ' v3(Vpt)];
        elseif ~isempty(regexp(s,'^\s*TElt=','once'))
            L{i}   = ['             TElt=  ' v6(R(:,1),[0;0;0])];
            L{i+1} = ['                    ' v6(R(:,2),[0;0;0])];
            L{i+2} = ['                    ' v6(R(:,3),[0;0;0])];
            L{i+3} = ['                    ' v6([0;0;0],R(:,1))];
            L{i+4} = ['                    ' v6([0;0;0],R(:,2))];
            L{i+5} = ['                    ' v6([0;0;0],R(:,3))];
        end
    end
    txt = strjoin(L, newline);
end

function R = surf_frame_(psi)
    z = psi(:)/norm(psi);
    yh = [0;1;0];   if abs(z(2)) > 0.95, yh = [1;0;0]; end
    y  = yh - (yh.'*z)*z;   y = y/norm(y);
    x  = cross(y, z);
    R  = [x, y, z];
end

function M = grab_all3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3); end
end

function n = grab_names_(txt)
    t = regexp(txt, '(?m)^\s*EltName=\s*(\S*)', 'tokens');
    n = cellfun(@(c) c{1}, t, 'UniformOutput', false);
end

function write_(f, txt)
    fid = fopen(f,'w');
    if fid < 0, error('clear_tilt:write', 'cannot write %s', f); end
    fprintf(fid,'%s',txt);   fclose(fid);
end
