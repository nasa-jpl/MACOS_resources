function [D, ok] = wall_recover(P, deck, opts)
%WALL_RECOVER  The design struct behind a committed afocal4 deck, verified.
%
%   [D, OK] = WALL_RECOVER(P, DECK) reads the design back out of a committed
%   prescription (RESULTS rule 9: an afocal4 design is fully recoverable
%   from its own `.in`) and then REBUILDS the deck from it and compares the
%   two byte for byte.  OK is that comparison.
%
%   WHY IT VERIFIES RATHER THAN TRUSTS.  This is the third copy of this
%   recovery in the study -- AFOCAL4_CLEARING has one and TAFOCAL4CLEAR has
%   one -- and the one thing that can go wrong with it is silent: it
%   returns a design that is nearly the committed one and every number
%   afterwards is measured on a deck nobody published.  It has already
%   happened once here, and the trap is worth restating:
%
%     READ THE SPACINGS FROM zElt, NOT FROM THE VERTICES.  The builder poses
%     the interface plane on the TRACED CHIEF, so on this deck the last
%     mirror's vertex sits 359 mm from the interface vertex while the
%     interface standoff is 343.  The vertex reading rebuilds a deck that is
%     not this one, and it silently shifted a whole scan.
%
%   Name-value:
%     'verify'  rebuild and compare (true).  With the union wall enabled in
%               P this would be judged by it, so the check is run with a
%               copy of P that has the wall OFF -- the committed deck fails
%               that wall by construction, which is the whole reason the
%               stage exists.
%
%   See also AFOCAL4_BUILD, WALL_SEED, WALL_POINT.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.verify (1,1) logical = true
    end

    txt = fileread(deck);
    Kc  = grab1_(txt,'KcElt');   Kr = grab1_(txt,'KrElt');
    zE  = grab1_(txt,'zElt');
    nM  = numel(Kc) - 1;                       % the last element is the interface
    D = struct('form','field', 'K',Kc(1:nM).', 'bias_deg',P.bias_deg, ...
               'ngrid',P.ngrid, 'rb',zeros(numel(P.rb_elts),2), 'tilt_deg',0);
    D.R2    = abs(Kr(2));
    D.t1    = zE(1);
    D.iface = zE(nM);
    fo = afocal_first_order([abs(Kr(1)) abs(Kr(2))], D.t1, ...
                            [false true], 'D',P.D, 'stop_ahead',P.stop_ahead);
    D.fm_standoff = -fo.y_marginal(2)/fo.u_marginal(2) - zE(2);

    ok = true;
    if opts.verify
        Q = P;   Q.pack.union_enforce = false;
        t = [tempname '.in'];
        c = onCleanup(@() del_(t)); %#ok<NASGU>
        afocal4_build(Q, D, t, 'verify',false);
        ok = isequal(fileread(t), txt);
        if ~ok
            error('macos:design:wall_recover:mismatch', ...
                  ['the recovered design does not rebuild %s byte for byte ' ...
                   '-- every number measured from here would belong to a ' ...
                   'deck nobody published.'], deck);
        end
    end
end

function v = grab1_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    v = zeros(1, numel(t));
    for i = 1:numel(t), v(i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 1); end
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
