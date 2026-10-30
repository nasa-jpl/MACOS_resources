function txt = oi_deck(X)
%OI_DECK  Emit the offset_imager prescription text from a design struct.
%
%   TXT = OI_DECK(X) returns the .in text for the template's five-element
%   train: M1, Stop (a Reference element carrying the system stop), M2,
%   M3, FP.  Write it to a file with oi_write.  Pure text emission -- no
%   engine calls; the conventions are the rodgers3 Stage-0 set (one
%   global frame, metres, beam enters +z, KrElt = signed CODE V radius,
%   psiElt (0,0,-1)-style tilted by alpha about x).
%
%   The stop is a REFERENCE ELEMENT at the stop plane with its vertex AT
%   the stop centre, so the engine's native element-bound stop machinery
%   (macos.stop(2,[0 0]) -> ChiefRayAiming) aims every field's chief
%   through it by real-ray iteration.  The header ApStop= (StopPos) form
%   is NOT used: it aims geometrically with no optics traversal
%   (srcaim.inc), which is wrong for this stop -- it sits behind M1.
%
%   Design struct X:
%     .R        1x3  signed radii, m          .K     1x3  conics
%     .asph     3x3  engine AsphCoefs per mirror (h^4,h^6,h^8 terms,
%                    m^-3/-5/-7; a zero row emits Surface=Conic)
%     .zern     1x3  cell, [] or struct('modes',1xN,'coef',1xN,'lMon',s)
%                    BornWolf, engine mode numbers, coef in m.  A mirror
%                    with a zern emits Surface=Zernike (conic + Zernike;
%                    its asph row must be zero).
%     .yde      1x3  mirror decenters, m (global y)
%     .ade      1x3  mirror tilts about x, deg (macos alpha sense)
%     .z_m1     M1 vertex z, m
%     .spacings 1x3  SIGNED [m1->stop, stop->m2, m2->m3], m
%     .stopC    3x1  stop centre (global, m)
%     .fpa      struct('Vpt',3x1,'psi',3x1)  posed focal plane
%     .EPD_m, .WL_m, .sampling, .name
%
%   Source lines are placeholders -- OI_SCORE re-emits ChfRayDir/Pos per
%   field, exactly as the rodgers3 gate runner does.
%
%   See also OI_WRITE, OI_SCORE, OFFSET_IMAGER_PARAMS.

    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));

    % ---- stations --------------------------------------------------------
    z_m1   = X.z_m1;
    z_stop = z_m1   + X.spacings(1);
    z_m2   = z_stop + X.spacings(2);
    z_m3   = z_m2   + X.spacings(3);
    zEl    = [z_m1, z_m2, z_m3];

    % ---- mirror frames (decenter, then tilt about x -- CODE V order) -----
    M = struct('Vpt',{},'psi',{},'frame',{});
    for m = 1:3
        al = X.ade(m);
        M(m).Vpt = [0; X.yde(m); zEl(m)];
        M(m).psi = [0;  sind(al); -cosd(al)];
        M(m).frame = struct('x',[1;0;0], ...
                            'y',[0;  cosd(al); sind(al)], ...
                            'z',[0; -sind(al); cosd(al)]);
    end

    % ---- header ----------------------------------------------------------
    s = sprintf('%% MACOS prescription -- offset_imager template (%s)\n', X.name);
    s = [s sprintf('%% emitted by oi_deck.m; stop = Reference elt 2 (native STOP aiming)\n')];
    s = [s sprintf('%% Source Definition\n')];
    s = [s sprintf('        ChfRayDir=  %s\n', v3([0;0;1]))];
    s = [s sprintf('        ChfRayPos=  %s\n', v3([0;0;-1]))];
    s = [s sprintf('          zSource=1.0E+22\n')];
    s = [s sprintf('        BaseUnits=  m\n')];
    s = [s sprintf('        WaveUnits=  m\n')];
    s = [s sprintf('           IndRef=1.0E+00\n')];
    s = [s sprintf('           Extinc=0.0E+00\n')];
    s = [s sprintf('          Wavelen=%.16E\n', X.WL_m)];
    s = [s sprintf('             Flux=1.0E+00\n')];
    s = [s sprintf('         Aperture=%.16E\n', X.EPD_m)];
    s = [s sprintf('         Obscratn=0.0E+00\n')];
    s = [s sprintf('         GridType=  Circular\n')];
    s = [s sprintf('         nGridpts=  %d\n', X.sampling)];
    s = [s sprintf('            xGrid=  %s\n', v3([1;0;0]))];
    s = [s sprintf('            yGrid=  %s\n', v3([0;1;0]))];
    s = [s sprintf('%% Element Definitions\n')];
    s = [s sprintf('             nElt=  5\n')];

    % ---- element 1: M1 ----------------------------------------------------
    s = [s elt_mirror_(1, 'M1', X, M(1), abs(z_stop - z_m1), v3)];

    % ---- element 2: the stop (Reference) ----------------------------------
    s = [s sprintf('             iElt=  2\n')];
    s = [s sprintf('          EltName=  Stop\n')];
    s = [s sprintf('          Element=  Reference\n')];
    s = [s sprintf('          Surface=  Flat\n')];
    s = [s sprintf('            KrElt=-1.0000000000000000E+22\n')];
    s = [s sprintf('            KcElt=0.0000000000000000E+00\n')];
    s = [s sprintf('           psiElt=  %s\n', v3([0;0;-1]))];
    s = [s sprintf('           VptElt=  %s\n', v3(X.stopC))];
    s = [s sprintf('           RptElt=  %s\n', v3(X.stopC))];
    s = [s sprintf('           IndRef=1.0E+00\n')];
    s = [s sprintf('           Extinc=0.0E+00\n')];
    s = [s sprintf('             nObs=  0\n')];
    s = [s sprintf('           ApType=  None\n')];
    s = [s sprintf('         PropType=  Geometric\n')];
    s = [s sprintf('             zElt=%.16E\n', abs(X.spacings(2)))];

    % ---- elements 3,4: M2, M3 ---------------------------------------------
    s = [s elt_mirror_(3, 'M2', X, M(2), abs(z_m3 - z_m2), v3, 2)];
    s = [s elt_mirror_(4, 'M3', X, M(3), norm(X.fpa.Vpt - M(3).Vpt), v3, 3)];

    % ---- element 5: FP -----------------------------------------------------
    s = [s sprintf('             iElt=  5\n')];
    s = [s sprintf('          EltName=  FP\n')];
    s = [s sprintf('          Element=  FocalPlane\n')];
    s = [s sprintf('          Surface=  Flat\n')];
    s = [s sprintf('            KrElt=-1.0000000000000000E+22\n')];
    s = [s sprintf('            KcElt=0.0000000000000000E+00\n')];
    s = [s sprintf('           psiElt=  %s\n', v3(X.fpa.psi))];
    s = [s sprintf('           VptElt=  %s\n', v3(X.fpa.Vpt))];
    s = [s sprintf('           RptElt=  %s\n', v3(X.fpa.Vpt))];
    s = [s sprintf('           IndRef=1.0E+00\n')];
    s = [s sprintf('           Extinc=0.0E+00\n')];
    s = [s sprintf('             nObs=  0\n')];
    s = [s sprintf('           ApType=  None\n')];
    s = [s sprintf('         PropType=  Geometric\n')];
    s = [s sprintf('             zElt=1.0000000000000000E+20\n')];

    % ---- output coordinate system ------------------------------------------
    s = [s sprintf('%% Output Coordinate System Definition\n')];
    s = [s sprintf('         nOutCord=  5\n')];
    T = [1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 0 1 0 0 0; ...
         0 0 0 0 1 0 0; 0 0 0 0 0 0 1];
    s = [s sprintf('             Tout=')];
    for r = 1:5
        if r > 1, s = [s sprintf('                  ')]; end %#ok<AGROW>
        s = [s sprintf('  %s\n', strtrim(sprintf('%.16E  ', T(r,:))))]; %#ok<AGROW>
    end
    txt = s;
end

% =========================================================================
function s = elt_mirror_(ie, nm, X, E, zNext, v3, m)
    if nargin < 7, m = 1; end
    zr = X.zern{m};
    hasA = any(X.asph(m,:) ~= 0);
    s = sprintf('             iElt=  %d\n', ie);
    s = [s sprintf('          EltName=  %s\n', nm)];
    s = [s sprintf('          Element=  Reflector\n')];
    if ~isempty(zr)
        if hasA
            error('oi_deck:zern_asph', ...
                  '%s carries both an asphere and a Zernike -- not emittable', nm);
        end
        s = [s sprintf('          Surface=  Zernike\n')];
    elseif hasA
        s = [s sprintf('          Surface=  Aspheric\n')];
    else
        s = [s sprintf('          Surface=  Conic\n')];
    end
    s = [s sprintf('            KrElt=%.16E\n', X.R(m))];
    s = [s sprintf('            KcElt=%.16E\n', X.K(m))];
    if hasA
        s = [s sprintf('       nAsphCoefs=  3\n')];
        s = [s sprintf('        AsphCoefs=  %.16E %.16E %.16E\n', X.asph(m,:))];
    end
    if ~isempty(zr)
        n = numel(zr.modes);
        s = [s sprintf('         ZernType=  BornWolf\n')];
        s = [s sprintf('             lMon=%.16E\n', zr.lMon)];
        s = [s sprintf('             pMon=  %s\n', v3(E.Vpt))];
        s = [s sprintf('             xMon=  %s\n', v3(E.frame.x))];
        s = [s sprintf('             yMon=  %s\n', v3(E.frame.y))];
        s = [s sprintf('             zMon=  %s\n', v3(E.frame.z))];
        s = [s sprintf('        nZernCoef=  %d\n', n)];
        s = [s sprintf('        ZernModes= %s\n', sprintf(' %d', zr.modes))];
        s = [s sprintf('         ZernCoef= %s\n', ...
                       sprintf(' %.16E', zr.coef(1:min(6,n))))];
        i = 7;
        while i <= n
            j = min(i+5, n);
            s = [s sprintf('                   %s\n', ...
                           sprintf(' %.16E', zr.coef(i:j)))]; %#ok<AGROW>
            i = j + 1;
        end
    end
    s = [s sprintf('           psiElt=  %s\n', v3(E.psi))];
    s = [s sprintf('           VptElt=  %s\n', v3(E.Vpt))];
    s = [s sprintf('           RptElt=  %s\n', v3(E.Vpt))];
    s = [s sprintf('           IndRef=1.0E+00\n')];
    s = [s sprintf('           Extinc=0.0E+00\n')];
    s = [s sprintf('             nObs=  0\n')];
    s = [s sprintf('           ApType=  None\n')];
    s = [s sprintf('         PropType=  Geometric\n')];
    s = [s sprintf('             zElt=%.16E\n', zNext)];
end
