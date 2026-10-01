function M = spectrometer_rx(G, file, opts)
%SPECTROMETER_RX  Emit a MACOS .in from a spectrometer_geom chain.
%   M = spectrometer_rx(G, file) writes the deck and returns the element
%   map M (.iG grating, .iFPA focal plane, .iReturn, .nElt, .file).
%
%   Conventions (the emission rules the gate tSpectrometerRx checks):
%   - spheres: VptElt = the chain's vertex point on the sphere, psiElt ->
%     the centre of curvature, KrElt = -R for EVERY sphere (concave and
%     convex alike, the Telescope-builder rule); planes: KrElt = -1e22,
%     psiElt = the chain's plane normal.
%   - Refractor: IndRef = the index AFTER the surface; a glass medium is
%     written as IndRef = 1 + GlassElt= <name> so the engine's Sellmeier
%     applies at every trace (engine fix b000390).
%   - Grating: Element= Grating, IndRef = the surrounding medium (1 here),
%     Extinc = 1e22, h1HOE = the dispersion direction (the engine projects
%     it into the tangent plane), OrderHOE = m (signed), RuleWidth = d in
%     BaseUnits.  A grating immersed in glass would carry GlassElt itself.
%   - point source: SrcPos = ChfRayPos + zSource*ChfRayDir = the slit point;
%     Aperture = the FULL cone angle 2u in radians (PtSource rotates the
%     chief by +-u).
%   - tail: a pass-through Reference 1 mm upstream of the FocalPlane, so
%     the grating is < nElt-2 and macos.stop(iG) is accepted (a Return
%     coincident with the FPA zeroes the last path length and the engine
%     drops the rays).
%   - ChfRayPos = slit + 0.2 mm along the chief (the engine starts rays
%     THERE and only sees surfaces beyond it), zSource = -0.2 mm.
%
%   Options: 'ngridpts' (41), 'wavelen' (G.src.lambda_c), 'name'.
    arguments
        G struct
        file (1,:) char
        opts.ngridpts (1,1) double = 41
        opts.wavelen (1,1) double = NaN
        opts.name (1,:) char = ''
        opts.terminal (1,:) char {mustBeMember(opts.terminal, {'geometric','farfield'})} = 'geometric'
        opts.L_ref (1,1) double {mustBePositive} = 0.1
    end
    if isnan(opts.wavelen), opts.wavelen = G.src.lambda_c; end
    if isempty(opts.name), opts.name = ['spectrometer_' G.form]; end
    F = @(v) sprintf('%.15G  %.15G  %.15G', v(1), v(2), v(3));
    d0 = G.src.chief_dir(:);
    xg = [1;0;0];  yg = cross(d0, xg);  yg = yg/norm(yg);  xg = cross(yg, d0);
    pos0 = G.slit(:) + G.src.zsrc_gap*d0;

    S = G.surf;  nS = numel(S);
    E = {};                                    % element records in order
    for k = 1:nS
        s = S(k);
        e = struct('name', s.name, 'surface', 'Flat', 'Kr', -1e22, 'Kc', 0, ...
                   'psi', s.psi(:), 'vpt', s.C(:), 'indref', 1, 'extinc', 0, ...
                   'glass', '', 'element', '', 'grating', [], 'proptype', 'Geometric', 'zelt', 1e22);
        if strcmp(s.kind, 'sphere')
            e.surface = 'Conic';  e.Kr = -s.R;  e.vpt = s.vpt(:);  e.psi = s.psi(:);
        end
        switch s.act
        case 'refract'
            e.element = 'Refractor';
            if ischar(s.n_out), e.glass = s.glass;  e.indref = 1; else, e.indref = s.n_out; end
        case 'reflect'
            e.element = 'Reflector';  e.extinc = 1e22;
        case 'grating'
            e.element = 'Grating';  e.extinc = 1e22;
            e.grating = struct('dir', G.grating.sdir(:), 'm', G.grating.m, 'd', G.grating.d);
        case 'stop'
            switch opts.terminal
            case 'geometric'
                % the FPA: a pass-through Reference 1 mm upstream + FocalPlane
                % (a Return COINCIDENT with the FocalPlane leaves the FPA with
                % zero path length and the engine drops those rays as a miss;
                % the Reference only exists so the grating index is < nElt-2,
                % the stop wrapper's range)
                e.element = 'Reference';  e.name = 'PreFPA';
                e.vpt = s.C(:) + 1e-3*s.psi(:);        % psi points against the beam
                E{end+1} = e;                                        %#ok<AGROW>
                e.element = 'FocalPlane';  e.name = 'FPA';  e.vpt = s.C(:);
            case 'farfield'
                % the Rx_Cass_FarField idiom, posed on the chief: FP_return
                % (Return, flat, AT the FPA) -> ExitPupil (Return, sphere of
                % radius L_ref centred on the FPA chief point, vertex L_ref
                % UPSTREAM along the chief, psi along the beam toward the
                % focus, KrElt = -L_ref, zElt = L_ref, PropType FarField) ->
                % FPA.  The sphere is a REFERENCE sphere, not the exit pupil
                % (the Offner's true exit pupil is at infinity -- telecentric
                % -- and FEX's 484 m crossing lies past the focus, where the
                % reversed rays never go).  spectrometer_wave re-poses it per
                % field and wavelength with macos.set_xp.
                [pts, dirs, ok] = G.trace(G.slit, G.src.chief_dir, G.src.lambda_c);
                assert(ok, 'spectrometer_rx: chief does not reach the FPA');
                pc = pts(:, end);  din = dirs(:, end-1);  din = din/norm(din);
                e.element = 'Return';  e.name = 'FP_return';  e.vpt = pc;  e.psi = din;
                E{end+1} = e;                                        %#ok<AGROW>
                e.element = 'Return';  e.name = 'ExitPupil';  e.surface = 'Conic';
                e.Kr = -opts.L_ref;  e.vpt = pc - opts.L_ref*din;  e.psi = din;
                e.proptype = 'FarField';  e.zelt = opts.L_ref;
                E{end+1} = e;                                        %#ok<AGROW>
                e = struct('name', 'FPA', 'surface', 'Flat', 'Kr', -1e22, 'Kc', 0, ...
                           'psi', din, 'vpt', pc, 'indref', 1, 'extinc', 0, 'glass', '', ...
                           'element', 'FocalPlane', 'grating', [], 'proptype', 'Geometric', 'zelt', 1e22);
            end
        end
        E{end+1} = e;                                                %#ok<AGROW>
    end
    nElt = numel(E);
    M.iG = find(cellfun(@(e) strcmp(e.element, 'Grating'), E));
    M.iFPA = nElt;  M.iRef = nElt - 1;  M.nElt = nElt;  M.file = file;  M.terminal = opts.terminal;
    if strcmp(opts.terminal, 'farfield'), M.iEP = nElt - 1;  M.iFPr = nElt - 2;  M.L_ref = opts.L_ref; end

    ln = {};
    ln{end+1} = sprintf('%% %s -- generated by spectrometer_rx (form %s, m=%+d, d=%.6g m, %.2f l/mm)', ...
        opts.name, G.form, G.grating.m, G.grating.d, G.grating.lines_per_mm);
    ln{end+1} = sprintf('        ChfRayDir=  %s', F(d0));
    ln{end+1} = sprintf('        ChfRayPos=  %s', F(pos0));
    ln{end+1} = sprintf('          zSource=  %.10G', -G.src.zsrc_gap);
    ln{end+1} =         '        BaseUnits=  m';
    ln{end+1} =         '        WaveUnits=  m';
    ln{end+1} =         '           IndRef=  1.0D+00';
    ln{end+1} =         '           Extinc=  0.0D+00';
    ln{end+1} = sprintf('          Wavelen=  %.9E', opts.wavelen);
    ln{end+1} =         '             Flux=  1.0D+00';
    ln{end+1} = sprintf('         Aperture=  %.12E', 2*G.src.u);
    ln{end+1} =         '         Obscratn=  0.0D+00';
    ln{end+1} =         '         GridType=  Circular';
    ln{end+1} = sprintf('         nGridpts=  %d', opts.ngridpts);
    ln{end+1} = sprintf('            xGrid=  %s', F(xg));
    ln{end+1} = sprintf('            yGrid=  %s', F(yg));
    ln{end+1} = sprintf('             nElt=  %d', nElt);
    for k = 1:nElt
        e = E{k};
        ln{end+1} = '';                                              %#ok<AGROW>
        ln{end+1} = sprintf('             iElt=  %d', k);
        ln{end+1} = sprintf('          EltName=  %s', e.name);
        ln{end+1} = sprintf('          Element=  %s', e.element);
        ln{end+1} = sprintf('          Surface=  %s', e.surface);
        ln{end+1} = sprintf('            KrElt=  %.10E', e.Kr);
        ln{end+1} = sprintf('            KcElt=  %.10E', e.Kc);
        ln{end+1} = sprintf('           psiElt=  %s', F(e.psi));
        ln{end+1} = sprintf('           VptElt=  %s', F(e.vpt));
        ln{end+1} = sprintf('           RptElt=  %s', F(e.vpt));
        ln{end+1} = sprintf('           IndRef=  %.6E', e.indref);
        if ~isempty(e.glass)
            ln{end+1} = sprintf('         GlassElt=  %s', e.glass);
        end
        ln{end+1} = sprintf('           Extinc=  %.6E', e.extinc);
        if ~isempty(e.grating)
            ln{end+1} = sprintf('            h1HOE=  %s', F(e.grating.dir));
            ln{end+1} = sprintf('         OrderHOE=  %d', e.grating.m);
            ln{end+1} = sprintf('        RuleWidth=  %.12E', e.grating.d);
        end
        ln{end+1} =         '            nCoat=  0';
        ln{end+1} =         '             nObs=  0';
        ln{end+1} =         '           ApType=  None';
        ln{end+1} = sprintf('         PropType=  %s', e.proptype);
        ln{end+1} = sprintf('             zElt=  %.15E', e.zelt);
        ln{end+1} =         '          nECoord=  -6';
    end
    ln{end+1} = '';
    ln{end+1} = '         nOutCord=  5';
    ln{end+1} = '             Tout=  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00';
    ln{end+1} = '                    0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00';
    ln{end+1} = '                    0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00';
    ln{end+1} = '                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00';
    ln{end+1} = '                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00';
    fid = fopen(file, 'w');  assert(fid > 0, 'spectrometer_rx: cannot write %s', file);
    fprintf(fid, '%s\n', ln{:});  fclose(fid);
end
