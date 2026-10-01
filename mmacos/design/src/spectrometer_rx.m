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
%   Options: 'ngridpts' (41), 'wavelen' (G.src.lambda_c), 'name', 'terminal',
%   'L_ref', 'apertures', 'margin', 'footprints', 'links' (Link= on the
%   return-pass copies and the pre-FPA Reference), 'opt' (the CALIB block;
%   see below).  M.names / M.link / M.opt record what was written.
    arguments
        G struct
        file (1,:) char
        opts.ngridpts (1,1) double = 41
        opts.wavelen (1,1) double = NaN
        opts.name (1,:) char = ''
        opts.terminal (1,:) char {mustBeMember(opts.terminal, {'geometric','farfield'})} = 'geometric'
        opts.L_ref (1,1) double {mustBePositive} = 0.1
        opts.apertures (1,1) logical = false
        opts.margin (1,1) double = 5e-3
        opts.footprints = []
        opts.links (1,1) logical = false
        opts.opt = []
    end
    if isnan(opts.wavelen), opts.wavelen = G.src.lambda_c; end
    if isempty(opts.name), opts.name = ['spectrometer_' G.form]; end
    F = @(v) sprintf('%.15G  %.15G  %.15G', v(1), v(2), v(3));
    d0 = G.src.chief_dir(:);
    % native-optimize block (CALIB): opts.opt.fovs(i).slit/.dir are the fields
    % (field 1 becomes the header's ChfRayDir/Pos -- the engine's parse counts
    % it), .wavelens the lambda list (the first is the header's Wavelen, the
    % rest ArrWaveLen), .weights, .target ('SPOT'), .wf_elt (default the FPA),
    % .max_iters, .var(j).name/.mask (1x8 [TIP TILT CLOCK DX DY PIST ROC
    % CONIC])/.asph (AsphCoef term indices).  OptChfRayPos is the RAY START
    % (slit + gap along the chief), the same convention as the header.
    O = opts.opt;
    if ~isempty(O)
        d0 = O.fovs(1).dir(:);
        opts.wavelen = O.wavelens(1);
    end
    xg = [1;0;0];  yg = cross(d0, xg);  yg = yg/norm(yg);  xg = cross(yg, d0);
    pos0 = G.slit(:) + G.src.zsrc_gap*d0;
    if ~isempty(O), pos0 = O.fovs(1).slit(:) + G.src.zsrc_gap*d0; end

    S = G.surf;  nS = numel(S);
    % declared apertures (BRIEF_to_dyson5 addendum 6): every optical surface
    % gets ApType Circular, ApVec = (footprint radius + margin, xc, yc) in its
    % aperture frame, with xObs written as global x (projected) so the frame
    % is the one the footprints were measured in; the engine then vignettes
    % what it should and macos.view_rx draws real bodies.
    FP = opts.footprints;
    if opts.apertures && isempty(FP), FP = G.footprints(); end
    E = {};                                    % element records in order
    for k = 1:nS
        s = S(k);
        e = struct('name', s.name, 'surface', 'Flat', 'Kr', -1e22, 'Kc', 0, ...
                   'psi', s.psi(:), 'vpt', s.C(:), 'indref', 1, 'extinc', 0, ...
                   'glass', '', 'element', '', 'grating', [], 'proptype', 'Geometric', 'zelt', 1e22, 'asph', [], ...
                   'ap', [], 'xobs', []);
        if opts.apertures && ~strcmp(s.act, 'stop')
            e.ap = [FP(k).radius + opts.margin, FP(k).xc, FP(k).yc];  e.xobs = FP(k).xap(:);
        end
        if strcmp(s.kind, 'sphere') || strcmp(s.kind, 'asph')
            e.surface = 'Conic';  e.Kr = -s.R;  e.vpt = s.vpt(:);  e.psi = s.psi(:);
            if isfield(s, 'Kc'), e.Kc = s.Kc; end
            if isfield(s, 'A') && ~isempty(s.A) && any(s.A ~= 0)
                e.surface = 'Aspheric';  e.asph = s.A(:)';   % AsphCoef(i) -> h^(2i+2) of sag along +psi
            end
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
                           'element', 'FocalPlane', 'grating', [], 'proptype', 'Geometric', 'zelt', 1e22, 'asph', [], ...
                           'ap', [], 'xobs', []);
            end
        end
        E{end+1} = e;                                                %#ok<AGROW>
    end
    nElt = numel(E);
    M.iG = find(cellfun(@(e) strcmp(e.element, 'Grating'), E));
    M.iFPA = nElt;  M.iRef = nElt - 1;  M.nElt = nElt;  M.file = file;  M.terminal = opts.terminal;
    M.apertures = opts.apertures;  M.footprints = FP;  M.margin = opts.margin;
    if strcmp(opts.terminal, 'farfield'), M.iEP = nElt - 1;  M.iFPr = nElt - 2;  M.L_ref = opts.L_ref; end

    % double-pass links: the return-pass copy of a surface follows its first
    % pass (Link= i; the engine applies PERTURB / ROC / CONIC / ASPH to both),
    % and the pass-through Reference before the FPA follows the FPA
    names = cellfun(@(e) e.name, E, 'uni', 0);
    link = zeros(1, nElt);
    if opts.links
        for k = 1:nElt
            tw = regexprep(names{k}, {'_in$', 'In$', '^PreFPA$'}, {'_out', 'Out', 'FPA'});
            j = find(strcmp(names, tw), 1);
            % only a twin with the SAME psi is one physical surface under the
            % engine's link (a PIST moves both along psi): the block's flat
            % face is written with opposite normals on its two passes and
            % stays unlinked
            if ~isempty(j) && j ~= k && ~strcmp(names{k}, tw) && abs(E{k}.psi(:)'*E{j}.psi(:) - 1) < 1e-9
                link(k) = j;
            end
        end
    end
    M.names = names;  M.link = link;  M.opt = O;
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
    if ~isempty(O)
        tgt = 'SPOT';  if isfield(O, 'target') && ~isempty(O.target), tgt = O.target; end
        wfe = M.iFPA;  if isfield(O, 'wf_elt') && ~isempty(O.wf_elt), wfe = O.wf_elt; end
        mx = 20;       if isfield(O, 'max_iters') && ~isempty(O.max_iters), mx = O.max_iters; end
        w = ones(1, numel(O.fovs));  if isfield(O, 'weights') && ~isempty(O.weights), w = O.weights; end
        ln{end+1} = sprintf('        OptTarget=  %s', tgt);
        ln{end+1} = sprintf('         OptWFElt=  %d', wfe);
        ln{end+1} = sprintf('       OptMaxItrs=  %d', mx);
        ln{end+1} =         '           OptFEX=  No';
        ln{end+1} =         '      OptSpotSize=  0.0';
        % OptRayGrid= (CALIB's own ray grid) is NOT written unless asked for:
        % any value (21, 31, 41 measured 2026-10-01, model 128) corrupts the
        % heap -- crash at exit, hang, crash in the solve -- while the default
        % grid runs clean.  Reported to CC (BRIEF_dyson5_beat4c.md).
        if isfield(O, 'raygrid') && ~isempty(O.raygrid)
            ln{end+1} = sprintf('       OptRayGrid=  %d', O.raygrid);
        end
        for j = 2:numel(O.fovs)
            dj = O.fovs(j).dir(:);  pj = O.fovs(j).slit(:) + G.src.zsrc_gap*dj;
            ln{end+1} = sprintf('     OptChfRayDir=  %s', F(dj));   %#ok<AGROW>
            ln{end+1} = sprintf('     OptChfRayPos=  %s', F(pj));   %#ok<AGROW>
        end
        ln{end+1} = sprintf('         OptFOVWt=  %s', strtrim(sprintf('%.6g  ', w)));
        if numel(O.wavelens) > 1
            ln{end+1} = sprintf('       ArrWaveLen=  %s', strtrim(sprintf('%.9E  ', O.wavelens(2:end))));
        end
    end
    ln{end+1} = sprintf('             nElt=  %d', nElt);
    for k = 1:nElt
        e = E{k};
        ln{end+1} = '';                                              %#ok<AGROW>
        ln{end+1} = sprintf('             iElt=  %d', k);
        ln{end+1} = sprintf('          EltName=  %s', e.name);
        if link(k) > 0, ln{end+1} = sprintf('             Link=  %d', link(k)); end
        ln{end+1} = sprintf('          Element=  %s', e.element);
        ln{end+1} = sprintf('          Surface=  %s', e.surface);
        ln{end+1} = sprintf('            KrElt=  %.10E', e.Kr);
        ln{end+1} = sprintf('            KcElt=  %.10E', e.Kc);
        if ~isempty(e.asph)
            % the parser reads nAsphCoef_Default = 4 values from this line (a
            % shorter line is an uncaught end-of-file that kills the host) --
            % pad with zeros to four, at most four terms (h^4 .. h^10)
            a4 = zeros(1, 4);  a4(1:numel(e.asph)) = e.asph(1:min(4, numel(e.asph)));
            ln{end+1} = sprintf('         AsphCoef=  %s', sprintf('%.15E  ', a4));
        end
        ln{end+1} = sprintf('           psiElt=  %s', F(e.psi));
        ln{end+1} = sprintf('           VptElt=  %s', F(e.vpt));
        ln{end+1} = sprintf('           RptElt=  %s', F(e.vpt));
        ln{end+1} = sprintf('           IndRef=  %.6E', e.indref);
        if ~isempty(e.glass)
            ln{end+1} = sprintf('         GlassElt=  %s', e.glass);
        end
        ln{end+1} = sprintf('           Extinc=  %.6E', e.extinc);
        if ~isempty(O) && isfield(O, 'var') && ~isempty(O.var)
            jv = find(strcmp({O.var.name}, e.name), 1);
            if ~isempty(jv)
                ln{end+1} = sprintf('           VarElt=  %s', strtrim(sprintf('%d ', O.var(jv).mask)));   %#ok<AGROW>
                if isfield(O.var, 'asph') && ~isempty(O.var(jv).asph)
                    ln{end+1} = sprintf('          OptAsph=  %d  %s', numel(O.var(jv).asph), strtrim(sprintf('%d ', O.var(jv).asph)));   %#ok<AGROW>
                end
            end
        end
        if ~isempty(e.grating)
            ln{end+1} = sprintf('            h1HOE=  %s', F(e.grating.dir));
            ln{end+1} = sprintf('         OrderHOE=  %d', e.grating.m);
            ln{end+1} = sprintf('        RuleWidth=  %.12E', e.grating.d);
        end
        ln{end+1} =         '            nCoat=  0';
        if ~isempty(e.xobs), ln{end+1} = sprintf('             xObs=  %s', F(e.xobs)); end
        ln{end+1} =         '             nObs=  0';
        if isempty(e.ap)
            ln{end+1} =     '           ApType=  None';
        else
            ln{end+1} =     '           ApType=  Circular';
            ln{end+1} = sprintf('            ApVec=  %.12E  %.12E  %.12E', e.ap(1), e.ap(2), e.ap(3));
        end
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
