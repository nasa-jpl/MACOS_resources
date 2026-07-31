function D = seq_detector(decks)
%SEQ_DETECTOR  Score his designs against HIS OWN detector, taken from the .seq.
%
%   D = SEQ_DETECTOR()
%
%   The band in section 2 of RUN_SEQ scores his optics against a detector WE
%   fit (align_focal_plane: per-field best focus, plane through the foci).
%   That is not his procedure: in CODE V the image DEFOCUS (THC 0) and image
%   TILT (ADC 0) are OPTIMISATION VARIABLES in every stage, solved against his
%   merit -- and the .seq now states the values he reached.
%
%   So the detector no longer has to be inferred.  This scores the same three
%   designs with the detector placed at his stated geometry:
%
%       station : the PIM paraxial image distance |s3f| from M3, plus that
%                 stage's image defocus
%       normal  : tilted by that stage's ADE about x, in OUR frame (i.e. with
%                 the decoded sign flip -- three independent witnesses, see
%                 PACKET.md Addendum 8.2)
%
%   HONEST LIMIT, stated rather than resolved.  The ADE sign is decoded.  The
%   DEFOCUS sign is NOT: CODE V writes that datum on the SI row, whose
%   thickness would ordinarily be the gap AFTER the image, and this leg's
%   thicknesses run negative.  Both signs are therefore scored and BOTH are
%   reported; the spread between them is the honest uncertainty this one
%   unresolved convention leaves.  Nothing is selected by score.
%
%   Reads the artifact decks RUN_SEQ section 2 wrote (his optics, our fitted
%   FPA) and overrides only the detector, which is exact -- rays are straight
%   after the last surface, so the sphere construction does not care where the
%   deck's own FocalPlane sits (same argument HIS_DESIGNS already relies on).

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common('seq');
    S = P.seq;  lam_nm = P.lambda_m*1e9;
    Frel = S.Frel;

    if nargin < 1 || isempty(decks)
        decks = { fullfile(here,'rodgers1_seq_rodgersS2.in'), 2, P.gt.s2_box
                  fullfile(here,'rodgers1_seq_rodgersS3.in'), 3, P.gt.s3_box
                  fullfile(here,'rodgers1_seq_rodgersS4.in'), 4, P.gt.s4_box };
    end

    banner('HIS DESIGNS SCORED AGAINST HIS OWN .seq DETECTOR');
    fprintf(['  detector station = |s3f| %+.6f mm from M3, plus that stage''s\n' ...
             '  image defocus; normal tilted by that stage''s ADE (decoded sign).\n'], ...
            abs(S.s3f_mm));

    D = struct('stage',[],'variant',{},'max',[],'avg',[],'ratio_max',[],'ratio_avg',[]);
    n = 0;
    for k = 1:size(decks,1)
        deck = decks{k,1};  st = decks{k,2};  gt = decks{k,3};
        if ~isfile(deck)
            fprintf('  (missing %s -- run run_seq(''sections'',2) first)\n', deck);
            continue;
        end
        % geometry of the deck we are scoring
        macos.init(P.model_size);
        txt = regexprep(fileread(deck), '(ApType=\s*)\S+', '$1None');
        tmp = [tempname '.in'];
        fid = fopen(tmp,'w'); fprintf(fid,'%s',txt); fclose(fid);
        macos.load_rx(tmp);
        nE  = macos.num_elt();
        vM3 = macos.get_elt_vpt(3);   vM3 = vM3(:);
        vFP = macos.get_elt_vpt(nE);  vFP = vFP(:);
        % The station must be measured ALONG THE AXIS, not along the chief.
        % CODE V's "recenter" dummy sits on the axis at M3 and the PIM image
        % distance is an axial thickness; at the +0.5 deg bias the chief
        % arrives ~9.7 deg off axis, so stepping |s3f| along the CHIEF puts the
        % plane ~72 mm off in z -- 6.5 um of defocus at f/20, which is what a
        % first cut of this function measured (7400 nm) before the fix.
        u = [0; 0; sign(vFP(3) - vM3(3))];        % M3 -> image, ON AXIS
        delete(tmp);

        ade = S.img_ADE_deg(st);
        if isnan(ade), ade = 0; end
        al  = -deg2rad(ade);                       % decoded: our alpha = -ADE
        psi = [0; sin(al); -cos(al)];
        % (the train is coaxial about z with the beam running -z into the
        %  image, so psi = [0, sin(alpha), -cos(alpha)] is the alpha-tilted
        %  detector normal in the SAME convention rigid_of reports.)

        banner('STAGE %d -- %s', st, deck);
        fprintf('  .seq image ADE %+10.6f deg -> our alpha %+10.6f deg\n', ade, -ade);
        fprintf('  .seq image defocus %+10.7f mm (BOTH signs scored)\n', S.img_defocus_mm(st));

        for sgn = [+1 -1]
            dz  = sgn * S.img_defocus_mm(st) * 1e-3;
            Vpt = vM3 + u * (abs(S.s3f_mm)*1e-3 + dz);
            s = strict_wfe_deck(deck, Frel, 'detector', ...
                                struct('Vpt',Vpt.','psi',psi.'));
            w = s.wfe_m(isfinite(s.wfe_m))*1e9;
            n = n + 1;
            D(n).stage = st;
            D(n).variant = sprintf('%+d x defocus', sgn);
            D(n).max = max(w);  D(n).avg = mean(w);
            D(n).ratio_max = max(w)/(gt(2)*lam_nm);
            D(n).ratio_avg = mean(w)/(gt(3)*lam_nm);
            fprintf(['    defocus %+d : STRICT max %9.3f avg %9.3f nm  ' ...
                     '(%d/%d fields)   ratio %5.3fx / %5.3fx\n'], ...
                    sgn, max(w), mean(w), numel(w), numel(s.wfe_m), ...
                    D(n).ratio_max, D(n).ratio_avg);
        end
        fprintf('    Rodgers reported          %9.3f     %9.3f nm\n', ...
                gt(2)*lam_nm, gt(3)*lam_nm);
    end

    banner('HIS-DETECTOR SUMMARY  (nm @ %g nm, his %d-point half box)', lam_nm, size(Frel,1));
    fprintf('  stage  detector-defocus   strict max/avg      ratio max/avg\n');
    for i = 1:numel(D)
        fprintf('   S%d     %-16s %8.1f/%-8.1f  %6.3fx / %6.3fx\n', ...
                D(i).stage, D(i).variant, D(i).max, D(i).avg, ...
                D(i).ratio_max, D(i).ratio_avg);
    end
    save(fullfile(here,'rodgers1_seq_detector.mat'),'D');
    fprintf('\nsaved rodgers1_seq_detector.mat\n');
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
