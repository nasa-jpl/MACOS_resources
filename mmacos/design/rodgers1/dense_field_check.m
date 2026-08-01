function D = dense_field_check(opts)
%DENSE_FIELD_CHECK  Is the residual on the AVERAGE a field-sampling artifact?
%
%   The strict metric reproduces Rodgers' reported MAX to 0.3% or better on
%   all three of his designs (PACKET.md Addendum 10, re-run natively after
%   the ColSource pupil fix).  The AVERAGE still sits high -- S3 reads
%   50.19 nm against his 46.38, +8.2%.  This asks the cheapest question
%   that could explain it: are we averaging over the same field set he is?
%
%   We score on the 15 XAN/YAN points his .seq files define.  That set is a
%   HALF box on a QUINCUNX -- 3 y-points at XAN = 0, 0.05, 0.1 and 2 at
%   XAN = 0.025, 0.075, plus 2 interior y-points on the XAN = 0 column
%   (rodgers_seq.m).  It is a solve sampling, chosen to constrain an
%   optimizer, and it is weighted toward the box EDGE.  CODE V's field-map
%   statistics are computed on CODE V's own map grid.  If WFE grows toward
%   the edge -- which on a biased box it does -- then an edge-weighted set
%   reports a HIGHER average for the same design, at the same max.
%
%   So: re-score the SAME committed decks with the SAME metric and the SAME
%   rung, changing ONLY the field sampling, and watch the average.  The max
%   is the control: it must not move (a denser grid can only find the same
%   or a slightly worse worst point).
%
%   Grids compared:
%     seq       his 15 quincunx points (the incumbent)
%     halfNxN   uniform (N x N) over the HALF box, XAN in [0,0.1]
%     fullNxN   uniform (N x N) over the FULL box, XAN in [-0.1,0.1]
%
%   The full box is the honest comparator if his map is over the whole
%   field; the design is symmetric about the y-z plane, so the full-box
%   average equals the half-box average up to sampling, and disagreement
%   between the two is itself a sampling diagnostic.
%
%   Name-value:
%     'decks' cellstr (default the three committed .seq-truth decks)
%     'N'     grid density, default 9 (-> 81 points per grid)
%     'save'  write rodgers1_dense_field.mat (default true)
%
%   NOTE this changes only the SCORING field set.  No deck is re-solved.

    arguments
        opts.decks (1,:) cell   = {}
        opts.N     (1,1) double = 9
        opts.save  (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);

    P = rodgers_common('seq');
    decks = opts.decks;
    if isempty(decks)
        decks = { fullfile(here,'rodgers1_seq_rodgersS2.in')
                  fullfile(here,'rodgers1_seq_rodgersS3.in')
                  fullfile(here,'rodgers1_seq_rodgersS4.in') };
    end
    tags = {'rodgersS2','rodgersS3','rodgersS4'};
    gts  = { P.gt.s2_box, P.gt.s3_box, P.gt.s4_box };

    N = opts.N;
    h = P.seq.fov_half_deg;                     % 0.1 deg
    u = linspace(0, h, N);   v = linspace(-h, h, N);
    [UX,UY] = meshgrid(u,v);  halfG = deg2rad([UX(:), UY(:)]);
    w = linspace(-h, h, N);
    [WX,WY] = meshgrid(w,v);  fullG = deg2rad([WX(:), WY(:)]);

    grids = { 'seq(15)',            P.seq.Frel
              sprintf('half%dx%d',N,N), halfG
              sprintf('full%dx%d',N,N), fullG };

    macos.init(P.model_size);
    D = struct('N',N,'grids',{grids(:,1)},'stage',struct([]));

    fprintf('\n############ FIELD-SAMPLING CHECK (metric and decks unchanged) ############\n');
    for c = 1:numel(decks)
        g = gts{c}*1e3;
        fprintf('\n  === %s   (CODE V reported: max %.2f  avg %.2f nm) ===\n', tags{c}, g(2), g(3));
        fprintf('  %-12s %6s | %8s %8s | %8s %8s\n', ...
                'field set','K','max nm','x his','avg nm','x his');
        D(c).tag = tags{c};  D(c).gt_nm = g;
        for k = 1:size(grids,1)
            L = ladder_(decks{c}, grids{k,2});
            a = L(:,4)*1e9;                      % bestfoc + LS tip/tilt rung
            a = a(isfinite(a));
            fprintf('  %-12s %6d | %8.2f %8.3f | %8.2f %8.3f\n', ...
                    grids{k,1}, numel(a), max(a), max(a)/g(2), mean(a), mean(a)/g(3));
            D(c).set(k).name = grids{k,1};
            D(c).set(k).K    = numel(a);
            D(c).set(k).max_nm = max(a);
            D(c).set(k).avg_nm = mean(a);
        end
    end

    fprintf(['\n  READING.  If the average falls toward his on the uniform grids while\n' ...
             '  the max holds, the residual on the average was the SAMPLING, not the\n' ...
             '  optics.  If it does not move, the average is a real disagreement and\n' ...
             '  the next suspect is his map''s own weighting.\n']);

    if opts.save
        out = fullfile(here,'rodgers1_dense_field.mat');
        save(out,'D');  fprintf('\n  saved %s\n', out);
    end
end

% =====================================================================
function L = ladder_(deck, Frel)
%LADDER_  The 4-rung reference ladder per field.  Lifted from PUPIL_AUDIT's
%   local ladder_both_ with the pupil-masking leg removed -- the engine now
%   traces the declared pupil natively, so there is nothing to mask.
    txt = strip_ap_(fileread(deck));
    [cdir0,cpos0,apst] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    [Vd, Nd] = deck_fp_(txt);  Nd = Nd/norm(Nd);
    K = size(Frel,1);  L = nan(K,4);
    tmp = [tempname '.in'];  cu = onCleanup(@() del_(tmp)); %#ok<NASGU>
    for k = 1:K
        [ri,nE] = trace_field_(txt,tmp,apst,stand, bx0+Frel(k,1), by0+Frel(k,2));
        p1 = ri.pos(:,1);  d1 = ri.dir(:,1);
        rp = trace_field_(txt,tmp,apst,stand, bx0+Frel(k,1)+1e-5, by0+Frel(k,2));
        X  = fex_cross_(p1,d1,rp.pos(:,1),rp.dir(:,1));
        ri = trace_field_(txt,tmp,apst,stand, bx0+Frel(k,1), by0+Frel(k,2));
        ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;  % engine OPD skips the chief
        if nnz(ok) < 10, continue; end
        L(k,:) = rungs_(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), p1,d1,Vd,Nd,X);
    end
end

function v = rungs_(Pp,Dd,Ll,p1,d1,Vd,Nd,X)
    f = strict_refs(Pp,Dd,Ll,p1,d1,Vd,Nd,X);
    v = nan(1,4);
    v(1) = f.wfe_chief;
    v(2) = f.wfe_centroid;
    e3 = d1(:)/norm(d1(:));
    e1 = [1;0;0] - e3*dot([1;0;0],e3);
    if norm(e1) < 1e-8, e1 = [0;1;0] - e3*dot([0;1;0],e3); end
    e1 = e1/norm(e1);  e2 = cross(e3,e1);
    tp = (e3.'*(X(:)-Pp))./(e3.'*Dd);  Q = Pp + Dd.*tp;
    px = (e1.'*(Q-X(:))).';  py = (e2.'*(Q-X(:))).';
    c0 = f.c_centroid;  R0 = f.R_centroid;
    ff = @(u) std(strict_sphere_opl(Pp,Dd,Ll, c0+e3*u, R0+u));
    u  = fminbnd(ff, -0.05, 0.05);
    v(3) = ff(u);
    W  = strict_sphere_opl(Pp,Dd,Ll, c0+e3*u, R0+u);
    Am = [ones(numel(px),1), px(:), py(:)];
    v(4) = std(W(:) - Am*(Am\W(:)));
end

function s = strip_ap_(txt),  s = regexprep(txt,'(ApType=\s*)\S+','$1None');  end
function [cdir,cpos,apst] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');  cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
end
function [V,N] = deck_fp_(txt)
    Vs = grabAll3_(txt,'VptElt');  Ps = grabAll3_(txt,'psiElt');
    V = Vs(:,end);  N = Ps(:,end);
end
function v = grab3_(txt,key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end
function M = grabAll3_(txt,key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens');
    M = zeros(3,numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end
function [ri,nE] = trace_field_(txt,tmp,apst,stand,bx,by)
    cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt,'(ChfRayDir=\s*)[^\n]*',['$1' v3_(cdir)]);
    s = regexprep(s,  '(ChfRayPos=\s*)[^\n]*',['$1' v3_(cpos)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
    macos.load_rx(tmp);
    nE = macos.num_elt();
    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
end
function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end
function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1-p2;  b = dot(d1,d2);  den = 1-b^2;
    if abs(den) < 1e-14, X = p1; return; end
    X = 0.5*((p1 + d1*((b*dot(d2,w0)-dot(d1,w0))/den)) + ...
             (p2 + d2*((dot(d2,w0)-b*dot(d1,w0))/den)));
end
function del_(p),  if exist(p,'file'), delete(p); end,  end
