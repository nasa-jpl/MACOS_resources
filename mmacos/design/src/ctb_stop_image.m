function s = ctb_stop_image(R, ring_as)
%CTB_STOP_IMAGE  Where the telescope's stop images after the CTB's OAP1.
%   S = CTB_STOP_IMAGE(R) traces the spliced train R.full_in (from
%   CTB_FOLD_BUILD) for the centre field and an 8-point ring of radius
%   RING_AS arcsec (default 3) and returns the distance (m) after OAP1's
%   pole at which the ring fields' chief rays cross the centre field's
%   chief: the image of the telescope's stop, where DM1 belongs (a DM away
%   from it sees the beam walk across it with field; the mmacos manual's
%   chapter 2, "case C").  The median over the ring's crossings.
%
%   See also CTB_FOLD_BUILD, CTB_FOLD_MEASURE.
    if nargin < 2, ring_as = 3; end
    txt = fileread(R.full_in);
    dB = str2double(regexp(txt, 'ChfRayDir=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once')).';
    pB = str2double(regexp(txt, 'ChfRayPos=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once')).';
    dB = dB/norm(dB);  L0 = norm(pB);
    F = [0 0; macos.design.field_ring(ring_as/60, 'units', 'arcmin', 'n', 8, 'inner', 0)];
    p = zeros(3, size(F,1));  q = p;
    macos.init(R.K.model);
    for k = 1:size(F,1)
        d = dB + [F(k,1); F(k,2); 0];  d = d/norm(d);
        t2 = regexprep(txt, 'ChfRayDir=[^\n]*', sprintf('ChfRayDir=  %.16E  %.16E  %.16E', d));
        t2 = regexprep(t2, 'ChfRayPos=[^\n]*', sprintf('ChfRayPos=  %.16E  %.16E  %.16E', -L0*d));
        tmp = [tempname '.in'];  fid = fopen(tmp, 'w');  fwrite(fid, t2);  fclose(fid);
        macos.load_rx(tmp);  delete(tmp);
        ri = macos.get_ray_info(macos.trace(R.ix.OAP1).nRays);
        p(:,k) = ri.pos(:,1);  q(:,k) = ri.dir(:,1)/norm(ri.dir(:,1));
    end
    sx = zeros(1, size(F,1) - 1);
    for k = 2:size(F,1)
        w0 = p(:,1) - p(:,k);  b = q(:,1).'*q(:,k);
        sx(k-1) = (b*(q(:,k).'*w0) - q(:,1).'*w0) / (1 - b^2);
    end
    s = median(sx);
end
