function mimg(data,stat,crop,ndec,cmap)
%MACOS.MIMG  Display a 2D map (OPD, Jacobian column, mask) with statistics.
%   macos.mimg(DATA)
%   macos.mimg(DATA, STAT, CROP, NDEC, CMAP)
%
%   STAT =  0   no statistics (DEFAULT)
%        = +/-1 display RMS and PV
%        = +/-2 display peak
%        = +/-3 display std
%        < 0    also strip the axis tick labels
%   CROP =  0   no cropping (DEFAULT)
%        =  1   crop to the nonzero data
%        =  2   crop symmetrically (crop, then macos.pad back to square)
%   NDEC        decimal places in the statistics label (3)
%   CMAP        colormap name ('jet')
%
%   Library display helper -- lived in examples/sensitivities/ until the
%   2026-08 templates reorganization put every universal helper in the
%   +macos package.
%
%   See also macos.pad, macos.m2v, macos.v2m.


if (nargin<=1), stat=0; end
if (nargin<=2), crop=0; end
if (nargin<=3), ndec=3; end
if (nargin<=4), cmap='jet'; end

% -------------------------
% Reshape if in python array form
% -------------------------
	sizdat = size(data);
	if length(sizdat) == 3
		if (sizdat(1) == 1) && (sizdat(2) == sizdat(3))
		    data = reshape(data, sizdat(2), sizdat(3));
		end
	end

% -------------------------
% Cropping
% -------------------------
if crop~=0
    [ys xs] = find(~~data);
    xmin=min(xs); xmax=max(xs);
    ymin=min(ys); ymax=max(ys);
    data = data(ymin:ymax,xmin:xmax);
    if crop==2
        data = macos.pad(data, length(data));
    end
end

% ----------------------
% Plot
% ----------------------
cla			% cla reset
imagesc(data);
axis image
colorbar
colormap(cmap);

% ------------------------
% Statistics (RMS, PV)
% ------------------------
if abs(stat)==1
    nzd = nonzeros(data);
    wfe = std(nzd);
    pv = max(nzd) - min(nzd);
    xlabel(['RMS = ', num2str(wfe) ', PV = ', num2str(pv, ndec)])
    
elseif abs(stat)==2
    nzd = nonzeros(data);
    peak = max(nzd);
    xlabel(['Peak = ', num2str(peak,ndec)])
    
elseif abs(stat)==3
    nzd = nonzeros(data);
    sdev = std(nzd);
    xlabel(['StDev = ', num2str(sdev,ndec)])
end
if stat < 0
    set(gca, 'xtick', []);
    set(gca, 'ytick', []);
end

