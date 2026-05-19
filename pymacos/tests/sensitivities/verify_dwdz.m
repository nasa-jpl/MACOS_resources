
%   verify_dwz.m -- Replays dwdz_(Rx name).mat
format compact

Rx_name = 'e5hex1'

load(['results/dwdz_' Rx_name]);

[nray,nz] = size(dwdz);

nzelt = n_zcoef - zmode_start + 1;

nelt = nz/nzelt;

% Loop to display sensitivities for each element 

ii=0;
ifig=0;
ncol=6;
nrow=double(ceil(nzelt/ncol));
for ielt=1:nelt
    kk=0;
    ifig=ifig+1;
    figure(ifig), clf

    for iz=1:nzelt
        kk=kk+1;
        ii=ii+1;
        opd=pad(v2m(dwdz(:,ii),indx),nGridPts);
        subplot(nrow,ncol,kk)
        mimg(opd,-1)
        title(channel_names{ii})
    end
    drawnow
    print(ifig,'-dpng',['dwdz,Elt' num2str(ielt)])
end



