
% Utility to vectorize matrix data
%   [vec,jndx]=m2v(mat,indx);
%   [vec,indx]=m2v(mat); -- Use this first call to set up indx 
%   [vec]=m2v(mat,indx); -- Use this for all subsequent calls

	function [vec,jndx]=m2v(mat,indx);
	if nargin==1,
	   [i,j,vec] = find(mat);		% Vectorize mat
	   jndx.i=i;
	   jndx.j=j;
	   jndx.size=size(mat);
	elseif nargin==2,
	   jndx=indx;
	   k=sub2ind(indx.size,indx.i,indx.j);
	   vec=mat(k);
	end;
	   
	   
