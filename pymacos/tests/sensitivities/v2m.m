
% Utility to vectorize matrix data
%   [mat]=v2m(vec,indx);

	function [mat]=vec2mat(vec,indx);
	m=indx.size(1);
	n=indx.size(2);
	mat = full(sparse(indx.i,indx.j,vec,m,n));
