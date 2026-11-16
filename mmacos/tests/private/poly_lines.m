function bounds = poly_lines(vertices)
%POLY_LINES  Linear equations Ax+By+C=0 for each polygon edge.
%   Returns N×3 [A B C] coefficients (N = number of vertices).
%   Edge i connects vertices(i, :) ↔ vertices(i-1, :) (wraps at i=1).
arguments
    vertices (:,2) double
end
n = size(vertices, 1);
bounds = zeros(n, 3);
for i = 1:n
    v1 = vertices(i, :);
    if i == 1
        v2 = vertices(end, :);
    else
        v2 = vertices(i-1, :);
    end
    bounds(i, :) = abc(v1, v2);
end
end

function out = abc(v1, v2)
tol = 3e-16;
if abs(v2(2) - v1(2)) <= tol
    out = [0, 1, -v1(2)];
elseif abs(v2(1) - v1(1)) <= tol
    out = [1, 0, -v1(1)];
else
    m = (v2(2) - v1(2)) / (v2(1) - v1(1));
    out = [-m, 1, v1(1)*m - v1(2)];
end
end
