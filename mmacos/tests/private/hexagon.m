function v = hexagon(s, dx, dy)
%HEXAGON  Top-flat hexagon of side length s centered at (dx, dy).
%   Returns 6×2 [x y] vertex array.
arguments
    s  (1,1) double {mustBePositive}
    dx (1,1) double = 0
    dy (1,1) double = 0
end
h = sqrt(3)/2 * s;
v = [ +s,    0;
      +s/2, +h;
      -s/2, +h;
      -s,    0;
      -s/2, -h;
      +s/2, -h ] + [dx, dy];
end
