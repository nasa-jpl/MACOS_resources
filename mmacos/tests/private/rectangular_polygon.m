function v = rectangular_polygon(wx, wy, dx, dy)
%RECTANGULAR_POLYGON  4-vertex rectangle centered at (dx, dy).
%   Returns 4×2 [x y] vertex array (CW from top-right).
arguments
    wx (1,1) double {mustBePositive}
    wy (1,1) double {mustBePositive}
    dx (1,1) double = 0
    dy (1,1) double = 0
end
hx = wx / 2;
hy = wy / 2;
v = [ hx,  hy;
     -hx,  hy;
     -hx, -hy;
      hx, -hy ] + [dx, dy];
end
