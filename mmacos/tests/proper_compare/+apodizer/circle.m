function fn = circle(r0)
%CIRCLE  Binary disk predicate of radius r0 (metres) centred at origin.
arguments
    r0 (1,1) double {mustBeNonnegative}
end
fn = @(x, y) (x.*x + y.*y) <= (r0 * r0);
end
