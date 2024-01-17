%             B^3   B^2   B^1  B^0
polynomial = [0.0, 0.81, -1.8, 1.0];
inverse_roots = roots(polynomial);
r = arrayfun(@(x) 1/x, inverse_roots);
r
alpha = 0.0;
phi = 0.0;
for i = 1:length(r)
    % only if we have complax and conjucate solutions
    if not( real(imag(r(i))) == 0.0 )
        d = imag(r(i));
        c = real(r(i));
        alpha = sqrt(c^2 + d^2);
        phi = atan( d / c );
        alpha
        phi
        break;
    end
end