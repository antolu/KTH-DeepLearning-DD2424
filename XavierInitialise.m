function [W, b] = XavierInitialise(dimensions)
% dimensions should be in format [D, M, C]

D = dimensions(1);
M = dimensions(2);
C = dimensions(3);

W = {1/sqrt(D) * randn(M, D), 1/sqrt(M) * randn(C, M)};
b = {zeros(M, 1), zeros(C, 1)};

end