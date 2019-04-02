function [gradW, gradb] = ComputeGradientsSVM(X, Y, b, W, lambda)

xSize = size(X);
batchSize = xSize(2);

S = W * X + b * ones(1, size(X, 2));

Sy = S .* Y;

Gbatch = heaviside(S - sum(Sy) + 1);

for i=1:batchSize
    pos = find(Y(:, i) == 1);
    Gbatch(pos, i) = -sum(Gbatch(:, i)) + Gbatch(pos, i);
end

dLdW = (1 / batchSize) * Gbatch * X';
dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

gradW = dLdW + 2 * lambda * W;
gradb = dLdb;

end