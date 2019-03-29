function [gradW, gradb] = ComputeGradients(X, Y, P, W, lambda)

xSize = size(X);
batchSize = xSize(2);

Gbatch = -(Y - P);

dLdW = (1 / batchSize) * Gbatch * X';
dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

gradW = dLdW + 2 * lambda * W;
gradb = dLdb;

end