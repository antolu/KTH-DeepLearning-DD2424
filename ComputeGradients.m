function [gradW, gradb] = ComputeGradients(X, Y, P, W, lambda)

batchSize = size(X, 2);

Gbatch = -(Y - P);

dLdW = (1 / batchSize) * Gbatch * X';
dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

k = 2 * lambda * W;

gradW = dLdW + k;
gradb = dLdb;

end