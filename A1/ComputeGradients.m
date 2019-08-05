function [gradW, gradb] = ComputeGradients(X, Y, P, W, lambda)
% Computes the gradients of W and b for a single layer perceptron
%   X - The data
%   Y - The targets
%   P - The SoftMax values
%   W - The current weights
%   lambda - L2 regularisation weight
%________________________________________________________________

    batchSize = size(X, 2);

    Gbatch = -(Y - P);

    dLdW = (1 / batchSize) * Gbatch * X';
    dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

    k = 2 * lambda * W;

    gradW = dLdW + k;
    gradb = dLdb;

end