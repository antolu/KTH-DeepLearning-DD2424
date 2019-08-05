function [gradW, gradb] = ComputeGradients(X, Y, P, H, W, lambda)
% Computes the gradients of W and b for a dual layer perceptron
%   X - The data
%   Y - The targets
%   P - The SoftMax values
%   H - The first layer activations
%   W - The current weights
%   lambda - L2 regularisation weight
%________________________________________________________________ 

    batchSize = size(X, 2);

    Gbatch2 = -(Y - P);

    dLdW2 = (1 / batchSize) * Gbatch2 * H';
    dLdb2 = (1 / batchSize) * Gbatch2 * ones(batchSize, 1);

    gradW2 = dLdW2 + 2 * lambda * W{2};
    gradb2 = dLdb2;

    Gbatch1 = W{2}' * Gbatch2;
    Ind = H;
    Ind(Ind>0) = 1;
    Gbatch1 = Gbatch1 .* Ind;

    dLdW1 = (1 / batchSize) * Gbatch1 * X';
    dLdb1 = (1 / batchSize) * Gbatch1 * ones(batchSize, 1);

    gradW1 = dLdW1 + 2 * lambda * W{1};
    gradb1 = dLdb1;

    gradW = {gradW1, gradW2};
    gradb = {gradb1, gradb2};

end