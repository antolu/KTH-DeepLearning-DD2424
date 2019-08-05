function [l, J] = ComputeCost(X, Y, W, b, lambda)
% Computes the cross-entropy loss for the classifier
%   X - The data
%   Y - The one-hot labels
%   W - The network weights
%   b - The network bias weights
%   lambda - L2 regularisation weight
   
    numberOfSamples = size(X, 2);

    P = EvaluateClassifier(X, W, b);

    k = sum(Y .* P);
    l = sum(-log(k));

    l = l / numberOfSamples;

    regularisation = lambda * sum(sum(W.^2));

    J = l + regularisation;

end
