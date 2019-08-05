function [P, H] = EvaluateClassifier(X, W, b) 
% Evaluates the dual layer perceptron
%   X - The data
%   W - The layer weights
%   b - The bias weights
%______________________________________

H = EvaluateLayer(X, W{1}, b{1}); H(H < 0) = 0;
P = SoftMax(EvaluateLayer(H, W{2}, b{2}));

end



function S = EvaluateLayer(X, W, b)
% Evaluates a single layer perceptron
%   X - The data
%   W - The layer weights
%   b - The bias weights
%______________________________________

    S = W * X + b * ones(1, size(X, 2));

end



function P = SoftMax(s)
% Computes the SoftMax values of the targets
%   s - targets
%________________________________________________________

    P = exp(s) ./ (sum(exp(s)));

end