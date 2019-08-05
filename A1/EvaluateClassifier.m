function P = EvaluateClassifier(X, W, b)
% Evaluates the single layer perceptron
%   X - The data
%   W - The layer weights
%   b - The bias weights
%______________________________________

    P = SoftMax(W * X + b * ones(1, size(X, 2)));

end



function P = SoftMax(s)
% Computes the SoftMax values of the targets
%   s - targets
%________________________________________________________

    P = exp(s) ./ (sum(exp(s)));

end
