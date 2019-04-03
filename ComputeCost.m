function [l, J] = ComputeCost(X, Y, W, b, lambda)

xSize = size(X);
numberOfSamples = xSize(2);

P = EvaluateClassifier(X, W, b);

% l = -log(Y' * P);
l = sum(-log(sum(Y .* P)));

% l = 0;
% for i=1:numberOfSamples
%    k = -log(Y(:, i)' * P(:, i));
%    l = l + k;
% end
% l = trace(l);

l = l / numberOfSamples;

regularisation = lambda * sum(sum(W.^2));

J = l + regularisation;

end