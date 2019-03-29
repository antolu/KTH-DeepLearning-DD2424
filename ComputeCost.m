function [l, J] = ComputeCost(X, Y, W, b, lambda)

xSize = size(X);
numberOfSamples = xSize(2);

P = EvaluateClassifier(X, W, b);

l = -log(Y' * P);
l = trace(l);

l = l / numberOfSamples;

regularisation = lambda * sum(sum(W.^2));

J = l + regularisation;

end