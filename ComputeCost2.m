function [l, J] = ComputeCost2(X, Y, W, b, lambda)

numberOfSamples = size(X, 2);

H = EvaluateClassifier(X, W{1}, b{1}); H(H < 0) = 0;
P = SoftMax(EvaluateClassifier(H, W{2}, b{2}));

l = sum(-log(sum(Y .* P)));

l = l / numberOfSamples;

regularisation = lambda * sum(sum(W{1}.^2)) + lambda * sum(sum(W{2}.^2));

J = l + regularisation;

end