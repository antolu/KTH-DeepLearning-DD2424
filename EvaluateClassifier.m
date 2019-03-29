function P = EvaluateClassifier(X, W, b)

SoftMax = @(s) exp(s) ./ (sum(exp(s)));

s = W * X + b * ones(1, size(X, 2));

P = SoftMax(s);

end