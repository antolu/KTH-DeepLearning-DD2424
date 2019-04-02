function S = EvaluateClassifier(X, W, b)

S = W * X + b * ones(1, size(X, 2));

end