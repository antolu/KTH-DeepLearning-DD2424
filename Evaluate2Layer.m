function [P, H] = Evaluate2Layer(X, W, b) 

H = EvaluateClassifier(X, W{1}, b{1}); H(H < 0) = 0;
P = SoftMax(EvaluateClassifier(H, W{2}, b{2}));

end