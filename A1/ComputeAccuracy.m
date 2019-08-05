function acc = ComputeAccuracy(X, y, W, b)
% Computes the accuracy of the single layer perceptron
%   X - The data
%   y - The labels
%   W - Layer weights
%   b - Bias weights
    
    numberOfSamples = size(X, 2);

    P = EvaluateClassifier(X, W, b);

    V = P==max(P);
    matches = V(sub2ind(size(V), y', 1:size(y)));

    acc = sum(matches);

    acc = acc / numberOfSamples;

end