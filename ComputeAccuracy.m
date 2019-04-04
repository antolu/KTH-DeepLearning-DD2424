function acc = ComputeAccuracy(X, y, W, b)

xSize = size(X);
numberOfSamples = xSize(2);

P = SoftMax(EvaluateClassifier(X, W, b));

V = P==max(P);
matches = V(sub2ind(size(V), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

end