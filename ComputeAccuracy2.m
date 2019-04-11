function acc = ComputeAccuracy2(X, y, W, b)

xSize = size(X);
numberOfSamples = xSize(2);

[P, H] = Evaluate2Layer(X, W, b);

V = P==max(P);
matches = V(sub2ind(size(V), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

end