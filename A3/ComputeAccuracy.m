function acc = ComputeAccuracy(X, y, NetParams)

xSize = size(X);
numberOfSamples = xSize(2);

[P, H] = EvaluatekLayer(X, NetParams);

V = P==max(P);
matches = V(sub2ind(size(V), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

end