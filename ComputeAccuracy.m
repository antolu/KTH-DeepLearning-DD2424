function acc = ComputeAccuracy(X, y, W, b)

xSize = size(X);
numberOfSamples = xSize(2);

P = EvaluateClassifier(X, W, b);

acc = 0;

for i=1:numberOfSamples
    k = find(P(:, i) == max(P(:, i)));
    
    if k == y(i)
        acc  = acc + 1;
    end
end

acc = acc / numberOfSamples;

end