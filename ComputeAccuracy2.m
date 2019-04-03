function acc = ComputeAccuracy2(X, y, W, b)

xSize = size(X);
numberOfSamples = xSize(2);

H = EvaluateClassifier(X, W{1}, b{1}); H(H < 0) = 0;
P = SoftMax(EvaluateClassifier(X, W{2}, b{2}));

acc = 0;

for i=1:numberOfSamples
    k = find(P(:, i) == max(P(:, i)));
    
    if k == y(i)
        acc  = acc + 1;
    end
end

acc = acc / numberOfSamples;

end