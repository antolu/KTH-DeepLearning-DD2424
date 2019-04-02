function [l, J] = ComputeCost(X, Y, W, b, lambda)

xSize = size(X);
numberOfSamples = xSize(2);

S = W * X + b * ones(1, size(X, 2));

l = 0;

for i=1:numberOfSamples
    pos = find(Y(:, i)==1);
    sy = S(pos, i);
    s = S(:, i) - sy;
    s = s + 1;
    s(s<0) = 0;
    k = sum(s) - 1;
    l = l + k;
end

l = l / numberOfSamples;

regularisation = lambda * sum(sum(W.^2));

J = l + regularisation;

end