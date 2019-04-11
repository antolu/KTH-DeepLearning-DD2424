function [l, J] = ComputeCost(X, Y, NetParams, lambda)

numberOfSamples = size(X, 2);

[P, H] = EvaluatekLayer(X, NetParams);

l = sum(-log(sum(Y .* P)));

l = l / numberOfSamples;

regularisation = 0;

for i=1:max(size(NetParams.W))
    regularisation = regularisation + sum(sum(NetParams.W{i}));
end

regularisation = lambda * regularisation;

J = l + regularisation;

end