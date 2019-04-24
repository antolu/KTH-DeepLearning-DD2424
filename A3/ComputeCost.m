function [l, J] = ComputeCost(X, Y, NetParams, lambda, varargin)

BNParams = parse_inputs(varargin);

numberOfSamples = size(X, 2);

[P, BParams] = ForwardPass(X, NetParams, 'BNParams', BNParams);

l = sum(-log(sum(Y .* P)));

l = l / numberOfSamples;

regularisation = 0;

for i=1:max(size(NetParams.W))
    regularisation = regularisation + sum(sum(NetParams.W{i}.^2));
end

regularisation = lambda * regularisation;

J = l + regularisation;

end

function [BNParams] = parse_inputs(inputs)

% Set defaults
BNParams.calculate_mean = 1;

% Go through option pairs
for a = 1:2:numel(inputs)
    switch lower(inputs{a})
        case 'bnparams'
            BNParams = inputs{a+1};
        otherwise
            error('Input option %s not recognized', inputs{a});
    end
end

return

end