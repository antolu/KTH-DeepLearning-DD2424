function acc = ComputeAccuracy(X, y, NetParams, varargin)

BNParams = parse_inputs(varargin);

xSize = size(X);
numberOfSamples = xSize(2);

[P, BParams] = ForwardPass(X, NetParams, 'BNParams', BNParams);

V = P==max(P);
matches = V(sub2ind(size(V), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

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