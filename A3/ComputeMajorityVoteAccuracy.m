function acc = ComputeMajorityVoteAccuracy(X, y, NetParams, varargin)

BNParams = parse_inputs(varargin);
numberOfSamples = size(X, 2);

matches = zeros(1, numberOfSamples);
votes = zeros(max(y), size(X, 2));

for i=1:size(NetParams.W, 1)
    if not(isempty(NetParams.W{i}))
        Params = NetParams;
        Params.W = NetParams.Wstar(i, :);
        Params.b = NetParams.bstar(i, :);
        
        [P, BParams] = ForwardPass(X, Params, 'BNParams', BNParams);

        V = P==max(P);
        votes = votes + V;
    end
end

finalVotes = votes==max(votes);

matches = finalVotes(sub2ind(size(finalVotes), y', 1:size(y)));

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