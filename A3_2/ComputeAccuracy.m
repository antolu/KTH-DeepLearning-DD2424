function acc = ComputeAccuracy(X, y, NetParams, varargin)

k = numel(NetParams.W);
[mu, v] = parse_inputs(varargin, k);

N = size(X, 2);

[P, H, ~, ~, ~, ~] = EvaluatekLayer(X, NetParams, 'mean', mu, 'variance', v);

V = P==max(P);
matches = V(sub2ind(size(V), y', 1:size(y)));

acc = sum(matches);

acc = acc / N;

end

function [mu, v] = parse_inputs(inputs, l)

    % Set defaults
    mu = cell(1, l); v = cell(1, l);
    for i=1:l
        mu{i} = 0;
        v{i} = 1;
    end

    % Go through option pairs
    for a = 1:2:numel(inputs)
        switch lower(inputs{a})
            case 'mean'
                mu = inputs{a+1};
            case 'variance'
                v = inputs{a+1};
            otherwise
                error('Input option %s not recognized', inputs{a});
        end
    end
    
    return
    
end