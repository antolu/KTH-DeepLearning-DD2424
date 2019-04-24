function [P, BNParams] = ForwardPass(X, NetParams, varargin) 

BNParams = parse_inputs(varargin);
batchSize = size(X, 2);

k = numel(NetParams.W);

S = cell(1, k-1);

if NetParams.use_bn
    if BNParams.calculate_mean
        mu = cell(1, k-1);
        v = cell(1, k-1);
    else
        mu = BNParams.mu;
        v = BNParams.v;
    end
    S_hat = cell(1, k-1);
    S_tilde = cell(1, k-1);
    
    gammas = NetParams.gammas;
    betas = NetParams.betas;
end

H = cell(1, k);
H{1} = X;

for i=1:k-1
    S{i} = EvaluateClassifier(H{i}, NetParams.W{i}, NetParams.b{i}); 

    if NetParams.use_bn

        if BNParams.calculate_mean
            mu{i} = mean(S{i}, 2);
            v{i} = sum((S{i} - mu{i}).^2, 2) / batchSize;
%             v{i} = var(S{i}, 0, 2) * (size(S{i}, 2) - 1) / size(S{i}, 2);
        end

        S_hat{i} = BatchNormalise(S{i}, mu{i}, v{i});
        S_tilde{i} = gammas{i} .* S_hat{i} + betas{i};

        S_tilde{i}(S_tilde{i} < 0) = 0;

        H{i+1} = S_tilde{i};
    else
        S{i}(S{i} < 0) = 0;

        H{i+1} = S{i};
    end
end

P = SoftMax(EvaluateClassifier(H{end}, NetParams.W{end}, NetParams.b{end}));

H = H(2:end);
BNParams.X = H;

if NetParams.use_bn
    BNParams.mu = mu;
    BNParams.v = v;

    BNParams.S = S;
    BNParams.S_hat = S_hat;
    BNParams.S_tilde = S_tilde;
end


end

function [S_hat] = BatchNormalise(S, mu, v)

S_hat = (diag(v + eps))^(-1/2) * (S - mu);

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