function [P, H, BNParams] = EvaluatekLayer(X, NetParams, varargin) 

[batchNormalise, calculateShift] = parse_inputs(varargin);

k = numel(NetParams.W);

% [P2, H2] = Evaluate2Layer(X, NetParams.W, NetParams.b);

mu = cell(1, k);
v = cell(1, k);

H = cell(1, k);
H{1} = X;

if calculateShift
    gammas = cell(1, k-1);
    betas = cell(1, k-1);
else
    gammas = NetParams.gammas;
    betas = NetParams.betas;
end

if not(batchNormalise)
    for i=1:k-1
        S = EvaluateClassifier(H{i}, NetParams.W{i}, NetParams.b{i}); 
        
        if batchNormalise
            
            if calculateShift
                gammas{i} = sqrt(var(S, 0, 2) * (size(S, 2) - 1) / size(S, 2) ); 
                betas{i} = mean(S, 2); 
            end
            
            mu{i} = mean(S, 2);
            v{i} = var(S, 0, 2) * (size(S, 2) - 1) / size(S, 2);
            
            S_hat = BatchNormalise(S, mu{i}, v{i});
            S_tilde = gammas{l} .* S_hat + betas{l};

            S_tilde(S_tilde < 0) = 0;

            H{i+1} = S_tilde;
        else
            S(S < 0) = 0;

            H{i+1} = S;
        end
    end
        
    P = SoftMax(EvaluateClassifier(H{end}, NetParams.W{end}, NetParams.b{end}));
end

H = H(2:end);

BNParams.mu = mu;
BNParams.v = v;

if calculateShift
    BNParams.gammas = gammas;
    BNParams.betas = betas;
end

end

function [S_hat] = BatchNormalise(S, mu, v)

S_hat = (diag(v + eps))^(-1/2) * (S - mu);

end

function [batchNormalise, calculateShift] = parse_inputs(inputs)

% Set defaults
batchNormalise = 0;
calculateShift = 0;

% Go through option pairs
for a = 1:2:numel(inputs)
    switch lower(inputs{a})
        case 'batchnormalise'
            batchNormalise = inputs{a+1};
        case 'calculateshift'
            calculateShift = inputs{a+1};
        otherwise
            error('Input option %s not recognized', inputs{a});
    end
end

return

end