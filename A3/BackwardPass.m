function Gradients = BackwardPass(X, Y, P, H, NetParams, lambda, varargin)

BNParams = parse_inputs(varargin);

batchSize = size(X, 2);
k = numel(NetParams.W);
% H = {{X}; H}; H = cat(2, H{:});

gradW = cell(1, max(k));
gradb = cell(1, max(k));

Gbatch = -(Y - P);

dLdW = (1 / batchSize) * Gbatch * H{end}';
dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

gradW{end} = dLdW + 2 * lambda * NetParams.W{end};
gradb{end} = dLdb;

Gbatch = NetParams.W{end}' * Gbatch;
Ind = H{end}; Ind(Ind>0) = 1;
Gbatch = Gbatch .* Ind;

for l=k-1:-1:1
    
    if NetParams.use_bn
        dJdgamma{l} = (1 / batchSize) * (Gbatch .* BNParams.S_hat{l}) * ones(batchSize, 1);
        dJdbeta{l} = (1 / batchSize) * Gbatch * ones(batchSize, 1);

        Gbatch = Gbatch .* (NetParams.gammas{l} * ones(1, batchSize));
        
        Gbatch = BatchNormBackPass(Gbatch, BNParams.S{l}, BNParams.mu{l}, BNParams.v{l});
    end
    
    if l == 1
        gradW{1} = (1 / batchSize) * Gbatch * X' + 2 * lambda * NetParams.W{1};
        gradb{1} = (1 / batchSize) * Gbatch * ones(batchSize, 1);
    else
        gradW{l} = (1 / batchSize) * Gbatch * H{l-1}' + 2 * lambda * NetParams.W{l};
        gradb{l} = (1 / batchSize) * Gbatch * ones(batchSize, 1);
    end
    
    if l>1
        Gbatch = NetParams.W{l}' * Gbatch;
        Ind = H{l-1}; Ind(Ind>0) = 1;
        Gbatch = Gbatch .* Ind;
    else
        break
    end
end

if NetParams.use_bn
    Gradients.gamma= dJdgamma;
    Gradients.beta = dJdbeta;
end

Gradients.W = gradW;
Gradients.b = gradb;

end



function [Gbatch] = BatchNormBackPass(Gbatch, S, mu, v)

n = size(Gbatch, 2);

sigma1 = ((v+eps).^(-0.5));
sigma2 = ((v+eps).^(-1.5));

G1 = Gbatch .*(sigma1 * ones(1, n));
G2 = Gbatch .*(sigma2 * ones(1, n));

D = S - mu * ones(1, n);

c = (G2 .* D) * ones(n, 1);

Gbatch = G1 - (1 / n) * G1 * ones(n, 1) - (1 / n) * D .* (c * ones(1, n));

end


function [BNParams] = parse_inputs(inputs)

% Set defaults
BNParams.use_bn = 0;

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