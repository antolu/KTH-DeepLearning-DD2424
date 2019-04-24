function Grads = ComputeGradients(X, Y, P, H, NetParams, varargin)
%ComputeGradients - Computes gradients (backward pass)
%
% Syntax: Grads = ComputeGradients(X, Y, P, H, NetParams)
%
% Computes backward pass

k = numel(NetParams.W);
N = size(X, 2);

[mu, v, S, S_hat] = parse_inputs(varargin, k);

Grads.W = cell(1, k);
Grads.b = cell(1, k);

G = -(Y - P);

dJdW = (1 / N) * G * H{k-1}' + 2 * NetParams.lambda * NetParams.W{k}; Grads.W{k} = dJdW;
dJdb = (1 / N) * G * ones(N, 1); Grads.b{k} = dJdb;

G = NetParams.W{k}' * G;
G = G .* (H{k-1} > 0);

for l=k-1:-1:1
    dJdGamma = (1 / N) * (G .* S_hat{l}) * ones(N, 1);
    dJdBeta = (1 / N) * G * ones(N, 1);

    G = G .* (NetParams.gamma{l} * ones(1, N));
    if NetParams.use_bn
        G = BackNormBackPass(G, S{l}, mu{l}, v{l});
    end

    if l > 1
        dJdW = (1 / N) * G * H{l-1}' + 2 * NetParams.lambda * NetParams.W{l};
    else
        dJdW = (1 / N) * G * X' + 2 * NetParams.lambda * NetParams.W{l};
    end
    dJdb = (1 / N) * G * ones(N, 1);

    Grads.W{l} = dJdW; Grads.b{l} = dJdb;
    Grads.gamma{l} = dJdGamma; Grads.beta{l} = dJdBeta;

    if l > 1
        G = NetParams.W{l}' * G;
        G = G .* (H{l-1} > 0);
    else
        break;
    end
end
    
end

function G = BackNormBackPass(G, S, mu, v)
    N = size(S, 2);
    s1 = ((v + eps).^(-0.5));
    s2 = ((v + eps).^(-1.5));

    G1 = G .* (s1 * ones(1, N));
    G2 = G .* (s2 * ones(1, N));

    D = S - mu * ones(1, N);
    c = (G2 .* D) * ones(N, 1);

    G = G1 - (1 / N) * G1 * ones(N, 1) - (1 / N) * D .* (c * ones(1, N));
end

function [mu, v, S, S_hat] = parse_inputs(inputs, l)

    % Set defaults
    mu = cell(1, l); v = cell(1, l); S = cell(1, l); S_hat = cell(1, l);
    for i=1:l
        mu{i} = 0;
        v{i} = 1;
        S{i} = 0;
        S_hat{i} = 0;
    end

    % Go through option pairs
    for a = 1:2:numel(inputs)
        switch lower(inputs{a})
            case 'mean'
                mu = inputs{a+1};
            case 'variance'
                v = inputs{a+1};
            case 's'
                S = inputs{a+1};
            case 'shat'
                S_hat = inputs{a+1};
            otherwise
                error('Input option %s not recognized', inputs{a});
        end
    end
    
    return
    
end