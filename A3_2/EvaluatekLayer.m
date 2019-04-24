function [P, H, mu, v, S, S_hat] = EvaluatekLayer(X, NetParams, varargin)
%EvaluatekLayer - Evaluates a k layer network (forward pass)
%
% Syntax: [P, H] = EvaluatekLayer(X, NetParams, varargin)
%
% Forward pass

    k = numel(NetParams.W);
    N = size(X, 2);

    [mu, v] = parse_inputs(varargin, k);

    S = cell(1, k-1);
    S_hat = cell(1, k-1);
    S_tilde = cell(1, k-1);
    H = cell(1, k-1);

    for l=1:k-1
        S{l} = EvaluateClassifier(X, NetParams.W{l}, NetParams.b{l});

        if NetParams.use_bn
            if mu{l} == 0
                mu{l} = sum(S{l}, 2) / N;
            end
            if v{l} == 1
                v{l} = sum((S{l} - mu{l}).^2, 2) / N;
            end
        end

        S_hat{l} = BatchNormalise(S{l}, mu{l}, v{l});
        S_tilde{l} = ScaleAndShift(S_hat{l}, NetParams.gamma{l}, NetParams.beta{l});
        X = max(S_tilde{l}, 0); H{l} = X;
    end

    S_end = EvaluateClassifier(X, NetParams.W{k}, NetParams.b{k});
    P = SoftMax(S_end);

end

function S_hat = BatchNormalise(S, mu, v)
% Performs batch normalisation
    S_hat = (diag(v) + eps)^(-.5) * (S - mu);
end

function S_tilde = ScaleAndShift(S_hat, gammas, betas)
% Scales and shifts normalised activations
    S_tilde = gammas .* S_hat + betas;
end

function S = EvaluateClassifier(X, W, b)
% Evaluates a single layer

    S = W * X + b * ones(1, size(X, 2));
    
end

function P = SoftMax(s)
% Applies the SoftMax function
    format long;
    P = exp(s) ./ (sum(exp(s)));
    
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