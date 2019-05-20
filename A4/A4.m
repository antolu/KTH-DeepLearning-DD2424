%% Read in data

addpath data
book_fname = "goblet_book.txt";
fid = fopen(book_fname, 'r', 'n', 'UTF-8');
book_data = fscanf(fid, '%c');

book_chars = unique(book_data);
K = numel(book_chars);

char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

for k=1:K
    char_to_ind(book_chars(k)) = k;
    ind_to_char(k) = book_chars(k);
end

%% Set hyperparameters

M = 100;
seq_length = 25;
eta = 0.1;
sig = 0.1;

rng(400);
RNN.b = zeros(M, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(M, K) * sig;
RNN.W = randn(M, M) * sig;
RNN.V = randn(K, M) * sig;

h0 = zeros(M, 1); 
% x0 = zeros(K, 1); x(23) = 1;

% seq = SyntesizeSequence(RNN, h0, x0, 25);

% txt = SequenceToText(ind_to_char, seq);

X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = OneHotRepresentation(char_to_ind, X_chars);
Y = OneHotRepresentation(char_to_ind, Y_chars);

%% Compare numerical gradient

[P, H, a, l] = ForwardPass(X, Y, RNN, h0);
% L = ComputeLoss(X, Y, RNN, h0);
Grads = BackwardPass(RNN, P, H, a, X, Y);

%%

NGrads = ComputeGradsNum(X, Y, RNN, 1e-4);

%% Correlation

for f = fieldnames(Grads)'
    correlation.(f{1}) = sum(abs(NGrads.(f{1}) - Grads.(f{1}))) / max(1e-6, sum(abs(NGrads.(f{1}))) + sum(abs(Grads.(f{1}))));
end

%% Clip gradients

% for f = fieldnames(Grads)'
%     Grads.(f{1}) = max(min(Grads.(f{1}), 5), -5);
% end

%% Run SGD

SGDParams.seq_length = seq_length;
SGDParams.h0 = h0;
SGDParams.char_to_ind = char_to_ind;
SGDParams.ind_to_char = ind_to_char;
SGDParams.book_data = book_data;
SGDParams.eta = eta;
SGDParams.n_epochs = 10;
SGDParams.gamma = 0.9;

RNN = SGD(RNN, SGDParams);

%%

function seq = SyntesizeSequence(RNN, h0, x0, n) 

    K = size(RNN.U, 2);
    
    a = cell(n, 1); h = cell(n+1, 1); x = cell(n+1, 1); 
    o = cell(n, 1); seq = zeros(n, 1); p = cell(n, 1);
    h{1} = h0; x{1} = x0;
    Y = zeros(K, n);

    for t=1:n
        a{t} = RNN.W * h{t} + RNN.U * x{t} + RNN.b;
        h{t+1} = tanh(a{t});
        o{t} = RNN.V * h{t+1};
        p{t} = SoftMax(o{t});

        cp = cumsum(p{t});
        j = rand;
        ixs = find(cp-j > 0);
        ii = ixs(1);

        y = zeros(K, 1); y(ii) = 1; 
        Y(:, n) = y;
        x{t+1} = y;
        seq(t) = ii;
    end

end


function [P, H, a, L] = ForwardPass(X, Y, RNN, h0)
    seq_length = size(X, 2);
    M = size(RNN.b, 1); K = size(RNN.c, 1);
    a = zeros(M, seq_length); H = zeros(M, seq_length+1); H(:, 1) = h0;
    o = zeros(K, seq_length); P = zeros(K, seq_length);

    for t=1:seq_length 
        a(:, t) = RNN.W * H(:, t) + RNN.U * X(:, t) + RNN.b;
        H(:, t+1) = tanh(a(:, t));
        o(:, t) = RNN.V * H(:, t+1) + RNN.c;
        P(:, t) = SoftMax(o(:, t));
    end

    L = sum(-log(sum(Y .* P)));
end


function L = ComputeLoss(X, Y, RNN, h0)
    %ComputeLoss - Computes the cross-entropy loss
    %
    % Syntax: L = ComputeLoss(X, Y, RNN, h0)
    %

    [p, ~, ~, l] = ForwardPass(X, Y, RNN, h0);

    L = l;
        
end


function Grads = BackwardPass(RNN, P, H, a, Y, X)
    seq_length = size(X, 2);
    [K, M] = size(RNN.V);

    G = -(Y - P);

    % dLdV = G * H(:, 2:end)';
    dLdV = 0;
    for t=1:seq_length
        dLdV = dLdV + G(:, t) * H(:, t+1)';
    end
    
    dLdc = sum(G, 2);

    Grads.V = dLdV;
    Grads.c = dLdc;

    dLdh = zeros(seq_length, M);
    dLda = zeros(seq_length, M);
    dLdh(seq_length, :) = G(:, seq_length)' * RNN.V;
    dLda(seq_length, :) = dLdh(seq_length, :) * diag(1 - tanh(a(:, seq_length)).^2);

    for t=seq_length-1:-1:1
        dLdh(t, :) = G(:, t)' * RNN.V + dLda(t+1, :) * RNN.W;
        dLda(t, :) = dLdh(t, :) * diag(1 - tanh(a(:, t)).^2);
    end

    G = dLda';

    dLdW = 0;
    for t=2:seq_length+1
        dLdW = dLdW + G(:, t-1) * H(:, t)';
    end

    dLdU = 0;
    for t=1:seq_length
        dLdU = dLdU + G(:, t) * X(:, t)';
    end

    dLdb = sum(G, 2);

    Grads.W = dLdW;
    Grads.U = dLdU;
    Grads.b = dLdb;

end


function P = SoftMax(s)

    P = exp(s) ./ (sum(exp(s)));
    
end


function txt = SequenceToText(ind_to_char, charseq)
    txt = char(zeros(1, numel(charseq)));

    for i=1:numel(charseq)
        txt(i) = ind_to_char(charseq(i));
    end
end


function onehot = OneHotRepresentation(char_to_ind, X)
    K = size(char_to_ind, 1);
    seq_length = numel(X);
    
    onehot = zeros(K, seq_length);

    for i=1:seq_length
        onehot(char_to_ind(X(i)), i) = 1;
    end
end


function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end


function Grads = ClipGradients(Grads)
    for f = fieldnames(Grads)'
        Grads.(f{1}) = max(min(Grads.(f{1}), 5), -5);
    end
end


function RNN = SGD(RNN, SGDParams)

    for epoch=1:SGDParams.n_epochs
        hprev = SGDParams.h0; 
        clear m;
    
        % Initialise momentum
        for f = fieldnames(RNN)'
            m.(f{1}){1} = 0;
        end
    
        for i=floor(1:size(SGDParams.book_data, 2)/SGDParams.seq_length)
            X_chars = SGDParams.book_data(SGDParams.seq_length*(i-1)+1:SGDParams.seq_length*i);
            Y_chars = SGDParams.book_data(SGDParams.seq_length*(i-1)+2:SGDParams.seq_length*i + 1);
            X = OneHotRepresentation(SGDParams.char_to_ind, X_chars);
            Y = OneHotRepresentation(SGDParams.char_to_ind, Y_chars);

            [P, H, a, l] = ForwardPass(X, Y, RNN, hprev);
            Grads = BackwardPass(RNN, P, H, a, X, Y);

            Grads = ClipGradients(Grads);

            for f = fieldnames(Grads)'
                m.(f{1}){i+1} = SGDParams.gamma * m.(f{1}){i} + (1 - SGDParams.gamma) * Grads.(f{1}).^2;
                eta = SGDParams.eta ./ sqrt(m.(f{1}){i+1} + eps);
                RNN.(f{1}) = RNN.(f{1}) - eta .* Grads.(f{1});
            end

            if exist('smooth_loss', 'var')
                smooth_loss = 0.999 * smooth_loss + 0.001 * l;
            else
                smooth_loss = l;
            end

            hprev = H(:, end);
            if mod(i, 100) == 0
                disp(smooth_loss)
            end
        end
    end

end