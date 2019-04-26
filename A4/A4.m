%% Read in data

addpath data
book_fname = "goblet_book.txt";
fid = fopen(book_fname, 'r');
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

M = 5;
seq_length = 25;
eta = 0.1;
sig = 0.01;

RNN.b = zeros(M, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(M, K) * sig;
RNN.W = randn(M, M) * sig;
RNN.V = randn(K, M) * sig;

h0 = zeros(M, 1); % h(3) = 1;
x0 = zeros(K, 1); x(23) = 1;

seq = SyntesizeSequence(RNN, h0, x0, 25);

txt = SequenceToText(ind_to_char, seq);

X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = OneHotRepresentation(char_to_ind, X_chars);
Y = OneHotRepresentation(char_to_ind, Y_chars);

%% Compare numerical gradient

[P, H, a] = ForwardPass(X, Y, RNN, h0);
L = ComputeLoss(X, Y, RNN, h0);
Grads = BackwardPass(RNN, P, H, a, X, Y);

%%

NGrads = ComputeGradsNum(X, Y, RNN, 1e-4);

%% Correlation

for f = fieldnames(RNN)'
    correlation.(f{1}) = sum(abs(NGrads.(f{1}) - Grads.(f{1}))) / max(1e-6, sum(abs(NGrads.(f{1}))) + sum(abs(Grads.(f{1}))));
end

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

function [p, h, a] = ForwardPass(X, Y, RNN, h0)
    seq_length = size(X, 2);
    a = cell(seq_length, 1); h = cell(seq_length+1, 1); x = cell(seq_length+1, 1); 
    o = cell(seq_length, 1); p = cell(seq_length, 1);
    
    h{1} = h0; 

    for t=1:seq_length
        a{t} = RNN.W * h{t} + RNN.U * X(:, t)+ RNN.b;
        h{t+1} = tanh(a{t});
        o{t} = RNN.V * h{t+1} + RNN.c;
        p{t} = SoftMax(o{t});
    end
end

function L = ComputeLoss(X, Y, RNN, h0)
    %ComputeLoss - Computes the cross-entropy loss
    %
    % Syntax: L = ComputeLoss(X, Y, RNN, h0)
    %

    seq_length = size(X, 2);

    [p, ~, ~] = ForwardPass(X, Y, RNN, h0);

    L = 0;
    for t=1:seq_length
        L = L -log(sum(Y(:, t) .* p{t}));
    end
        
end

function Grads = BackwardPass(RNN, P, H, a, Y, X)
    seq_length = numel(H)-1;
    [M, K] = size(RNN.U);

    dLdo = cell(1, seq_length);
    dLdh = cell(1, seq_length);
    dLda = cell(1, seq_length);

    for i=1:seq_length
        dLdo{i} = -(Y(:, i) - P{i})';
    end

    dLdh{seq_length} = dLdo{seq_length} * RNN.V;
    dLda{seq_length} = dLdh{seq_length} * diag(1 - tanh(a{seq_length}).^2);

    for t=seq_length-1:-1:1
        dLdh{t} = dLdo{t} * RNN.V + dLda{t+1} * RNN.W;
        dLda{t} = dLdh{t} * diag(1 - tanh(a{t}).^2);
    end

    dLdV = zeros(K, M);
    for t=1:seq_length
        dLdV = dLdV + dLdo{t}' * H{t+1}';
    end

    dLdc = zeros(K, 1);
    for t=1:seq_length
        dLdc = dLdc + dLdo{i}';
    end

    dLdW = zeros(M, M);
    for t=1:seq_length
        dLdW = dLdW + dLda{t}' * H{t}';
    end

    dLdU = zeros(M, K);
    for t=1:seq_length
        dLdU = dLdU + dLda{t}' * X(:, i)';
    end

    dLdb = zeros(M, 1);
    for t=1:seq_length
        dLdb = dLdb + dLda{t}';
    end

    Grads.V = dLdV;
    Grads.c = dLdc;
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