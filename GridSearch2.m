addpath datasets\cifar-10

[Xtrain1, Ytrain1, ytrain1] = LoadBatchZeroMean('data_batch_1.mat');
[Xtrain2, Ytrain2, ytrain2] = LoadBatchZeroMean('data_batch_2.mat');
[Xtrain3, Ytrain3, ytrain3] = LoadBatchZeroMean('data_batch_3.mat');
[Xtrain4, Ytrain4, ytrain4] = LoadBatchZeroMean('data_batch_4.mat');
[Xtrain5, Ytrain5, ytrain5] = LoadBatchZeroMean('data_batch_5.mat');

Xtrain = [Xtrain1, Xtrain2, Xtrain3, Xtrain4, Xtrain5];
Ytrain = [Ytrain1, Ytrain2, Ytrain3, Ytrain4, Ytrain5];
ytrain = [ytrain1; ytrain2; ytrain3; ytrain4; ytrain5];

% Shuffle data
random = randperm(size(Xtrain,2));
Xtrain = Xtrain(:, random);
Ytrain = Ytrain(:, random);
ytrain = ytrain(random, :);

Xval = Xtrain(:, end-4999:end);
Yval = Ytrain(:, end-4999:end);
yval = ytrain(end-4999:end);

Xtrain = Xtrain(:, 1:end-5000);
Ytrain = sparse(Ytrain(:, 1:end-5000));
ytrain = ytrain(1:end-5000);

[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

X.train = Xtrain; X.val = Xval; X.test = Xtest;
Y.train = Ytrain; Y.val = Yval; Y.test = Ytest;
y.train = ytrain; y.val = yval; y.test = ytest;

%% Initialize W, b

D = size(Xtrain, 1);
M = 50;
C = 10;

dimensions = [D, M, C];

% rng(400);
[W, B] = XavierInitialise(dimensions);


%% Evaluate

H = EvaluateClassifier(Xtrain(:, 1:100), W{1}, b{1})
H(H < 0) = 0;
P = SoftMax(EvaluateClassifier(H, W{2}, b{2}))

J = ComputeCost2(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda)

[gradW, gradb] = ComputeGradients2(Xtrain(:, 1:100), Ytrain(:, 1:100), P, H, W, lambda);

[ngradb, ngradW] = ComputeGradsNum2(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda, 1e-6);

correlation(1) = sum(abs(ngradW{1} - gradW{1})) / max(1e-6, sum(abs(ngradW{1})) + sum(abs(gradW{1})));
correlation(2) = sum(abs(ngradW{2} - gradW{2})) / max(1e-6, sum(abs(ngradW{2})) + sum(abs(gradW{2})));

%% Initialise parameters

MAX_EPOCH = 1000;

lambda = 0;
l_min = -5;
l_max = -1;

%%

% Lambdas to search
l_min = -5;
l_max = -1;
lambda = logspace(l_min, l_max);

best_accuracy = zeros(1, size(lambda, 2));

for i=1:size(lambda, 2)
    
    GDParams.n_cycles = 2;
    GDParams.eta_min = 1e-5;
    GDParams.eta_max = 1e-1;
    GDParams.l = 0;
    GDParams.t = 0;
    GDParams.n_batch = 100;
    GDParams.n_s = 2 * floor(size(X.train, 2) / GDParams.n_batch);
    GDParams.n_epochs = 2 * GDParams.n_cycles * GDParams.n_s / GDParams.n_batch;
    GDParams.start_epoch = 1;
    GDParams.lambda = lambda(i);
    
    clear J_train J_test J_val l_train l_val l_test
    
    [W, b] = XavierInitialise(dimensions);

    Wstar = cell(MAX_EPOCH, 2);
    bstar = cell(MAX_EPOCH, 2);
    accuracy.train = zeros(1, GDParams.n_epochs + 1);
    accuracy.validation = zeros(1, GDParams.n_epochs + 1);
    accuracy.test = zeros(1, GDParams.n_epochs + 1);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    t = 0;
    
    [l_train(1), J_train(1)]  = ComputeCost2(Xtrain, Ytrain, Ws, bs, GDParams.lambda); J.train = J_train; l.train = l_train;
    [l_val(1), J_val(1)] = ComputeCost2(Xval, Yval, Ws, bs, GDParams.lambda); J.val = J_val; l.val = l_val;
    [l_test(1), J_test(1)] = ComputeCost2(Xtest, Ytest, Ws, bs, GDParams.lambda); J.test = J_test; l.test = l_test;

    accuracy.train(1) = ComputeAccuracy2(Xtrain, ytrain, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy2(Xval, yval, Ws, bs);
    accuracy.test(1) = ComputeAccuracy2(Xtest, ytest, Ws, bs);

    [Ws, bs, J, l, accuracy, t, eta] = MiniBatchGD2(X, Y, y, GDParams, Ws, bs, J, l, accuracy, t);
    
    % Compute best accuracy
    best_accuracy(i) = max(accuracy.validation);
end

dataname = ["data_ns", GDParams.n_s, "_lmin", l_min, "lmax", l_max, ".mat"];

save('GDParams', 'best_accuracy', 'lambda');

