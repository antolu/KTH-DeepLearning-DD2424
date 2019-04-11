function acc = ComputeAccuracy2(X, y, W, b)

xSize = size(X);
numberOfSamples = xSize(2);

[P, H] = Evaluate2Layer(X, W, b);

V = P==max(P);
matches = V(sub2ind(size(V), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

end

function [l, J] = ComputeCost2(X, Y, W, b, lambda)

numberOfSamples = size(X, 2);

H = EvaluateClassifier(X, W{1}, b{1}); H(H < 0) = 0;
P = SoftMax(EvaluateClassifier(H, W{2}, b{2}));

l = sum(-log(sum(Y .* P)));

l = l / numberOfSamples;

regularisation = lambda * sum(sum(W{1}.^2)) + lambda * sum(sum(W{2}.^2));

J = l + regularisation;

end

function [gradW, gradb] = ComputeGradients2(X, Y, P, H, W, lambda)

xSize = size(X);
batchSize = xSize(2);

Gbatch2 = -(Y - P);

dLdW2 = (1 / batchSize) * Gbatch2 * H';
dLdb2 = (1 / batchSize) * Gbatch2 * ones(batchSize, 1);

gradW2 = dLdW2 + 2 * lambda * W{2};
gradb2 = dLdb2;

Gbatch1 = W{2}' * Gbatch2;
Ind = H;
Ind(Ind>0) = 1;
Gbatch1 = Gbatch1 .* Ind;

dLdW1 = (1 / batchSize) * Gbatch1 * X';
dLdb1 = (1 / batchSize) * Gbatch1 * ones(batchSize, 1);

gradW1 = dLdW1 + 2 * lambda * W{1};
gradb1 = dLdb1;

gradW = {gradW1, gradW2};
gradb = {gradb1, gradb2};

end

function S = EvaluateClassifier(X, W, b)

S = W * X + b * ones(1, size(X, 2));

end

function P = SoftMax(s)

P = exp(s) ./ (sum(exp(s)));

end

function [X, Y, y] = LoadBatchZeroMean(filename, varargin)

A = load(filename);

X = double(A.data');

meanX = mean(X, 2);
stdX = std(X, 0, 2);

X = X - repmat(meanX, [1, size(X, 2)]);
X = X ./ repmat(stdX, [1, size(X, 2)]);

y = double(A.labels) + 1;

Y = zeros(10, 10000);

for i=1:10000
    Y(y(i), i) = 1;
end

end

function [W, b] = XavierInitialise(dimensions)
% dimensions should be in format [D, M, C]

D = dimensions(1);
M = dimensions(2);
C = dimensions(3);

W = {1/sqrt(D) * randn(M, D), 1/sqrt(M) * randn(C, M)};
b = {zeros(M, 1), zeros(C, 1)};

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Grid search for lambda
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

[Xtest, Ytest, ytest] = LoadBatchZeroMean('test_batch.mat');

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

% lambda = 0;
% l_min = -5;
% l_max = -1;
l_min = 0.001;
l_max = 0.008;

%%

% Lambdas to search
% l_min = -5;
% l_max = -1;
% lambda = logspace(l_min, l_max);
lambda = linspace(l_min, l_max, 50);

best_accuracy = zeros(1, size(lambda, 2));
accuracies = cell(1, size(lambda, 2));

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
    
    accuracies{i} = accuracy;
    
    % Compute best accuracy
    best_accuracy(i) = max(accuracy.validation);
    i
end

dataname = ["data_ns", GDParams.n_s, "_lmin", l_min, "lmax", l_max, ".mat"];

save(join(dataname, ""), 'GDParams', 'best_accuracy', 'lambda', 'accuracies');

%% Plot accuracy

figure; 

plottitle = ["Accuracy vs \lambda plot, coarse search"];

title(join(plottitle, ""), 'Interpreter','tex');

semilogx(lambda, best_accuracy, 'LineWidth', 1.2);

xlabel('\lambda');
ylabel('accuracy');
axis([0, max(lambda), 0.8 * min(best_accuracy), 0.6]);

plotname = ["plots/accuracy_lmin", l_min, "_lmax", l_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Code for first half of assignment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath datasets\cifar-10

[Xtrain, Ytrain, trainy] = LoadBatchZeroMean('data_batch_1.mat');
[Xval, Yval, yval] = LoadBatchZeroMean('data_batch_2.mat');
[Xtest, Ytest, ytest] = LoadBatchZeroMean('test_batch.mat');

X.train = Xtrain; X.val = Xval; X.test = Xtest;
Y.train = Ytrain; Y.val = Yval; Y.test = Ytest;
y.train = trainy; y.val = yval; y.test = ytest;

%% Initialize W, b

D = size(Xtrain, 1);
M = 50;
C = 10;

dimensions = [D, M, C];

% rng(400);
[W, b] = XavierInitialise(dimensions);

MAX_EPOCH = 1000;

lambda = 0;

%% Evaluate

H = EvaluateClassifier(Xtrain(:, 1:100), W{1}, b{1})
H(H < 0) = 0;
P = SoftMax(EvaluateClassifier(H, W{2}, b{2}))

J = ComputeCost2(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda)

[gradW, gradb] = ComputeGradients2(Xtrain(:, 1:100), Ytrain(:, 1:100), P, H, W, lambda);

[ngradb, ngradW] = ComputeGradsNum2(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda, 1e-6);

correlation(1) = sum(abs(ngradW{1} - gradW{1})) / max(1e-6, sum(abs(ngradW{1})) + sum(abs(gradW{1})));
correlation(2) = sum(abs(ngradW{2} - gradW{2})) / max(1e-6, sum(abs(ngradW{2})) + sum(abs(gradW{2})));

%% 

GDParams{1}.n_cycles = 1;
GDParams{1}.eta_min = 1e-5;
GDParams{1}.eta_max = 1e-1;
GDParams{1}.n_s = 500;
GDParams{1}.l = 0;
GDParams{1}.t = 0;
GDParams{1}.n_batch = 100;
GDParams{1}.n_epochs = 2 * GDParams{1}.n_cycles * GDParams{1}.n_s / GDParams{1}.n_batch;
GDParams{1}.start_epoch = 1;
GDParams{1}.lambda = 0.01;

for i=1:size(GDParams)
    
    clear J_train J_test J_val l_train l_val l_test

    Wstar = cell(MAX_EPOCH, 2);
    bstar = cell(MAX_EPOCH, 2);
    accuracy.train = zeros(1, GDParams{i}.n_epochs + 1);
    accuracy.validation = zeros(1, GDParams{i}.n_epochs + 1);
    accuracy.test = zeros(1, GDParams{i}.n_epochs + 1);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    t = 0;
    
    [l_train(1), J_train(1)]  = ComputeCost2(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
    [l_val(1), J_val(1)] = ComputeCost2(Xval, Yval, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
    [l_test(1), J_test(1)] = ComputeCost2(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

    accuracy.train(1) = ComputeAccuracy2(Xtrain, trainy, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy2(Xval, yval, Ws, bs);
    accuracy.test(1) = ComputeAccuracy2(Xtest, ytest, Ws, bs);

    [Ws, bs, J, l, accuracy, t, eta] = MiniBatchGD2(X, Y, y, GDParams{i}, Ws, bs, J, l, accuracy, t);

    % Plot cost

    figure; 

    plottitle = ["cost vs epoch plot, \eta=", GDParams{i}.eta_max, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), J.train, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), J.val, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), J.test, 'LineWidth', 1.2);
    

    legend('training cost', 'validation cost', 'test cost');

    xlabel('update step');
    ylabel('cost');
    axis([0, GDParams{i}.n_epochs*GDParams{i}.n_batch, 0.75 * min(J.train), 1.1 * max(J.train)]);

    plotname = ["plots/cost_lambda", GDParams{i}.lambda, "_etamin", GDParams{i}.eta_min, "_etamax", GDParams{i}.eta_max, ".eps"];
    hold off

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
    
    % Plot loss

    figure; 

    plottitle = ["loss vs epoch plot, \eta=", GDParams{i}.eta_min, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), l.train, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), l.val, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), l.test, 'LineWidth', 1.2);
    hold off

    legend('training loss', 'validation loss', 'test loss');

    xlabel('update step');
    ylabel('loss');
    axis([0, GDParams{i}.n_epochs*GDParams{i}.n_batch, 0.75 * min(l.train), 1.1 * max(l.train)]);

    plotname = ["plots/loss_lambda", GDParams{i}.lambda, "_etamin", GDParams{i}.eta_min, "_etamax", GDParams{i}.eta_max, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
    
    % Plot accuracy

    figure; 

    plottitle = ["accuracy vs epoch plot, \eta=", GDParams{i}.eta_min, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), accuracy.train, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), accuracy.validation, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_batch:(GDParams{i}.n_epochs*GDParams{i}.n_batch), accuracy.test, 'LineWidth', 1.2);
    hold off

    legend('training accuracy', 'validation accuracy', 'test accuracy', 'Location','southeast');

    xlabel('update step');
    ylabel('accuracy');
    axis([0, GDParams{i}.n_epochs*GDParams{i}.n_batch, 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

    plotname = ["plots/accuracy_lambda", GDParams{i}.lambda, "_etamin", GDParams{i}.eta_min, "_etamax", GDParams{i}.eta_max, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
end
