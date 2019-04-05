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

Xval = Xtrain(:, end-999:end);
Yval = Ytrain(:, end-999:end);
yval = ytrain(end-999:end);

Xtrain = Xtrain(:, 1:end-1000);
Ytrain = sparse(Ytrain(:, 1:end-1000));
ytrain = ytrain(1:end-1000);

[Xtest, Ytest, ytest] = LoadBatchZeroMean('test_batch.mat');

X.train = Xtrain; X.val = Xval; X.test = Xtest;
Y.train = Ytrain; Y.val = Yval; Y.test = Ytest;
y.train = ytrain; y.val = yval; y.test = ytest;

%% Initialize W, b

D = size(X.train, 1);
M = 100;
C = 10;

dimensions = [D, M, C];

% rng(400);
[W, b] = XavierInitialise(dimensions);

MAX_EPOCH = 1000;

lambda = 0;

%% 

GDParams{1}.n_cycles = 3;
GDParams{1}.eta_min = 0.001;
GDParams{1}.eta_max = 0.005;
GDParams{1}.l = 0;
GDParams{1}.t = 0;
GDParams{1}.n_batch = 100;
GDParams{1}.n_s = floor(4 * size(X.train, 2) / GDParams{1}.n_batch);
GDParams{1}.n_epochs = floor(size(X.train, 2) / GDParams{1}.n_s / GDParams{1}.n_cycles / 2);
GDParams{1}.start_epoch = 1;
GDParams{1}.lambda = 0.0027;

for i=1:size(GDParams)
    
    clear J_train J_test J_val l_train l_val l_test

    accuracy.train = zeros(1, GDParams{1}.n_epochs + 1);
    accuracy.validation = zeros(1, GDParams{1}.n_epochs + 1);
    accuracy.test = zeros(1, GDParams{1}.n_epochs + 1);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    t = 0;
    
    [l_train(1), J_train(1)]  = ComputeCost2(X.train, Y.train, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
    [l_val(1), J_val(1)] = ComputeCost2(X.val, Y.val, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
    [l_test(1), J_test(1)] = ComputeCost2(X.test, Y.test, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

    accuracy.train(1) = ComputeAccuracy2(X.train, y.train, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy2(X.val, y.val, Ws, bs);
    accuracy.test(1) = ComputeAccuracy2(X.test, y.test, Ws, bs);

    [Ws, bs, J, l, accuracy, t, eta] = MiniBatchGDCyclical(X, Y, y, GDParams{i}, Ws, bs, J, l, accuracy, t);
    
    % Plot accuracy

    figure; 

    plottitle = ["accuracy vs epoch plot, \eta=", GDParams{i}.eta_min, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot(0:GDParams{i}.n_s:2*GDParams{i}.n_s*GDParams{i}.n_cycles, accuracy.train, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_s:2*GDParams{i}.n_s*GDParams{i}.n_cycles, accuracy.validation, 'LineWidth', 1.2);
    plot(0:GDParams{i}.n_s:2*GDParams{i}.n_s*GDParams{i}.n_cycles, accuracy.test, 'LineWidth', 1.2);
    hold off

    legend('training accuracy', 'validation accuracy', 'test accuracy', 'Location','southeast');

    xlabel('update step');
    ylabel('accuracy');
    axis([0, 2*GDParams{i}.n_s*GDParams{i}.n_cycles, 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

    plotname = ["plots/accuracy_lambda", GDParams{i}.lambda, "_etamin", GDParams{i}.eta_min, "_etamax", GDParams{i}.eta_max, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
end
