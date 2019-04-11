%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to analyse best range of eta for cyclical learning rate, and plot
% resuls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
% 
% Xval = Xtrain(:, end-999:end);
% Yval = Ytrain(:, end-999:end);
% yval = ytrain(end-999:end);
% 
% Xtrain = Xtrain(:, 1:end-1000);
% Ytrain = sparse(Ytrain(:, 1:end-1000));
% ytrain = ytrain(1:end-1000);

[Xtest, Ytest, ytest] = LoadBatchZeroMean('test_batch.mat');

X.train = Xtrain; X.test = Xtest;
Y.train = Ytrain; Y.test = Ytest;
y.train = ytrain; y.test = ytest;

%% Initialize W, b

D = size(X.train, 1);
M = 500;
C = 10;

dimensions = [D, M, C];

% rng(400);
[W, b] = XavierInitialise(dimensions);

MAX_EPOCH = 1000;

lambda = 0;

%% 

GDParams.n_cycles = 1;
GDParams.eta_min = 1e-6;
GDParams.eta_max = 0.1;
GDParams.l = 0;
GDParams.t = 0;
GDParams.n_batch = 100;
GDParams.n_s = floor(4 * size(X.train, 2) / GDParams.n_batch);
GDParams.n_epochs = floor(size(X.train, 2) / GDParams.n_s / GDParams.n_cycles / 2);
GDParams.start_epoch = 1;
GDParams.lambda = 0.0027;

accuracy.train = zeros(1, GDParams.n_cycles*GDParams.n_s + 1);

Ws = W;
bs = b;
j = zeros(1, MAX_EPOCH);
t = 0;

[l_train(1), J_train(1)]  = ComputeCost2(X.train, Y.train, Ws, bs, GDParams.lambda); J.train = J_train; l.train = l_train;

accuracy.train(1) = ComputeAccuracy2(X.train, y.train, Ws, bs);

[Ws, bs, J, l, accuracy, t, eta] = MiniBatchGDCyclicalAnalyse(X, Y, y, GDParams, Ws, bs, J, l, accuracy, t);

% Plot accuracy

figure; 

plottitle = ["accuracy vs \eta plot, M=", M];

title(join(plottitle, ""), 'Interpreter','tex');

semilogx(eta, accuracy.train(1:end-1), 'LineWidth', 1.2);

legend('training accuracy', 'Location','southeast');

xlabel('\eta');
ylabel('accuracy');
% axis([0, max(eta), 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

plotname = ["plots/accuracy_M", M, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

close all;

% Plot cost

figure; 

plottitle = ["cost vs \eta plot, M=", M];

title(char(join(plottitle, "")), 'Interpreter','tex');

semilogx(eta, J.train, 'LineWidth', 1.2);

legend('training cost', 'Location','southeast');

xlabel('\eta');
ylabel('accuracy');
% axis([0, max(eta), 0.8 * min(J.train), 1.1 * max(J.train)]);

plotname = ["plots/cost_M", M, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

close all;
