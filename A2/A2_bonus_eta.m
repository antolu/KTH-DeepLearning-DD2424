%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to analyse best range of eta for cyclical learning rate, and plot
% resuls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath ../datasets/cifar-10
addpath ..
mkdir plots

[Xtrain1, Ytrain1, ytrain1] = LoadBatchZeroMean('data_batch_1.mat');
[Xtrain2, Ytrain2, ytrain2] = LoadBatchZeroMean('data_batch_2.mat');
[Xtrain3, Ytrain3, ytrain3] = LoadBatchZeroMean('data_batch_3.mat');
[Xtrain4, Ytrain4, ytrain4] = LoadBatchZeroMean('data_batch_4.mat');
[Xtrain5, Ytrain5, ytrain5] = LoadBatchZeroMean('data_batch_5.mat');

X.train = [Xtrain1, Xtrain2, Xtrain3, Xtrain4, Xtrain5];
Y.train = [Ytrain1, Ytrain2, Ytrain3, Ytrain4, Ytrain5];
y.train = [ytrain1; ytrain2; ytrain3; ytrain4; ytrain5];

% Shuffle data
random = randperm(size(X.train, 2));
X.train = X.train(:, random);
Y.train = Y.train(:, random);
y.train = y.train(random, :);
% 
% X.val = X.train(:, end-999:end);
% Y.val = Y.train(:, end-999:end);
% y.val = y.train(end-999:end);
% 
% X.train = X.train(:, 1:end-1000);
% Y.train = sparse(Y.train(:, 1:end-1000));
% y.train = y.train(1:end-1000);

[X.test, Y.test, y.test] = LoadBatchZeroMean('test_batch.mat');


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

[l.train(1), J.train(1)]  = ComputeCost(X.train, Y.train, Ws, bs, GDParams.lambda); 

accuracy.train(1) = ComputeAccuracy(X.train, y.train, Ws, bs);

[Ws, bs, J, l, accuracy, t, eta] = MiniBatchGDBestEta(X, Y, y, GDParams, Ws, bs, J, l, accuracy, t);

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
