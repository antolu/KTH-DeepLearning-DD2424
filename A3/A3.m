addpath ..\datasets\cifar-10

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
C = 10;
% M = [50, 40, 10];
% M = [50, 50];
M = [50, 30, 20, 20, 10, 10, 10, 10];

% rng(400);
[W, b] = XavierInitialise(C, D, M);

NetParams.W = W;
NetParams.b = b;
NetParams.use_bn = 0;

MAX_EPOCH = 1000;

lambda = 0;

%% Evaluate

[P, H] = EvaluatekLayer(Xtrain(:, 1:10), NetParams);
% [P2, H2] = Evaluate2Layer(Xtrain(:, 1:10), NetParams.W, NetParams.b);

% H = EvaluateClassifier(Xtrain(:, 1:100), W{1}, b{1})
% H(H < 0) = 0;
% P = SoftMax(EvaluateClassifier(H, W{2}, b{2}))

J = ComputeCost(Xtrain(:, 1:10), Ytrain(:, 1:10), NetParams, lambda)

[gradW, gradb] = ComputeGradients(Xtrain(:, 1:10), Ytrain(:, 1:10), P, H, NetParams, lambda);
[gradW2, gradb2] = ComputeGradients2(Xtrain(:, 1:10), Ytrain(:, 1:10), P, H, NetParams.W, lambda);

Grads = ComputeGradsNumSlow(Xtrain(:, 1:10), Ytrain(:, 1:10), NetParams, lambda, 1e-6);
% [ngradb, ngradW] = ComputeGradsNum(Xtrain(:, 1:10), Ytrain(:, 1:10), NetParams.W, NetParams.b, lambda, 1e-6);

ngradW = Grads.W;
ngradb = Grads.b;

for i=1:max(size(gradW))
    correlation(i) = sum(abs(ngradW{i} - gradW{i})) / max(1e-6, sum(abs(ngradW{i})) + sum(abs(gradW{i})));
end

%% 

GDParams.n_cycles = 2;
% GDParams.eta_min = 0.000600994000000000;
% GDParams.eta_min = 0.00065;
% GDParams.eta_max = 0.08;
GDParams.eta_min = 1e-5;
GDParams.eta_max = 0.1;
GDParams.l = 0;
GDParams.t = 0;
GDParams.n_batch = 100;
GDParams.n_s = floor(4 * size(X.train, 2) / GDParams.n_batch);
GDParams.n_epochs = floor(GDParams.n_batch * GDParams.n_cycles * 2 *GDParams.n_s / size(X.train, 2));
GDParams.start_epoch = 1;
% GDParams.lambda = 0.0027;
GDParams.lambda = 0.005;

clear J_train J_test J_val l_train l_val l_test

accuracy.train = zeros(1, GDParams.n_epochs + 1);
accuracy.validation = zeros(1, GDParams.n_epochs + 1);
accuracy.test = zeros(1, GDParams.n_epochs + 1);

j = zeros(1, MAX_EPOCH);
t = 0;

[l_train(1), J_train(1)]  = ComputeCost(X.train, Y.train, NetParams, GDParams.lambda); J.train = J_train; l.train = l_train;
[l_val(1), J_val(1)] = ComputeCost(X.val, Y.val, NetParams, GDParams.lambda); J.val = J_val; l.val = l_val;
[l_test(1), J_test(1)] = ComputeCost(X.test, Y.test, NetParams, GDParams.lambda); J.test = J_test; l.test = l_test;

accuracy.train(1) = ComputeAccuracy(X.train, y.train, NetParams);
accuracy.validation(1) = ComputeAccuracy(X.val, y.val, NetParams);
accuracy.test(1) = ComputeAccuracy(X.test, y.test, NetParams);

[Ws, bs, J, l, accuracy, t, eta] = MiniBatchGDCyclical(X, Y, y, GDParams, NetParams, J, l, accuracy, t);

%%

% Plot accuracy

figure; 

plottitle = ["accuracy vs update step plot, M=", M];

title(join(plottitle, ""), 'Interpreter','tex');

hold on
plot(0:GDParams.n_s/4:2*GDParams.n_s*GDParams.n_cycles, accuracy.train, 'LineWidth', 1.2);
plot(0:GDParams.n_s/4:2*GDParams.n_s*GDParams.n_cycles, accuracy.validation, 'LineWidth', 1.2);
plot(0:GDParams.n_s/4:2*GDParams.n_s*GDParams.n_cycles, accuracy.test, 'LineWidth', 1.2);
hold off

legend('training accuracy', 'validation accuracy', 'test accuracy', 'Location','southeast');

xlabel('update step');
ylabel('accuracy');
axis([0, 2*GDParams.n_s*GDParams.n_cycles, 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

plotname = ["plots/accuracy_M", M, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

close all;

% Plot cost

figure; 

plottitle = ["cost vs update step plot, M=", M];

title(join(plottitle, ""), 'Interpreter','tex');

hold on
plot(0:GDParams.n_s/4:2*GDParams.n_s*GDParams.n_cycles, J.train, 'LineWidth', 1.2);
plot(0:GDParams.n_s/4:2*GDParams.n_s*GDParams.n_cycles, J.val, 'LineWidth', 1.2);
plot(0:GDParams.n_s/4:2*GDParams.n_s*GDParams.n_cycles, J.test, 'LineWidth', 1.2);
hold off

legend('training accuracy', 'validation accuracy', 'test accuracy', 'Location','northeast');

xlabel('update step');
ylabel('accuracy');
axis([0, 2*GDParams.n_s*GDParams.n_cycles, 0.8 * min(J.train), 1.1 * max(J.train)]);

plotname = ["plots/cost_M", M, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

close all;
