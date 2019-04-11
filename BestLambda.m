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
% random = randperm(size(Xtrain,2));
% Xtrain = Xtrain(:, random);
% Ytrain = Ytrain(:, random);
% ytrain = ytrain(random, :);

Xval = Xtrain(:, end-999:end);
Yval = Ytrain(:, end-999:end);
yval = ytrain(end-999:end);

Xtrain = Xtrain(:, 1:end-1000);
Ytrain = Ytrain(:, 1:end-1000);
ytrain = ytrain(1:end-1000);

[Xtest, Ytest, ytest] = LoadBatchZeroMean('test_batch.mat');

X.train = Xtrain; X.val = Xval; X.test = Xtest;
Y.train = Ytrain; Y.val = Yval; Y.test = Ytest;
y.train = ytrain; y.val = yval; y.test = ytest;

%% Initialize W, b

D = size(X.train, 1);
M = 50;
C = 10;

dimensions = [D, M, C];

% rng(400);
[W, b] = XavierInitialise(dimensions);


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

% lambda = 0.00273469387755102;
lambda = 0.00157142857142857;
% l_min = -5;
% l_max = -1;
% l_min = 0.001;
% l_max = 0.006;

%%

% Lambdas to search
% l_min = -5;
% l_max = -1;
% lambda = logspace(l_min, l_max);
% lambda = linspace(l_min, l_max, 50);
% 
% best_accuracy = zeros(1, size(lambda, 2));
% accuracies = cell(1, size(lambda, 2));

% for i=1:size(lambda, 2)

i = 1;
    
GDParams.n_cycles = 6;
GDParams.eta_min = 1e-5;
GDParams.eta_max = 1e-1;
GDParams.l = 0;
GDParams.t = 0;
GDParams.n_batch = 100;
GDParams.n_s = 2 * floor(size(X.train, 2) / GDParams.n_batch);
GDParams.n_epochs = floor(2 * GDParams.n_cycles * GDParams.n_s / GDParams.n_batch);
GDParams.start_epoch = 1;
GDParams.lambda = lambda(i);

clear J_train J_test J_val l_train l_val l_test

[W, b] = XavierInitialise(dimensions);

Wstar = cell(MAX_EPOCH, 2);
bstar = cell(MAX_EPOCH, 2);
accuracy.train = zeros(1, 1);
accuracy.validation = zeros(1, 1);
accuracy.test = zeros(1, 1);

Ws = W;
bs = b;
j = zeros(1, MAX_EPOCH);
t = 0;

[l_train(1), J_train(1)]  = ComputeCost2(X.train, Y.train, Ws, bs, GDParams.lambda); J.train = J_train; l.train = l_train;
[l_val(1), J_val(1)] = ComputeCost2(X.val, Y.val, Ws, bs, GDParams.lambda); J.val = J_val; l.val = l_val;
[l_test(1), J_test(1)] = ComputeCost2(X.test, Y.test, Ws, bs, GDParams.lambda); J.test = J_test; l.test = l_test;

accuracy.train(1) = ComputeAccuracy2(X.train, y.train, Ws, bs);
accuracy.validation(1) = ComputeAccuracy2(X.val, y.val, Ws, bs);
accuracy.test(1) = ComputeAccuracy2(X.test, y.test, Ws, bs);

[Ws, bs, J, l, accuracy, t, eta] = MiniBatchGD2(X, Y, y, GDParams, Ws, bs, J, l, accuracy, t);

accuracies{i} = accuracy;

% Compute best accuracy
best_accuracy(i) = max(accuracy.validation);
i
% end

dataname = ["data_ns", GDParams.n_s, "_lmin", l_min, "lmax", l_max, ".mat"];

save(join(dataname, ""), 'GDParams', 'best_accuracy', 'lambda', 'accuracies');

%% Plots

% Plot cost

figure; 

plottitle = ["cost vs epoch plot, =", ", \lambda=", GDParams.lambda];

title(join(plottitle, ""), 'Interpreter','tex');

hold on
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), J.train, 'LineWidth', 1.2);
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), J.val, 'LineWidth', 1.2);
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), J.test, 'LineWidth', 1.2);


legend('training cost', 'validation cost', 'test cost');

xlabel('update step');
ylabel('cost');
axis([0, GDParams.n_cycles*GDParams.n_s*2, 0.75 * min(J.train), 1.1 * max(J.train)]);

plotname = ["plots/cost_lambda", GDParams.lambda, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];
hold off

saveas(gca, join(plotname, ""), 'epsc');

close all;

% Plot loss

figure; 

plottitle = ["loss vs epoch plot, ", ", \lambda=", GDParams.lambda];

title(join(plottitle, ""), 'Interpreter','tex');

hold on
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), l.train, 'LineWidth', 1.2);
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), l.val, 'LineWidth', 1.2);
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), l.test, 'LineWidth', 1.2);
hold off

legend('training loss', 'validation loss', 'test loss');

xlabel('update step');
ylabel('loss');
axis([0, GDParams.n_cycles*GDParams.n_s*2, 0.75 * min(l.train), 1.1 * max(l.train)]);

plotname = ["plots/loss_lambda", GDParams.lambda, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

close all;

% Plot accuracy

figure; 

plottitle = ["accuracy vs epoch plot,", ", \lambda=", GDParams.lambda];

title(join(plottitle, ""), 'Interpreter','tex');

hold on
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), accuracy.train, 'LineWidth', 1.2);
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), accuracy.validation, 'LineWidth', 1.2);
plot(0:(GDParams.n_s/2):(GDParams.n_cycles*GDParams.n_s*2), accuracy.test, 'LineWidth', 1.2);
hold off

legend('training accuracy', 'validation accuracy', 'test accuracy', 'Location','southeast');

xlabel('update step');
ylabel('accuracy');
axis([0, GDParams.n_cycles*GDParams.n_s*2, 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

plotname = ["plots/accuracy_lambda", GDParams.lambda, "_etamin", GDParams.eta_min, "_etamax", GDParams.eta_max, ".eps"];

saveas(gca, join(plotname, ""), 'epsc');

close all;