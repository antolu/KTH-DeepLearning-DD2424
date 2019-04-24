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
% random = randperm(size(Xtrain,2));
% Xtrain = Xtrain(:, random);
% Ytrain = Ytrain(:, random);
% ytrain = ytrain(random, :);

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

D = size(Xtrain, 1);
% D = 100;
C = 10;
% M = [50, 40, 10];
M = [50, 50];
% M = [50, 30, 20, 20, 10, 10, 10, 10];

% rng(400);
[W, b] = Initialise(C, D, M, 'Initialisation', 'he');
% [W, b] = Initialise(C, D, M, 'Initialisation', 'uniform', 'Sigma', 0.001);

NetParams.W = W;
NetParams.b = b;

NetParams.use_dropout = 0;
NetParams.P = 0.8;
NetParams.alpha = 0.9;

NetParams.gammas = cell(1, numel(M));
NetParams.betas = cell(1, numel(M));
NetParams.use_bn = 0;
% NetParams.use_bn = 1;

for i=1:numel(M)
    NetParams.gammas{i} = ones(M(i), 1);
    NetParams.betas{i} = zeros(M(i), 1);
end

lambda = 0.1;

%% Check numerical gradient

% [P, H] = EvaluatekLayer(X.train(:, 1:10), NetParams);
    
% Hp = EvaluateClassifier(Xtrain(1:100, 1:10), W{1}, b{1});
% Hp(Hp < 0) = 0;
% P2 = SoftMax(EvaluateClassifier(Hp, W{2}, b{2}));
% 
% H2 = {Hp};
% 
% [gradW, gradb] = ComputeGradients2(X.train(1:100, 1:10), Y.train(:, 1:10), P2, H2, W, lambda);

[P, BNParams] = ForwardPass(X.train, NetParams);
% J = ComputeCost(X.train(1:100, 1:10), Y.train(:, 1:10), NetParams, lambda, 'BNParams', BNParams)

Grads = BackwardPass(X.train(1:100, 1:10), Y.train(:, 1:10), P, BNParams.X, NetParams, lambda, 'BNParams', BNParams);
% [gradW2, gradb2] = ComputeGradients2(X.train(:, 1:10), Y.train(:, 1:10), P, H, NetParams.W, lambda);

NGrads = ComputeGradsNumSlow(X.train(1:100, 1:10), Y.train(:, 1:10), NetParams, lambda, 1e-6);
% [ngradb, ngradW] = ComputeGradsNum(Xtrain(:, 1:10), Ytrain(:, 1:10), NetParams.W, NetParams.b, lambda, 1e-6);

for i=1:max(numel(Grads.W))
    correlation.W(i) = sum(abs(NGrads.W{i} - Grads.W{i})) / max(1e-6, sum(abs(NGrads.W{i})) + sum(abs(Grads.W{i})));
    correlation.b(i) = sum(abs(NGrads.b{i} - Grads.b{i})) / max(1e-6, sum(abs(NGrads.b{i})) + sum(abs(Grads.b{i})));
end

for i=1:max(numel(Grads.W)-1)
    if NetParams.use_bn
        correlation.gamma(i) = sum(abs(NGrads.gammas{i} - Grads.gamma{i})) / max(1e-6, sum(abs(NGrads.gammas{i})) + sum(abs(Grads.gamma{i})));
        correlation.beta(i) = sum(abs(NGrads.betas{i} - Grads.beta{i})) / max(1e-6, sum(abs(NGrads.betas{i})) + sum(abs(Grads.beta{i})));
    end
end

%% 

clear J accuracy

GDParams.n_cycles = 2;
% GDParams.eta_min = 0.000600994000000000;
% GDParams.eta_min = 0.00065;
% GDParams.eta_max = 0.08;
GDParams.eta_min = 1e-5;
GDParams.eta_max = 0.1;
GDParams.l = 0;
GDParams.t = 0;
GDParams.n_batch = 100;
GDParams.n_s = floor(5 * size(X.train, 2) / GDParams.n_batch);
GDParams.n_epochs = floor(GDParams.n_batch * GDParams.n_cycles * 2 *GDParams.n_s / size(X.train, 2));
GDParams.start_epoch = 1;
% GDParams.lambda = 0.0027;
GDParams.lambda = 0.005;

accuracy.train = zeros(1, GDParams.n_epochs + 1);
accuracy.validation = zeros(1, GDParams.n_epochs + 1);
accuracy.test = zeros(1, GDParams.n_epochs + 1);

t = 0;

[l.train(1), J.train(1)]  = ComputeCost(X.train, Y.train, NetParams, GDParams.lambda); 
[l.val(1), J.val(1)] = ComputeCost(X.val, Y.val, NetParams, GDParams.lambda); 
[l.test(1), J.test(1)] = ComputeCost(X.test, Y.test, NetParams, GDParams.lambda);

accuracy.train(1) = ComputeAccuracy(X.train, y.train, NetParams);
accuracy.validation(1) = ComputeAccuracy(X.val, y.val, NetParams);
accuracy.test(1) = ComputeAccuracy(X.test, y.test, NetParams);

[Ws, bs, J, l, accuracy, t, eta] = MiniBatchGDCyclical(X, Y, y, GDParams, NetParams, J, l, accuracy, t);

%%

% Plot accuracy

figure; 

% plottitle = ["accuracy vs update step plot, M=", M];

% title(join(plottitle, ""), 'Interpreter','tex');

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
