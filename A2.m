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
