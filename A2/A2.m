addpath ../datasets/cifar-10
addpath ..
mkdir plots

[X.train, Y.train, y.train] = LoadBatchZeroMean('data_batch_1.mat');
[X.val, Y.val, y.val] = LoadBatchZeroMean('data_batch_2.mat');
[X.test, Y.test, y.test] = LoadBatchZeroMean('test_batch.mat');


%% Initialize W, b

D = size(X.train, 1);
M = 50;
C = 10;

dimensions = [D, M, C];

% rng(400);
[W, b] = XavierInitialise(dimensions);

MAX_EPOCH = 1000;

lambda = 2;

%% Evaluate

[P, H] = EvaluateClassifier(X.train(:, 1:100), W, b);

J = ComputeCost(X.train(:, 1:100), Y.train(:, 1:100), W, b, lambda)

[gradW, gradb] = ComputeGradients(X.train(:, 1:100), Y.train(:, 1:100), P, H, W, lambda);

[ngradb, ngradW] = ComputeGradsNum(X.train(:, 1:100), Y.train(:, 1:100), W, b, lambda, 1e-6);

rel_error(1) = sum(abs(ngradW{1} - gradW{1})) / max(1e-6, sum(abs(ngradW{1})) + sum(abs(gradW{1})));
rel_error(2) = sum(abs(ngradW{2} - gradW{2})) / max(1e-6, sum(abs(ngradW{2})) + sum(abs(gradW{2})));

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

clear J;

for i=1:size(GDParams)
    
    Wstar = cell(MAX_EPOCH, 2);
    bstar = cell(MAX_EPOCH, 2);
    
    l.train = zeros(1, GDParams{i}.n_epochs + 1); 
    l.val = zeros(1, GDParams{i}.n_epochs + 1); 
    l.test = zeros(1, GDParams{i}.n_epochs + 1);
    
    J.train = zeros(1, GDParams{i}.n_epochs + 1);
    J.val = zeros(1, GDParams{i}.n_epochs + 1);
    J.test = zeros(1, GDParams{i}.n_epochs + 1);
    
    accuracy.train = zeros(1, GDParams{i}.n_epochs + 1);
    accuracy.validation = zeros(1, GDParams{i}.n_epochs + 1);
    accuracy.test = zeros(1, GDParams{i}.n_epochs + 1);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    t = 0;
    
    [l.train(1), J.train(1)]  = ComputeCost(X.train, Y.train, Ws, bs, GDParams{i}.lambda); 
    [l.val(1), J.val(1)] = ComputeCost(X.val, Y.val, Ws, bs, GDParams{i}.lambda); 
    [l.test(1), J.test(1)] = ComputeCost(X.test, Y.test, Ws, bs, GDParams{i}.lambda); 

    accuracy.train(1) = ComputeAccuracy(X.train, y.train, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy(X.val, y.val, Ws, bs);
    accuracy.test(1) = ComputeAccuracy(X.test, y.test, Ws, bs);

    [Ws, bs, J, l, accuracy, t, eta] = MiniBatchGD(X, Y, y, GDParams{i}, Ws, bs, J, l, accuracy, t);

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
