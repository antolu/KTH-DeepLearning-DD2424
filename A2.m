addpath datasets\cifar-10

[Xtrain, Ytrain, trainy] = LoadBatchZeroMean('data_batch_1.mat');
[Xval, Yval, yval] = LoadBatchZeroMean('data_batch_2.mat');
[Xtest, Ytest, ytest] = LoadBatchZeroMean('test_batch.mat');

%% Initialize W, b

D = size(Xtrain, 1);
M = 50;
C = 10;

rng(400);
W = {1/sqrt(D) * randn(M, D), 1/sqrt(1) * randn(C, M)};
b = {zeros(M, 1), zeros(C, 1)};

MAX_EPOCH = 40;

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

GDParams = cell(4);

GDParams{1}.eta = 0.1;
GDParams{1}.n_batch = 100;
GDParams{1}.n_epochs = 40;
GDParams{1}.start_epoch = 1;
GDParams{1}.lambda = 0;

GDParams{2}.eta = 0.01;
GDParams{2}.n_batch = 100;
GDParams{2}.n_epochs = 40;
GDParams{2}.start_epoch = 1;
GDParams{2}.lambda = 0;

GDParams{3}.eta = 0.01;
GDParams{3}.n_batch = 100;
GDParams{3}.n_epochs = 40;
GDParams{3}.start_epoch = 1;
GDParams{3}.lambda = 0.1;

GDParams{4}.eta = 0.01;
GDParams{4}.n_batch = 100;
GDParams{4}.n_epochs = 40;
GDParams{4}.start_epoch = 1;
GDParams{4}.lambda = 1;

for i=1:4
    
    clear J_train J_test J_val

    Wstar = cell(MAX_EPOCH);
    bstar = cell(MAX_EPOCH);
    accuracy.train = zeros(1, MAX_EPOCH);
    accuracy.validation = zeros(1, MAX_EPOCH);
    accuracy.test = zeros(1, MAX_EPOCH);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    
    [l_train(1), J_train(1)]  = ComputeCost2(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
    [l_val(1), J_val(1)] = ComputeCost2(Xval, Yval, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
    [l_test(1), J_test(1)] = ComputeCost2(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

    accuracy.train(1) = ComputeAccuracy2(Xtrain, trainy, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy2(Xval, yval, Ws, bs);
    accuracy.test(1) = ComputeAccuracy2(Xtest, ytest, Ws, bs);

    for epoch=1:MAX_EPOCH
        GDParams{i}.n_epochs = epoch;

        [Ws, bs] = MiniBatchGD2(Xtrain, Ytrain, GDParams{i}, Ws, bs);

        Wstar{epoch} = Ws; bstar{epoch} = bs;
        [l_train(epoch+1), J_train(epoch+1)]  = ComputeCost2(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
        [l_val(epoch+1), J_val(epoch+1)] = ComputeCost2(Xval, Yval, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
        [l_test(epoch+1), J_test(epoch+1)] = ComputeCost2(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

        accuracy.train(epoch+1) = ComputeAccuracy2(Xtrain, trainy, Ws, bs);
        accuracy.validation(epoch+1) = ComputeAccuracy2(Xval, yval, Ws, bs);
        accuracy.test(epoch+1) = ComputeAccuracy2(Xtest, ytest, Ws, bs);

        epoch
        GDParams{i}.start_epoch = epoch;
    end
    
    % Save data
    dataname = ["data_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".mat"];

    save(join(dataname, ""), 'Wstar', 'bstar', 'J', 'accuracy', 'l');
    
    % Plot W

    for k=1:10
        im = reshape(Wstar{MAX_EPOCH}(k, :), 32, 32, 3);
        s_im{k} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{k} = permute(s_im{k}, [2, 1, 3]);
    end

    montage(s_im, 'Size', [2, 5]);

    montagename = ["plots/W_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(montagename, ""), 'epsc');

    % Plot cost

    figure; 

    plottitle = ["cost vs epoch plot, \eta=", GDParams{i}.eta, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot([0, 1:MAX_EPOCH], J.train, 'LineWidth', 1.2);
    plot([0, 1:MAX_EPOCH], J.val, 'LineWidth', 1.2);
    plot([0, 1:MAX_EPOCH], J.test, 'LineWidth', 1.2);
    

    legend('training cost', 'validation cost', 'test cost');

    xlabel('epoch');
    ylabel('cost');
    axis([0, MAX_EPOCH, 0.75 * min(J.test), 1.1 * max(J.test)]);

    plotname = ["plots/cost_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];
    hold off

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
    
    % Plot loss

    figure; 

    plottitle = ["loss vs epoch plot, \eta=", GDParams{i}.eta, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot([0, 1:MAX_EPOCH], l.train, 'LineWidth', 1.2);
    plot([0, 1:MAX_EPOCH], l.val, 'LineWidth', 1.2);
    plot([0, 1:MAX_EPOCH], l.test, 'LineWidth', 1.2);
    hold off

    legend('training loss', 'validation loss', 'test loss');

    xlabel('epoch');
    ylabel('loss');
    axis([0, MAX_EPOCH, 0.75 * min(l.test), 1.1 * max(l.test)]);

    plotname = ["plots/loss_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
    
    % Plot accuracy

    figure; 

    plottitle = ["accuracy vs epoch plot, \eta=", GDParams{i}.eta, ", \lambda=", GDParams{i}.lambda];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot([0, 1:MAX_EPOCH], accuracy.train, 'LineWidth', 1.2);
    plot([0, 1:MAX_EPOCH], accuracy.validation, 'LineWidth', 1.2);
    plot([0, 1:MAX_EPOCH], accuracy.test, 'LineWidth', 1.2);
    hold off

    legend('training accuracy', 'validation accuracy', 'test accuracy', 'Location','southeast');

    xlabel('epoch');
    ylabel('accuracy');
    axis([0, MAX_EPOCH, 0.8 * min(accuracy.test), 1.1 * max(accuracy.test)]);

    plotname = ["plots/accuracy_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
end
