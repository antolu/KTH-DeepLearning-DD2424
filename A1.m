addpath datasets\cifar-10

[Xtrain, Ytrain, trainy] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval] = LoadBatch('data_batch_2.mat');
[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

%% Initialize W, b

rng(400);
W = 0.01 * randn(10, 3072);
b = 0.01 * randn(10, 1);
lambda = 0;

MAX_EPOCH = 40;

%% Evaluate

P = EvaluateClassifier(Xtrain(:, 1:100), W, b)

J = ComputeCost(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda)

[gradW, gradb] = ComputeGradients(Xtrain(:, 1:100), Ytrain(:, 1:100), P, W, lambda);

[ngradb, ngradW] = ComputeGradsNum(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda, 1e-6);

correlation = sum(abs(ngradW - gradW)) / max(1e-6, sum(abs(ngradW)) + sum(abs(gradW)));

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

    for epoch=1:MAX_EPOCH
        GDParams{i}.n_epochs = epoch;

        [Ws, bs] = MiniBatchGD(Xtrain, Ytrain, GDParams{i}, Ws, bs);

        Wstar{epoch} = Ws; bstar{epoch} = bs;
        [l_train(epoch) J_train(epoch)]  = ComputeCost(Xtrain, Ytrain, Ws, bs, lambda); J.train = J_train; l.train = l_train;
        [l_val(epoch) J_val(epoch)] = ComputeCost(Xval, Yval, Ws, bs, lambda); J.val = J_val; l.val = l_val;
        [l_test(epoch) J_test(epoch)] = ComputeCost(Xtest, Ytest, Ws, bs, lambda); J.test = J_test; l.test = l_test;

        accuracy.train(epoch) = ComputeAccuracy(Xtrain, trainy, Ws, bs);
        accuracy.validation(epoch) = ComputeAccuracy(Xval, yval, Ws, bs);
        accuracy.test(epoch) = ComputeAccuracy(Xtest, ytest, Ws, bs);

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

    plottitle = ["cost vs epoch plot, $\eta=", GDParams{i}.eta, ", \lambda=", GDParams{i}.lambda, "$"];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot(1:MAX_EPOCH, J.train, 'LineWidth', 1.2);
    plot(1:MAX_EPOCH, J.val, 'LineWidth', 1.2);
    plot(1:MAX_EPOCH, J.test, 'LineWidth', 1.2);
    hold off

    legend('training cost', 'validation cost', 'test cost');

    xlabel('epoch');
    ylabel('cost');
    axis([0, 40, 0.75 * min(J.test), 1.1 * max(J.test)]);

    plotname = ["plots/cost_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
    
    % Plot loss

    figure; 

    plottitle = ["loss vs epoch plot, $\eta=", GDParams{i}.eta, ", \lambda=", GDParams{i}.lambda, "$"];

    title(join(plottitle, ""), 'Interpreter','tex');

    hold on
    plot(1:MAX_EPOCH, l.train, 'LineWidth', 1.2);
    plot(1:MAX_EPOCH, l.val, 'LineWidth', 1.2);
    plot(1:MAX_EPOCH, l.test, 'LineWidth', 1.2);
    hold off

    legend('training loss', 'validation loss', 'test loss');

    xlabel('epoch');
    ylabel('loss');
    axis([0, 40, 0.75 * min(l.test), 1.1 * max(l.test)]);

    plotname = ["plots/loss_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
end
