addpath datasets\cifar-10

%% Standard datasets

[Xtrain, Ytrain, ytrain] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval] = LoadBatch('data_batch_2.mat');
[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

%% Full datasets

[Xtrain1, Ytrain1, ytrain1] = LoadBatch('data_batch_1.mat');
[Xtrain2, Ytrain2, ytrain2] = LoadBatch('data_batch_2.mat');
[Xtrain3, Ytrain3, ytrain3] = LoadBatch('data_batch_3.mat');
[Xtrain4, Ytrain4, ytrain4] = LoadBatch('data_batch_4.mat');
[Xtrain5, Ytrain5, ytrain5] = LoadBatch('data_batch_5.mat');

Xtrain = [Xtrain1, Xtrain2, Xtrain3, Xtrain4, Xtrain5];
Ytrain = [Ytrain1, Ytrain2, Ytrain3, Ytrain4, Ytrain5];
ytrain = [ytrain1; ytrain2; ytrain3; ytrain4; ytrain5];

Xval = Xtrain(:, end-999:end);
Yval = Ytrain(:, end-999:end);
yval = ytrain(end-999:end);

Xtrain = Xtrain(:, 1:end-1000);
Ytrain = sparse(Ytrain(:, 1:end-1000));
ytrain = ytrain(1:end-1000);

[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

%% Initialize W, b

rng(400);
W = 0.01 * randn(10, 3072);
b = 0.01 * randn(10, 1);

MAX_EPOCH = 500;

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
    
    clear J_train J_test J_val l_train l_val l_test

    Wstar = cell(MAX_EPOCH);
    bstar = cell(MAX_EPOCH);
    accuracy.train = zeros(1, MAX_EPOCH);
    accuracy.validation = zeros(1, MAX_EPOCH);
    accuracy.test = zeros(1, MAX_EPOCH);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    
    [l_train(1), J_train(1)]  = ComputeCost(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
    [l_val(1), J_val(1)] = ComputeCost(Xval, Yval, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
    [l_test(1), J_test(1)] = ComputeCost(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

    accuracy.train(1) = ComputeAccuracy(Xtrain, ytrain, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy(Xval, yval, Ws, bs);
    accuracy.test(1) = ComputeAccuracy(Xtest, ytest, Ws, bs);

    for epoch=1:MAX_EPOCH
        GDParams{i}.n_epochs = epoch;

        [Ws, bs] = MiniBatchGD(Xtrain, Ytrain, GDParams{i}, Ws, bs);

        Wstar{epoch} = Ws; bstar{epoch} = bs;
        [l_train(epoch+1), J_train(epoch+1)]  = ComputeCost(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
        [l_val(epoch+1), J_val(epoch+1)] = ComputeCost(Xval, Yval, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
        [l_test(epoch+1), J_test(epoch+1)] = ComputeCost(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

        accuracy.train(epoch+1) = ComputeAccuracy(Xtrain, ytrain, Ws, bs);
        accuracy.validation(epoch+1) = ComputeAccuracy(Xval, yval, Ws, bs);
        accuracy.test(epoch+1) = ComputeAccuracy(Xtest, ytest, Ws, bs);

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
    hold off

    legend('training cost', 'validation cost', 'test cost');

    xlabel('epoch');
    ylabel('cost');
    axis([0, MAX_EPOCH, 0.75 * min(J.test), 1.1 * max(J.test)]);

    plotname = ["plots/cost_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

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
    axis([0, MAX_EPOCH, 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

    plotname = ["plots/accuracy_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
end
