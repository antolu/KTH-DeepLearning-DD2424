addpath datasets/cifar-10

[Xtrain, Ytrain, trainy] = LoadBatch('data_batch_1.mat');
[Xval, Yval, yval] = LoadBatch('data_batch_2.mat');
[Xtest, Ytest, ytest] = LoadBatch('test_batch.mat');

%% Initialize W, b

rng(400);
W = 0.01 * randn(10, 3072);
b = 0.01 * randn(10, 1);

MAX_EPOCH = 100;
lambda = 2;

%% Evaluate

lambda = 0;

J = ComputeCost(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda)

[gradW, gradb] = ComputeGradients(Xtrain(:, 1:100), Ytrain(:, 1:100), b, W, lambda);

[ngradb, ngradW] = ComputeGradsNum(Xtrain(:, 1:100), Ytrain(:, 1:100), W, b, lambda, 1e-6);

correlation = sum(abs(ngradW - gradW)) / max(1e-6, sum(abs(ngradW)) + sum(abs(gradW)));

%% 

GDParams = cell(4);

GDParams{1}.eta = 0.001;
GDParams{1}.n_batch = 100;
GDParams{1}.n_epochs = MAX_EPOCH;
GDParams{1}.start_epoch = 1;
GDParams{1}.lambda = 0;

GDParams{2}.eta = 0.01;
GDParams{2}.n_batch = 100;
GDParams{2}.n_epochs = MAX_EPOCH;
GDParams{2}.start_epoch = 1;
GDParams{2}.lambda = 0;

GDParams{3}.eta = 0.001;
GDParams{3}.n_batch = 100;
GDParams{3}.n_epochs = MAX_EPOCH;
GDParams{3}.start_epoch = 1;
GDParams{3}.lambda = 0.1;
clear J

for i=1:3
    
    clear J_train J_test J_val l_val l_train l_test

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

    accuracy.train(1) = ComputeAccuracy(Xtrain, trainy, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy(Xval, yval, Ws, bs);
    accuracy.test(1) = ComputeAccuracy(Xtest, ytest, Ws, bs);

    for epoch=1:MAX_EPOCH
        GDParams{i}.n_epochs = epoch;

        [Ws, bs] = MiniBatchGD(Xtrain, Ytrain, GDParams{i}, Ws, bs);

        Wstar{epoch} = Ws; bstar{epoch} = bs;
        [l_train(epoch+1), J_train(epoch+1)]  = ComputeCost(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); J.train = J_train; l.train = l_train;
        [l_val(epoch+1), J_val(epoch+1)] = ComputeCost(Xval, Yval, Ws, bs, GDParams{i}.lambda); J.val = J_val; l.val = l_val;
        [l_test(epoch+1), J_test(epoch+1)] = ComputeCost(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); J.test = J_test; l.test = l_test;

        accuracy.train(epoch+1) = ComputeAccuracy(Xtrain, trainy, Ws, bs);
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
    axis([0, MAX_EPOCH, 0.8 * min(accuracy.train), 1.1 * max(accuracy.train)]);

    plotname = ["plots/accuracy_lambda", GDParams{i}.lambda, "_eta", GDParams{i}.eta, ".eps"];

    saveas(gca, join(plotname, ""), 'epsc');
    
    close all;
end



function [l, J] = ComputeCost(X, Y, W, b, lambda)
% Computes the SVM loss for the classifier
%   X - The data
%   Y - The one-hot labels
%   W - The network weights
%   b - The network bias weights
%   lambda - L2 regularisation weight

    numberOfSamples = size(X, 2);

    S = W * X + b * ones(1, size(X, 2));

    l = 0;

    for i=1:numberOfSamples
        pos = find(Y(:, i)==1);
        sy = S(pos, i);
        s = S(:, i) - sy;
        s = s + 1;
        s(s<0) = 0;
        k = sum(s) - 1;
        l = l + k;
    end

    l = l / numberOfSamples;

    regularisation = lambda * sum(sum(W.^2));

    J = l + regularisation;

end



function acc = ComputeAccuracy(X, y, W, b)
% Computes the accuracy of the single layer perceptron
%   X - The data
%   y - The labels
%   W - Layer weights
%   b - Bias weights
    
    numberOfSamples = size(X, 2);

    P = SoftMax(EvaluateClassifier(X, W, b));

    V = P==max(P);
    matches = V(sub2ind(size(V), y', 1:size(y)));

    acc = sum(matches);

    acc = acc / numberOfSamples;

end



function [gradW, gradb] = ComputeGradients(X, Y, b, W, lambda)
% Computes the gradients of W and b for a single layer perceptron
% with SVM loss
%   X - The data
%   Y - The targets
%   W - The current weights
%   b - The bias weights
%   lambda - L2 regularisation weight
%________________________________________________________________


    batchSize = size(X, 2);

    S = W * X + b * ones(1, size(X, 2));

    Sy = S .* Y;

    Gbatch = heaviside(S - sum(Sy) + 1);

    for i=1:batchSize
        pos = find(Y(:, i) == 1);
        Gbatch(pos, i) = -sum(Gbatch(:, i)) + Gbatch(pos, i);
    end

    dLdW = (1 / batchSize) * Gbatch * X';
    dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

    gradW = dLdW + 2 * lambda * W;
    gradb = dLdb;

end



function S = EvaluateClassifier(X, W, b)
% Evaluates the single layer perceptron
%   X - The data
%   W - The layer weights
%   b - The bias weights
%______________________________________

    S = W * X + b * ones(1, size(X, 2));

end



function [Wstar, bstar] = MiniBatchGD(X, Y, GDParams, W, b)
% Mini-batch gradient descent algorithm
%   X - The data
%   Y - One-hot labels
%   GDParams - structure of parameters
%   W - Network weights
%   b - Network bias weights

    N = size(X, 2);

    n_epochs = GDParams.n_epochs;

    for i=GDParams.start_epoch:n_epochs
        random = randperm(size(X,2));
    %     X = X(:, random);
    %     Y = Y(:, random);

        for j=1:N/GDParams.n_batch
            j_start = (j-1) * GDParams.n_batch + 1;
            j_end = j * GDParams.n_batch;
            inds = j_start:j_end;

            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);

            [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, b, W, GDParams.lambda);

            W = W - GDParams.eta * gradW;
            b = b - GDParams.eta * gradb;
        end

    %     GDParams.eta = 0.9 * GDParams.eta;
    end

    Wstar = W;
    bstar = b;

end



function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)
% Computes numerical gradients

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    [~, c] = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        [~, c2] = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)   

        W_try = W;
        W_try(i) = W_try(i) + h;
        [~, c2] = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c) / h;
    end

end