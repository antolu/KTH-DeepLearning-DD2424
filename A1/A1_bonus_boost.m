addpath datasets/cifar-10
addpath A1

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
    
    Wstar = cell(1, MAX_EPOCH);
    bstar = cell(1, MAX_EPOCH);
    
    l.train = zeros(1, MAX_EPOCH); 
    l.val = zeros(1, MAX_EPOCH); 
    l.test = zeros(1, MAX_EPOCH);
    
    J.train = zeros(1, MAX_EPOCH); 
    J.val = zeros(1, MAX_EPOCH); 
    J.test = zeros(1, MAX_EPOCH);
    
    accuracy.train = zeros(1, MAX_EPOCH); 
    accuracy.validation = zeros(1, MAX_EPOCH); 
    accuracy.test = zeros(1, MAX_EPOCH);

    Ws = W;
    bs = b;
    j = zeros(1, MAX_EPOCH);
    
    [l.train(1), J.train(1)]  = ComputeCost(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); 
    [l.val(1), J.val(1)] = ComputeCost(Xval, Yval, Ws, bs, GDParams{i}.lambda); 
    [l.test(1), J.test(1)] = ComputeCost(Xtest, Ytest, Ws, bs, GDParams{i}.lambda);

    accuracy.train(1) = ComputeAccuracy(Xtrain, ytrain, Ws, bs);
    accuracy.validation(1) = ComputeAccuracy(Xval, yval, Ws, bs);
    accuracy.test(1) = ComputeAccuracy(Xtest, ytest, Ws, bs);

    for epoch=1:MAX_EPOCH
        GDParams{i}.n_epochs = epoch;

        [Ws, bs] = MiniBatchGD(Xtrain, Ytrain, GDParams{i}, Ws, bs);

        Wstar{epoch} = Ws; bstar{epoch} = bs;
        [l.train(epoch+1), J.train(epoch+1)]  = ComputeCost(Xtrain, Ytrain, Ws, bs, GDParams{i}.lambda); 
        [l.val(epoch+1), J.val(epoch+1)] = ComputeCost(Xval, Yval, Ws, bs, GDParams{i}.lambda); 
        [l.test(epoch+1), J.test(epoch+1)] = ComputeCost(Xtest, Ytest, Ws, bs, GDParams{i}.lambda); 
        
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



function [l, J] = ComputeCost(X, Y, W, b, lambda)
% Computes the cross-entropy loss for the classifier
%   X - The data
%   Y - The one-hot labels
%   W - The network weights
%   b - The network bias weights
%   lambda - L2 regularisation weight

    numberOfSamples = size(X, 2);

    P = SoftMax(EvaluateClassifier(X, W, b));

    k = sum(Y .* P);
    l = sum(-log(k));

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



function [gradW, gradb] = ComputeGradients(X, Y, P, W, lambda)
% Computes the gradients of W and b for a single layer perceptron
%   X - The data
%   Y - The targets
%   P - The SoftMax values
%   W - The current weights
%   lambda - L2 regularisation weight
%________________________________________________________________

    batchSize = size(X, 2);

    Gbatch = -(Y - P);

    dLdW = (1 / batchSize) * Gbatch * X';
    dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

    k = 2 * lambda * W;

    gradW = dLdW + k;
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



function P = SoftMax(s)
% Computes the SoftMax values of the targets
%   s - targets
%________________________________________________________

    P = exp(s) ./ (sum(exp(s)));

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

            P = SoftMax(EvaluateClassifier(Xbatch, W, b));

            [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, P, W, GDParams.lambda);

            W = W - GDParams.eta * gradW;
            b = b - GDParams.eta * gradb;
        end

    %     GDParams.eta = 0.9 * GDParams.eta;
    end

    Wstar = W;
    bstar = b;

end
