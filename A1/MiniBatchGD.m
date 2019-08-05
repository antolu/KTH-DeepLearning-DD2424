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
