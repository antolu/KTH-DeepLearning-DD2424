function [Wstar, bstar, J, l, accuracy, t, eta] = MiniBatchGD(X, Y, y, GDParams, W, b, J, l, accuracy, t)
% Mini-batch gradient descent algorithm
%   X - The data
%   Y - One-hot labels
%   y - labels
%   GDParams - structure of parameters
%   W - Network weights
%   b - Network bias weights
%   J - Cost structure
%   l - Loss structure
%   accuracy - accuracy structure
%   t - plot X-axis
%______________________________________________________
    
    
    N = size(X.train, 2);

    J_train = J.train; l_train = l.train;
    J_val = J.val; l_val = l.val;
    J_test = J.test; l_test = l.test;

    setbreak = 0;

    for epoch=GDParams.start_epoch:GDParams.n_epochs
        random = randperm(size(X,2));
        X = X(:, random);
        Y = Y(:, random);

        for j=1:N/GDParams.n_batch

            GDParams.l = floor(t / (2 * GDParams.n_s));
            if GDParams.l >= GDParams.n_cycles
                setbreak = 1;
                break;
            end

            if (t >= 2 * (GDParams.l) * GDParams.n_s) && (t <= (2 * (GDParams.l) + 1) * GDParams.n_s)
                eta_t = GDParams.eta_min + ((t - 2 * GDParams.l * GDParams.n_s) / GDParams.n_s) * (GDParams.eta_max - GDParams.eta_min);
            else 
                eta_t = GDParams.eta_max - ((t - (2 * GDParams.l + 1) * GDParams.n_s) / GDParams.n_s) * (GDParams.eta_max - GDParams.eta_min);
            end

            eta(t + 1) = eta_t;

            j_start = (j-1) * GDParams.n_batch + 1;
            j_end = j * GDParams.n_batch;
            inds = j_start:j_end;

            Xbatch = X.train(:, inds);
            Ybatch = Y.train(:, inds);

            [P, H] = EvaluateClassifier(Xbatch, W, b);

            [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, P, H, W, GDParams.lambda);

            W{1} = W{1} - eta_t * gradW{1};
            b{1} = b{1} - eta_t * gradb{1};

            W{2} = W{2} - eta_t * gradW{2};
            b{2} = b{2} - eta_t * gradb{2};

            t = t + 1;
        end
        if setbreak == 1
            break;
        end
        [l.train(epoch + 1), J.train(epoch + 1)]  = ComputeCost(X.train, Y.train, W, b, GDParams.lambda); 
        [l.val(epoch + 1), J.val(epoch + 1)] = ComputeCost(X.val, Y.val, W, b, GDParams.lambda); 
        [l.test(epoch + 1), J.test(epoch + 1)] = ComputeCost(X.test, Y.test, W, b, GDParams.lambda); 

        accuracy.train(epoch + 1) = ComputeAccuracy(X.train, y.train, W, b);
        accuracy.validation(epoch + 1) = ComputeAccuracy(X.val, y.val, W, b);
        accuracy.test(epoch + 1) = ComputeAccuracy(X.test, y.test, W, b);
        epoch
    end

    Wstar = W;
    bstar = b;

end