function [Wstar, bstar, J, l, accuracy, t, eta] = MiniBatchGDBestEta(X, Y, y, GDParams, W, b, J, l, accuracy, t)

N = size(X.train, 2);

J_train = J.train; l_train = l.train;

setbreak = 0;
epoch = 0;
p = 0.8;

Wstar = cell(GDParams.n_cycles, 2);
bstar = cell(GDParams.n_cycles, 2);

old_l = 0;

while 1
    epoch = epoch + 1;
    
    random = randperm(size(X,2));
    X = X(:, random);
    Y = Y(:, random);
    
    for j=1:N/GDParams.n_batch
        
        GDParams.l = floor(t / (2 * GDParams.n_s));
        
        % Ensemble
        if GDParams.l > old_l 
            old_l = GDParams.l
                        
            Wstar(GDParams.l, :) = W;
            bstar(GDParams.l, :) = b;
            
        end
        
        if t >= GDParams.n_s
            setbreak = 1;
            break;
        end
        
        if mod(t, 100)==0
            t
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
        
        % Dropout
        % U = (rand(size(H)) < p) / p;
        % H = H .* U;
        
        [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, P, H, W, GDParams.lambda);

        W{1} = W{1} - eta_t * gradW{1};
        b{1} = b{1} - eta_t * gradb{1};
        
        W{2} = W{2} - eta_t * gradW{2};
        b{2} = b{2} - eta_t * gradb{2};
                
        accuracy.train(t + 1) = ComputeAccuracy(X.train, y.train, W, b);
        J.train(t + 1) = ComputeCost(X.train, Y.train, W, b, GDParams.lambda);

        t = t + 1;
    end
    if setbreak == 1
        break;
    end
    epoch
end

end