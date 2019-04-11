function [Wstar, bstar, J, l, accuracy, t, eta] = MiniBatchGDCyclical(X, Y, y, GDParams, NetParams, J, l, accuracy, t)

N = size(X.train, 2);

setbreak = 0;
epoch = 0;

ALPHA = 0.9;

Wstar = cell(GDParams.n_cycles, numel(NetParams.W));
bstar = cell(GDParams.n_cycles, numel(NetParams.b));

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
                        
            Wstar(GDParams.l, :) = NetParams.W;
            bstar(GDParams.l, :) = NetParams.b;
            
            NetParams.Wstar = Wstar;
            NetParams.bstar = bstar;
            
            BNParams.calculate_mean = 0;
%             accuracy.train_ensemble(GDParams.l) = ComputeMajorityVoteAccuracy(X.train, y.train, NetParams, 'BNParams', BNParams);
%             accuracy.validation_ensemble(GDParams.l) = ComputeMajorityVoteAccuracy(X.val, y.val, NetParams, 'BNParams', BNParams);
%             accuracy.test_ensemble(GDParams.l) = ComputeMajorityVoteAccuracy(X.test, y.test, NetParams, 'BNParams', BNParams);
        end
        
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
        
        if NetParams.use_bn
            [P, BNParams] = ForwardPass(Xbatch, NetParams);
            
            if exist('mu_av', 'var') == 1
                for i=1:numel(mu_av)
                    mu_av{i} = NetParams.alpha * mu_av{i} + (1 - NetParams.alpha) * BNParams.mu{i};
                    v_av{i} = NetParams.alpha * v_av{i} + (1 - NetParams.alpha) * BNParams.v{i};
                end
            else
                mu_av = BNParams.mu;
                v_av = BNParams.v;
            end
            
            BNParams.calculate_mean = 0;
            
            Grads = BackwardPass(Xbatch, Ybatch, P, BNParams.X, NetParams, GDParams.lambda, 'BNParams', BNParams);
            
            for i=1:numel(NetParams.W)
                NetParams.W{i} = NetParams.W{i} - eta_t * Grads.W{i};
                NetParams.b{i} = NetParams.b{i} - eta_t * Grads.b{i};
            end
            
            for i=1:numel(NetParams.W)-1
                NetParams.gammas{i} = NetParams.gammas{i} - eta_t * Grads.gamma{i};
                NetParams.betas{i} = NetParams.betas{i} - eta_t * Grads.beta{i};
            end
        else
            [P, H] = EvaluatekLayer(Xbatch, NetParams);
            
            [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, P, H, NetParams, GDParams.lambda);
            
            for i=1:numel(NetParams.W)
                NetParams.W{i} = NetParams.W{i} - eta_t * gradW{i};
                NetParams.b{i} = NetParams.b{i} - eta_t * gradb{i};
            end
        end
        
        % Dropout
%         if NetParams.use_dropout
%             U = (rand(size(H)) < NetParams.P) / NetParams.P;
%             H = H .* U;
% 
%             S = EvaluateClassifier(H, W{2}, b{2});
%             P = SoftMax(S);
%         end
        
        t = t + 1;
    end
    if setbreak == 1
        break;
    end
    
    if NetParams.use_bn
        clear BNParams;
        BNParams.calculate_mean = 0;
        BNParams.mu = mu_av;
        BNParams.v = v_av;
    else
        BNParams.calculate_mean = 1;
    end

%     [l.train(epoch + 1), J.train(epoch + 1)]  = ComputeCost(X.train, Y.train, NetParams, GDParams.lambda, 'BNParams', BNParams); 
%     [l.val(epoch + 1), J.val(epoch + 1)] = ComputeCost(X.val, Y.val, NetParams, GDParams.lambda, 'BNParams', BNParams); 
%     [l.test(epoch + 1), J.test(epoch + 1)] = ComputeCost(X.test, Y.test, NetParams, GDParams.lambda, 'BNParams', BNParams); 

    accuracy.train(epoch + 1) = ComputeAccuracy(X.train, y.train, NetParams, 'BNParams', BNParams);
    accuracy.validation(epoch + 1) = ComputeAccuracy(X.val, y.val, NetParams, 'BNParams', BNParams);
    accuracy.test(epoch + 1) = ComputeAccuracy(X.test, y.test, NetParams, 'BNParams', BNParams);
    epoch
end

end