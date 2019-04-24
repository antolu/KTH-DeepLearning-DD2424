function [Wstar, bstar, J, l, accuracy, t, eta] = MiniBatchGDCyclical(X, Y, y, GDParams, NetParams, J, l, accuracy, t)

N = size(X.train, 2);

setbreak = 0;
epoch = 0;

Wstar = cell(GDParams.n_cycles, 2);
bstar = cell(GDParams.n_cycles, 2);

NetParams.lambda = GDParams.lambda;

old_l = 0;

while 1
    epoch = epoch + 1;
    
    random = randperm(size(X.train,2));
    X.train = X.train(:, random);
    Y.train = Y.train(:, random);
    
    for j=1:N/GDParams.n_batch
        
        GDParams.l = floor(t / (2 * GDParams.n_s));
        
        % Ensemble
        if GDParams.l > old_l 
            old_l = GDParams.l;
            disp(GDParams.l);
                        
            % Wstar(GDParams.l, :) = W;
            % bstar(GDParams.l, :) = b;
            
            % accuracy.train_ensemble(GDParams.l) = ComputeMajorityVoteAccuracy(X.train, y.train, Wstar, bstar);
            % accuracy.validation_ensemble(GDParams.l) = ComputeMajorityVoteAccuracy(X.val, y.val, Wstar, bstar);
            % accuracy.test_ensemble(GDParams.l) = ComputeMajorityVoteAccuracy(X.test, y.test, Wstar, bstar);
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

        [P, H, mu, v, S, S_hat] = EvaluatekLayer(Xbatch, NetParams);

        if exist('mu_av', 'var')
            for i=1:numel(mu)
                mu_av{i} = NetParams.alpha * mu_av{i} + (1 - NetParams.alpha) * mu{i};
                v_av{i} = NetParams.alpha * v_av{i} + (1 - NetParams.alpha) * v{i};
            end
        else
            mu_av = mu; v_av = v;
        end

        Grads = ComputeGradients(Xbatch, Ybatch, P, H, NetParams, 'mean', mu, 'variance', v, 'S', S, 'Shat', S_hat);

        for i=1:numel(NetParams.W)
            NetParams.W{i} = NetParams.W{i} - eta_t * Grads.W{i};
            NetParams.b{i} = NetParams.b{i} - eta_t * Grads.b{i};
        end

        for i=1:numel(NetParams.W)-1
            NetParams.gamma{i} = NetParams.gamma{i} - eta_t * Grads.gamma{i};
            NetParams.beta{i} = NetParams.beta{i} - eta_t * Grads.beta{i};
        end

        t = t + 1;
    end
    if setbreak == 1
        break;
    end
    % [l_train(epoch + 1), J_train(epoch + 1)]  = ComputeCost2(X.train, Y.train, W, b, GDParams.lambda); 
    % [l_val(epoch + 1), J_val(epoch + 1)] = ComputeCost2(X.val, Y.val, W, b, GDParams.lambda); 
    % [l_test(epoch + 1), J_test(epoch + 1)] = ComputeCost2(X.test, Y.test, W, b, GDParams.lambda); 

    accuracy.train(epoch + 1) = ComputeAccuracy(X.train, y.train, NetParams, 'mean', mu_av, 'variance', v_av);
    accuracy.validation(epoch + 1) = ComputeAccuracy(X.val, y.val, NetParams, 'mean', mu_av, 'variance', v_av);
    accuracy.test(epoch + 1) = ComputeAccuracy(X.test, y.test, NetParams, 'mean', mu_av, 'variance', v_av);
    disp(epoch);
end

end