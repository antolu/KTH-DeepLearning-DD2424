function [Wstar, bstar] = MiniBatchGD2(X, Y, GDParams, W, b)

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

        H = EvaluateClassifier(Xbatch, W{1}, b{1}); H(H < 0) = 0;
        S = EvaluateClassifier(H, W{2}, b{2});
        P = SoftMax(S);

        [gradW, gradb] = ComputeGradients2(Xbatch, Ybatch, P, H, W, GDParams.lambda);

        W{1} = W{1} - GDParams.eta * gradW{1};
        b{1} = b{1} - GDParams.eta * gradb{1};
        
        W{2} = W{2} - GDParams.eta * gradW{2};
        b{2} = b{2} - GDParams.eta * gradb{2};
    end
    
%     GDParams.eta = 0.9 * GDParams.eta;
end

Wstar = W;
bstar = b;

end