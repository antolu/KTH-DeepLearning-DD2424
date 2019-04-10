function [gradW, gradb] = ComputeGradients(X, Y, P, H, NetParams, lambda)

batchSize = size(X, 2);
k = max(size(NetParams.W));
H = {{X}; H}; H = cat(2, H{:});

gradW = cell(1, max(k));
gradb = cell(1, max(k));

Gbatch = -(Y - P);

dLdW = (1 / batchSize) * Gbatch * H{end}';
dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);

gradW{end} = dLdW + 2 * lambda * NetParams.W{end};
gradb{end} = dLdb;

for i=k-1:-1:1
    Gbatch = NetParams.W{i+1}' * Gbatch;
    Ind = H{i+1};
    Ind(Ind>0) = 1;
    Gbatch = Gbatch .* Ind;
    
    dLdW = (1 / batchSize) * Gbatch * H{i}';
    dLdb = (1 / batchSize) * Gbatch * ones(batchSize, 1);
    
    gradW{i} = dLdW + 2 * lambda * NetParams.W{i};
    gradb{i} = dLdb;
end

end