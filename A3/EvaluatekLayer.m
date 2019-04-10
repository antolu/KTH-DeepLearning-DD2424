function [P, H] = EvaluatekLayer(X, NetParams, varargin) 

batchNormalise = parse_inputs(varargin);

k = numel(NetParams.W);

[P2, H2] = Evaluate2Layer(X, NetParams.W, NetParams.b);

if not(batchNormalise)
    H = cell(1, k);
    
    H{1} = X;
    
    for i=1:k-1
        H{i+1} = EvaluateClassifier(H{i}, NetParams.W{i}, NetParams.b{i}); 
        H{i+1}(H{i+1} < 0) = 0;
    end
        
    P = SoftMax(EvaluateClassifier(H{end}, NetParams.W{end}, NetParams.b{end}));
end

H = H(2:end);

end



function [batchNormalise] = parse_inputs(inputs)

% Set defaults
batchNormalise = 0;

% Go through option pairs
for a = 1:2:numel(inputs)
    switch lower(inputs{a})
        case 'batchnormalise'
            batchNormalise = inputs{a+1};
        otherwise
            error('Input option %s not recognized', inputs{a});
    end
end

return

end