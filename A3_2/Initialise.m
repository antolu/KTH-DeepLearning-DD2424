function [W, b] = Initialise(C, D, dimensions, varargin)
% Initialise
% dimensions should be in format [D, M, C]

[initialisation, sigma] = parse_inputs(varargin);

k = max(size(dimensions)) + 1;

dimensions = [D, dimensions, C];

W = cell(1, k);
b = cell(1, k);

rng(43);

if strcmp(initialisation, 'xavier')
    for i=1:k
        W{i} = 1/sqrt(dimensions(i)) * randn(dimensions(i), dimensions(i+1))';
        b{i} = zeros(1, dimensions(i+1))';
    end
elseif strcmp(initialisation, 'he')
    for i=1:k
        W{i} = sqrt(2/dimensions(i)) * randn(dimensions(i), dimensions(i+1))';
        b{i} = zeros(1, dimensions(i+1))';
    end
elseif strcmp(initialisation, 'normal')
    for i=1:k
        W{i} = sigma * randn(dimensions(i+1), dimensions(i));
        b{i} = sigma * randn(dimensions(i+1), 1);
    end
else
    error("Unrecognised option");
end

end

function [initialisation, sigma] = parse_inputs(inputs)

% Set defaults
initialisation = 'uniform';
sigma = 0.01;

% Go through option pairs
for a = 1:2:numel(inputs)
    switch lower(inputs{a})
        case 'initialisation'
            initialisation = inputs{a+1};
        case 'sigma'
            sigma = inputs{a+1};
        otherwise
            error('Input option %s not recognized', inputs{a});
    end
end

return

end