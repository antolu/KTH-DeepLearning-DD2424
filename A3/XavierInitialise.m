function [W, b] = XavierInitialise(C, D, dimensions)
% Initialise
% dimensions should be in format [D, M, C]

k = max(size(dimensions)) + 1;

dimensions = [D, dimensions, C];

W = cell(1, k);
b = cell(1, k);

for i=1:k
    W{i} = 1/sqrt(dimensions(i)/2) * randn(dimensions(i+1), dimensions(i));
    b{i} = zeros(dimensions(i+1), 1);
end

end