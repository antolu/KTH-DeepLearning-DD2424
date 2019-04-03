function [X, Y, y] = LoadBatchZeroMean(filename, varargin)

A = load(filename);

X = double(A.data')./255;

meanX = mean(X, 2);
stdX = std(X, 0, 2);

X = X - repmat(meanX, [1, size(X, 2)]);
% X = X ./ repmat(stdX, [1, size(X, 2)]);

y = double(A.labels) + 1;

Y = zeros(10, 10000);

for i=1:10000
    Y(y(i), i) = 1;
end

end