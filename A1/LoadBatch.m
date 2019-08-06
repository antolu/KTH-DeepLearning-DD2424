function [X, Y, y] = LoadBatch(filename)

A = load(filename);

X = double(A.data')./255;

y = double(A.labels) + 1;

Y = zeros(10, 10000);

for i=1:10000
    Y(y(i), i) = 1;
end

end