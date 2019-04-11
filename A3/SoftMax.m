function P = SoftMax(s)

P = exp(s) ./ (sum(exp(s)));

end