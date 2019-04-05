function acc = ComputeMajorityVoteAccuracy(X, y, W, b)

xSize = size(X);
numberOfSamples = xSize(2);

matches = zeros(1, numberOfSamples);
votes = zeros(max(y), size(X, 2));

if size(W, 1)==1
    [P, H] = Evaluate2Layer(X, W, b);
    votes = P==max(P);
else
    for i=1:size(W, 1)
        if not(isempty(W{i}))
            [P, H] = Evaluate2Layer(X, W(i, :), b(i, :));

            V = P==max(P);
            votes = votes + V;
        end
    end
end

finalVotes = votes==max(votes);

matches = finalVotes(sub2ind(size(finalVotes), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

end