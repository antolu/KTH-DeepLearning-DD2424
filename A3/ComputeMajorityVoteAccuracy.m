function acc = ComputeMajorityVoteAccuracy(X, y, NetParams)

numberOfSamples = size(X, 2);

matches = zeros(1, numberOfSamples);
votes = zeros(max(y), size(X, 2));

for i=1:size(NetParams.W, 1)
    if not(isempty(NetParams.W{i}))
        Params.W = NetParams.W(i, :);
        Params.b = NetParams.b(i, :);
        
        [P, H] = EvaluatekLayer(X, Params);

        V = P==max(P);
        votes = votes + V;
    end
end

finalVotes = votes==max(votes);

matches = finalVotes(sub2ind(size(finalVotes), y', 1:size(y)));

acc = sum(matches);

acc = acc / numberOfSamples;

end