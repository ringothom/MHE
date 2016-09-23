function d = fn_diff(x,pred,obs)
    % fitness function: Difficulty
    % input: weights vector, prediction matrix, observations vector
    % output: difficulty (minimize for maximizing diversity)
    x    = round(x/sum(x),3);
    ind  = find(x);
    pred = round(pred(1:end,ind));
    % proportion of correct classifiers
    k = (transpose(size(pred,2)*ones(1,size(pred,1)))...
        -sum(abs(bsxfun(@minus,pred,obs)),2))/size(pred,2);
    d = var(k);
end