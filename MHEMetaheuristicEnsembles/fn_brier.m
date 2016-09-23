function b = fn_brier(x,pred,obs)
    % fitness function: Brier Score
    % input: weights vector, prediction matrix, observations vector
    % output: Brier Score (minimize for maximizing accuracy)

    x = x/sum(x);
    weightPred = bsxfun(@times,pred,x);
    sumPred = sum(weightPred,2);
    b = 1/size(pred,1)*sum((sumPred-obs).^2);
end