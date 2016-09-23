function k = fn_ia(x,pred,obs)
% fitness function: Interrater Agreement
% input: weights vector, prediction matrix, observations vector
% output: interrater agreement (minimize for maximizing diversity)
    x    = round(x/sum(x),3);
    ind  = find(x);
    pred = round(pred(1:end,ind));
    p    = 1/(size(pred,2)*size(pred,1))*...
            sum(size(pred,1)*ones(1,size(pred,2))-...
                sum(abs(bsxfun(@minus,pred,obs)),1));
    rho  = size(pred,2)-sum(abs(bsxfun(@minus,pred,obs)),2);
    k    = 1-((1/size(pred,2)*sum(rho'*(size(pred,2)*...
            ones(1,size(pred,1))'-rho)))/...
            (size(pred,1)*(size(pred,2)-1)*p*(1-p)));
end