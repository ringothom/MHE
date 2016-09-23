%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YOEA112
% Project Title: Implementation of Firefly Algorithm (FA) in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
%{
Copyright (c) 2015, Yarpiz (www.yarpiz.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
	  
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
%}
% Note: This is a re-design as a function developed by R.Thomschke. For the 
% original go to www.yarpiz.com.

function [BestSol] = fireflyalgorithm(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Firefly Algorithm
%
% input:    CostFunction    function to minimize
%           nVar            number of variables
%           lb              lower bound (scalar)
%           ub              upper bound (scalar)
%           MaxIt           maximal number of iterations
%           nPop            population size
%           impr            maximal number of iterations without
%                           improvement
%
% output:   BestSol         best position, best cost

%% Problem Definition

% nVar  Number of Decision Variables

VarSize=[1 nVar];       % Decision Variables Matrix Size

VarMin= lb;             % Decision Variables Lower Bound
VarMax= ub;             % Decision Variables Upper Bound

%% Firefly Algorithm Parameters

%MaxIt Maximum Number of Iterations

%nPop  Number of Fireflies (Swarm Size)

gamma=1;            % Light Absorption Coefficient

beta0=2;            % Attraction Coefficient Base Value

alpha=0.2;          % Mutation Coefficient

alpha_damp=0.98;    % Mutation Coefficient Damping Ratio

delta=0.05*(VarMax-VarMin);     % Uniform Mutation Range

m=2;

if isscalar(VarMin) && isscalar(VarMax)
    dmax = (VarMax-VarMin)*sqrt(nVar);
else
    dmax = norm(VarMax-VarMin);
end

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Cost=[];

% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Fireflies

parfor i = 1:nPop
    pop (i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop (i).Cost=feval(CostFunction,(pop (i).Position));
end
[~,ind_BestSol]=min([pop.Cost]);
BestSol     = pop(ind_BestSol);


% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
BestCostPrevious = Inf;
r = 0;
it = 1;

%% Firefly Algorithm Main Loop

while it <= MaxIt && r <= impr
    
    newpop=repmat(firefly,nPop,1);
    newsol=repmat(firefly,nPop,1);
    [newsol(:).Cost] = deal(Inf);
    for i=1:nPop
        newpop (i).Cost = inf;
        positionNow = pop(i).Position;
        costNow = pop(i).Cost;
        parfor j=1:nPop
            if pop (j).Cost < costNow
                rij=norm (positionNow-pop (j).Position)/dmax;
                beta=beta0*exp(-gamma*rij^m);
                e=times(delta,unifrnd(-1,+1,VarSize));
                
                newsol (j).Position = pop (j).Position ...
                                + beta*rand (VarSize).*(positionNow - pop (j).Position) ...
                                + alpha*e;
                
                newsol (j).Position=max(newsol (j).Position,VarMin);
                newsol (j).Position=min(newsol (j).Position,VarMax);
                
                newsol (j).Cost=feval(CostFunction,(newsol (j).Position));
            end
        end
        
        [~,ind_BestNewsol]=min([newsol.Cost]);
        if newsol (ind_BestNewsol).Cost <= newpop (i).Cost
            newpop(i) = newsol (ind_BestNewsol);
            if newpop (i).Cost<=BestSol.Cost
                BestSol=newpop(i);
            end
        end
        
    
    % Merge
    mergepop=[pop
         newpop];  % #ok
    
    % Sort
    [~, SortOrder]=sort([mergepop.Cost]);
    mergepop=mergepop(SortOrder);
    
    % Truncate
    pop=mergepop(1:nPop);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;
    
    
    if BestCostPrevious - BestCost(it) > 1e-6
        r = 0;
    else
        r = r+1;  
    end
    BestCostPrevious = BestCost(it);
    it = it+1;
end
end