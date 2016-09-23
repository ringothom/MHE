% Project Code: YPEA107
% Project Title: Implementation of Differential Evolution (DE) in MATLAB
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

function [BestSol] = de(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Differential Evolution
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

VarSize = [1 nVar];   % Decision Variables Matrix Size

VarMin  = lb;          % Lower Bound of Decision Variables
VarMax  = ub;          % Upper Bound of Decision Variables

%% DE Parameters

beta_min = 0.2;   % Lower Bound of Scaling Factor
beta_max = 0.8;   % Upper Bound of Scaling Factor

pCR      = 0.2;   % Crossover Probability

%% Initialization

empty_individual.Position = [];
empty_individual.Cost     = [];

BestSol.Cost = inf;

pop = repmat(empty_individual,nPop,1);

parfor i=1:nPop
    pop (i).Position = unifrnd(VarMin,VarMax,VarSize);
    pop (i).Cost     = feval(CostFunction,pop (i).Position);   
end

% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop            = pop(SortOrder);

% Best Solution Ever Found
BestSol  = pop(1);

BestCost = zeros(MaxIt,1);

%% DE Main Loop
r  = 0;
it = 1;
BestCostPrevious = Inf;

while it <= MaxIt && r <= impr
    
    for i = 1:nPop
        
        x = pop (i).Position;
        
        A = randperm(nPop);
        
        A(A==i) = [];
        
        a = A(1);
        b = A(2);
        c = A(3);
        
        % Mutation
        % beta=unifrnd(beta_min,beta_max);
        beta = unifrnd(beta_min,beta_max,VarSize);
        y    = pop (a).Position+beta.*(pop (b).Position-pop (c).Position);
        y    = max(y, VarMin);
	y    = min(y, VarMax);
		
        % Crossover
        z  = zeros(size(x));
        j0 = randi([1 numel(x)]);
        for j = 1:numel(x)
            if j == j0 || rand <= pCR
                z(j) = y(j);
            else
                z(j) = x(j);
            end
        end
        
        NewSol.Position = z;
        NewSol.Cost     = CostFunction(NewSol.Position);
        
        if NewSol.Cost < pop (i).Cost
            pop(i) = NewSol;
            
            if pop (i).Cost < BestSol.Cost
               BestSol = pop(i);
            end
        end
        
    end
    
    % Update Best Cost
    BestCost(it) = BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    if BestCostPrevious - BestCost(it) > 1e-6
        r = 0;
    else
        r = r+1;  
    end
    BestCostPrevious = BestCost(it);
    it = it+1;
    
end
end