% Project Code: YPEA113
% Project Title: Biogeography-Based Optimization (BBO) in MATLAB
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


function [BestSol] = bbo(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Biogeography-Based Optimization 
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

VarMin  = lb;         % Decision Variables Lower Bound
VarMax  = ub;         % Decision Variables Upper Bound

%% BBO Parameters

% MaxIt=1000;          % Maximum Number of Iterations
% 
% nPop=50;            % Number of Habitats (Population Size)

KeepRate = 0.2;                      % Keep Rate
nKeep    = round(KeepRate*nPop);     % Number of Kept Habitats

nNew = nPop-nKeep;                % Number of New Habitats

% Migration Rates
mu     = linspace(1,0,nPop);      % Emmigration Rates
lambda = 1-mu;                    % Immigration Rates

alpha     = 0.9;
pMutation = 0.1;
sigma     = 0.02*(VarMax-VarMin);

%% Initialization

% Empty Habitat
habitat.Position = [];
habitat.Cost     = [];

% Create Habitats Array
pop = repmat(habitat,nPop,1);

% Initialize Habitats
parfor i = 1:nPop
    pop (i).Position = unifrnd(VarMin,VarMax,VarSize);
    pop (i).Cost     = feval(CostFunction,pop(i).Position);
end

% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop            = pop(SortOrder);

% Best Solution Ever Found
BestSol = pop(1);

% Array to Hold Best Costs
BestCost         = zeros(MaxIt,1);
BestCostPrevious = Inf;
r  = 0;
it = 1;

%% BBO Main Loop

while it <= MaxIt && r <= impr
    
    newpop = pop;
    for i = 1:nPop
        for k = 1:nVar
            % Migration
            if rand <= lambda(i)
                % Emmigration Probabilities
                EP    = mu;
                EP(i) = 0;
                EP    = EP/sum(EP);
                
                % Select Source Habitat
                j = RouletteWheelSelection(EP);
                
                % Migration
                newpop (i).Position(k) = pop (i).Position(k) ...
                    +alpha*(pop (j).Position(k)-pop (i).Position(k));
                
            end
            
            % Mutation
            if rand <= pMutation
                newpop(i).Position(k) = newpop(i).Position(k)+sigma*randn;
            end
        end
        
        % Apply Lower and Upper Bound Limits
        newpop(i).Position = max(newpop(i).Position, VarMin);
        newpop(i).Position = min(newpop(i).Position, VarMax);
        
        % Evaluation
        newpop (i).Cost = CostFunction(newpop (i).Position);
    end
    
    % Sort New Population
    [~, SortOrder] = sort([newpop.Cost]);
    newpop         = newpop(SortOrder);
    
    % Select Next Iteration Population
    pop = [pop(1:nKeep)
           newpop(1:nNew)];
     
    % Sort Population
    [~, SortOrder] = sort([pop.Cost]);
    pop            = pop(SortOrder);
    
    % Update Best Solution Ever Found
    BestSol = pop(1);
    
    % Store Best Cost Ever Found
    BestCost(it) = BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' ...
        num2str(BestCost(it))]);
    
    if BestCostPrevious - BestCost(it) > 1e-6
        r = 0;
    else
        r = r+1;  
    end
    BestCostPrevious = BestCost(it);
    it = it+1;
end
end

function j = RouletteWheelSelection(P)

    r = rand;
    C = cumsum(P);
    j = find(r<=C,1,'first');

end
