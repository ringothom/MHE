% Project Code: YPEA111
% Project Title: Implementation of TLBO in MATLAB
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

function [BestSol] = tlbo(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Teaching-Learning-Based Optimization
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

VarSize = [1 nVar]; % Unknown Variables Matrix Size

VarMin = lb;        % Unknown Variables Lower Bound
VarMax = ub;        % Unknown Variables Upper Bound

%% TLBO Parameters


%% Initialization 

% Empty Structure for Individuals
empty_individual.Position = [];
empty_individual.Cost     = [];

% Initialize Population Array
pop = repmat(empty_individual, nPop, 1);

% Initialize Best Solution
BestSol.Cost = inf;

% Initialize Population Members
for i = 1 : nPop
    pop (i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop (i).Cost     = feval(CostFunction,pop (i).Position);
end

% Sort Harmony Memory
[~, SortOrder] = sort([pop.Cost]);
pop            = pop(SortOrder);

% Update Best Solution Ever Found
BestSol = pop(1);

% Initialize Best Cost Record
BestCosts = zeros(MaxIt,1);

%% TLBO Main Loop
r  = 0;
it = 1;
BestCostPrevious = Inf;

while it <= MaxIt && r <= impr
    
    % Calculate Population Mean
    Mean = 0;
    for i = 1:nPop
        Mean = Mean + pop (i).Position;
    end
    Mean = Mean/nPop;
    
    % Select Teacher
    Teacher = pop(1);
    for i = 2:nPop
        if pop (i).Cost < Teacher.Cost
            Teacher = pop(i);
        end
    end
    
    % Teacher Phase
    for i = 1:nPop
        % Create Empty Solution
        newsol = empty_individual;
        
        % Teaching Factor
        TF = randi([1 2]);
        
        % Teaching (moving towards teacher)
        newsol.Position = pop (i).Position ...
            + rand(VarSize).*(Teacher.Position - TF*Mean);
        
        % Clipping
        newsol.Position = max(newsol.Position, VarMin);
        newsol.Position = min(newsol.Position, VarMax);
        
        % Evaluation
        newsol.Cost = CostFunction(newsol.Position);
        
        % Comparision
        if newsol.Cost < pop (i).Cost
            pop(i) = newsol;
            if pop (i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end
    end
    
    % Learner Phase
    for i = 1:nPop
        
        A    = 1:nPop;
        A(i) = [];
        j    = A(randi(nPop-1));
        
        Step = pop (i).Position - pop (j).Position;
        if pop (j).Cost < pop (i).Cost
            Step = -Step;
        end
        
        % Create Empty Solution
        newsol = empty_individual;
        
        % Teaching (moving towards teacher)
        newsol.Position = pop (i).Position + rand (VarSize).*Step;
        
        % Clipping
        newsol.Position = max(newsol.Position, VarMin);
        newsol.Position = min(newsol.Position, VarMax);
        
        % Evaluation
        newsol.Cost = CostFunction(newsol.Position);
        
        % Comparision
        if newsol.Cost < pop (i).Cost
            pop(i) = newsol;
            if pop (i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end
    end
    
    % Store Record for Current Iteration
    BestCosts(it) = BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' ...
        num2str(BestCosts(it))]);
    
    if BestCostPrevious - BestCosts(it) > 1e-6
        r = 0;
    else
        r = r+1;  
    end
    BestCostPrevious = BestCosts(it);
    it = it+1;
end
end