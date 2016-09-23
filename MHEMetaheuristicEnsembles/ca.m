% Project Code: YPEA125
% Project Title: Implementation of Cultural Algorithm in MATLAB
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

function [BestSol] = ca(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Cultural Algorithm
%
% input:    CostFunction    function to minimize
%           nVar            number of variables
%           lb              lower bound (scalar)
%           ub              upper bound (scalar)
%           MaxIt           maximal number of iterations
%           impr            maximal number of iterations without
%                           improvement
%
% output:   BestSol         best position, best cost

%% Problem Definition

VarSize = [1 nVar];   % Decision Variables Matrix Size

VarMin = lb;         % Decision Variables Lower Bound
VarMax = ub;         % Decision Variables Upper Bound

%% Cultural Algorithm Settings

pAccept = 0.35;                   % Acceptance Ratio
nAccept = round(pAccept*nPop);    % Number of Accepted Individuals

alpha = 0.3;

% beta=0.5;

%% Initialization

% Initialize Culture
Culture.Situational.Cost     = inf;
Culture.Situational.Position = inf(VarSize);
Culture.Normative.Min        = inf(VarSize);
Culture.Normative.Max        = -inf(VarSize);
Culture.Normative.L          = inf(VarSize);
Culture.Normative.U          = inf(VarSize);

% Empty Individual Structure
empty_individual.Position = [];
empty_individual.Cost     = [];

% Initialize Population Array
pop = repmat(empty_individual,nPop,1);

% Generate Initial Solutions
parfor i = 1:nPop
    pop (i).Position = unifrnd(VarMin,VarMax,VarSize);
    pop (i).Cost     = feval(CostFunction,pop (i).Position);
end

% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop            = pop(SortOrder);

% Adjust Culture using Selected Population
spop    = pop(1:nAccept);
Culture = AdjustCulture(Culture,spop);

% Update Best Solution Ever Found
BestSol = Culture.Situational;

% Array to Hold Best Costs
BestCost         = zeros(MaxIt,1);
BestCostPrevious = Inf;
r  = 0;
it = 1;

%% Cultural Algorithm Main Loop

while it <= MaxIt && r <= impr
    
    % Influnce of Culture
    for i = 1:nPop
        
        % % 1st Method (using only Normative component)
%         sigma=alpha*Culture.Normative.Size;
%         pop (i).Position=pop (i).Position+sigma.*randn(VarSize);
        
        % % 2nd Method (using only Situational component)
%         for j=1:nVar
%            sigma=0.1*(VarMax-VarMin);
%            dx=sigma*randn;
%            if pop (i).Position(j)<Culture.Situational.Position(j)
%                dx=abs(dx);
%            elseif pop (i).Position(j)>Culture.Situational.Position(j)
%                dx=-abs(dx);
%            end
%            pop (i).Position(j)=pop (i).Position(j)+dx;
%         end
        
        % % 3rd Method (using Normative and Situational components)
        for j = 1:nVar
          sigma = alpha*Culture.Normative.Size(j);
          dx    = sigma*randn;
          if pop (i).Position(j) < Culture.Situational.Position(j)
              dx = abs(dx);
          elseif pop (i).Position(j) > Culture.Situational.Position(j)
              dx = -abs(dx);
          end
          pop (i).Position(j) = pop (i).Position(j)+dx;
        end        
        pop(i).Position = max(pop(i).Position,VarMin);
        pop(i).Position = min(pop(i).Position,VarMax);
        
        % % 4th Method (using Size and Range of Normative component)
%         for j=1:nVar
%           sigma=alpha*Culture.Normative.Size(j);
%           dx=sigma*randn;
%           if pop (i).Position(j)<Culture.Normative.Min(j)
%               dx=abs(dx);
%           elseif pop (i).Position(j)>Culture.Normative.Max(j)
%               dx=-abs(dx);
%           else
%               dx=beta*dx;
%           end
%           pop (i).Position(j)=pop (i).Position(j)+dx;
%         end        
        
        pop (i).Cost = CostFunction(pop (i).Position);
        
    end
    
    % Sort Population
    [~, SortOrder] = sort([pop.Cost]);
    pop            = pop(SortOrder);

    % Adjust Culture using Selected Population
    spop    = pop(1:nAccept);
    Culture = AdjustCulture(Culture,spop);

    % Update Best Solution Ever Found
    BestSol = Culture.Situational;
    
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

function Culture = AdjustCulture(Culture,spop)

    n    = numel(spop);
    nVar = numel(spop(1).Position);
    
    for i = 1:n
        if spop(i).Cost < Culture.Situational.Cost
            Culture.Situational = spop(i);
        end
        
        for j = 1:nVar
            if spop(i).Position(j) < Culture.Normative.Min(j) ...
                    || spop(i).Cost < Culture.Normative.L(j)
                Culture.Normative.Min(j) = spop (i).Position(j);
                Culture.Normative.L(j)   = spop (i).Cost;
            end
            if spop(i).Position(j) > Culture.Normative.Max(j) ...
                    || spop (i).Cost < Culture.Normative.U(j)
                Culture.Normative.Max(j) = spop (i).Position(j);
                Culture.Normative.U(j)   = spop (i).Cost;
            end
        end
    end

    Culture.Normative.Size = Culture.Normative.Max-Culture.Normative.Min;
    
end
