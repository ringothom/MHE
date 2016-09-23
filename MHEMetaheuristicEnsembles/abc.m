% Project Code: YPEA114
% Project Title: Implementation of Artificial Bee Colony in MATLAB
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

function [BestSol] = abc(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Artificial Bee Colony Optimization
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

%% ABC Settings

nOnlooker = nPop;         % Number of Onlooker Bees

L = round(0.6*nVar*nPop); % Abandonment Limit Parameter (Trial Limit)

a = 1;                    % Acceleration Coefficient Upper Bound

%% Initialization

% Empty Bee Structure
empty_bee.Position = [];
empty_bee.Cost     = [];

% Initialize Population Array
pop = repmat(empty_bee,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost = inf;

% Create Initial Population
parfor i = 1:nPop
    pop (i).Position = unifrnd(VarMin,VarMax,VarSize);
    pop (i).Cost     = feval(CostFunction,(pop (i).Position));
end
[~,ind_BestSol] = min([pop.Cost]);
BestSol         = pop(ind_BestSol);

% Abandonment Counter
C = zeros(nPop,1);

% Array to Hold Best Cost Values
BestCost         = zeros(MaxIt,1);
BestCostPrevious = Inf;
r  = 0;
it = 1;

%% ABC Main Loop

while it <= MaxIt && r <= impr
        
    % Recruited Bees
    for i = 1:nPop
        
        % Choose k randomly, not equal to i
        K = [1:i-1 i+1:nPop];
        k = K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi = a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position = pop (i).Position+phi.*...
            (pop (i).Position-pop (k).Position);
        
        % ADDED: newbee into bounds
        newbee.Position = max(newbee.Position,VarMin);
        newbee.Position = min(newbee.Position,VarMax);
        
        % Evaluation
        newbee.Cost = feval(CostFunction,newbee.Position);
        
        % Comparision
        if newbee.Cost <= pop (i).Cost
            pop(i) = newbee;
        else
            C(i) = C(i)+1;
        end
        
    end
    
    % Calculate Fitness Values and Selection Probabilities
    F = zeros(nPop,1);
    MeanCost = mean([pop.Cost]);
    parfor i = 1:nPop
        F(i) = exp(-pop(i).Cost/MeanCost); % Convert Cost to Fitness
    end
    P = F/sum(F);
    
    % Onlooker Bees
    for m = 1:nOnlooker
        
        % Select Source Site
        i = RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to i
        K = [1:i-1 i+1:nPop];
        k = K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi = a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position = pop (i).Position+phi.*...
            (pop (i).Position-pop (k).Position);
        
        % ADDED: newbee into bounds
        newbee.Position = max(newbee.Position,VarMin);
        newbee.Position = min(newbee.Position,VarMax);
        
        % Evaluation
        newbee.Cost = CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost <= pop (i).Cost
            pop(i) = newbee;
        else
            C(i) = C(i)+1;
        end
        
    end
    
    % Scout Bees
    parfor i = 1:nPop
        if C(i) >= L
            pop (i).Position = unifrnd(VarMin,VarMax,VarSize);
            pop (i).Cost     = feval(CostFunction,pop (i).Position);
            C(i)=0;
        end
    end
    
    % Update Best Solution Ever Found
    for i = 1:nPop
        if pop (i).Cost <= BestSol.Cost
            BestSol = pop(i);
        end
    end
    
    % Store Best Cost Ever Found
    BestCost(it) = BestSol.Cost;
    
    % Display Iteration Information
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

function i=RouletteWheelSelection(P)

    r=rand;
    
    C=cumsum(P);
    
    i=find(r<=C,1,'first');

end