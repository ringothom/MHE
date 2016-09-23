% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA104
% Project Title: Ant Colony Optimization for Continuous Domains (ACOR)
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

function [BestSol] = aco(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Ant Colony Optimization
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

% CostFunction=@(x) Sphere(x);        % Cost Function

% nVar=10;             % Number of Decision Variables

VarSize=[1 nVar];   % Variables Matrix Size

VarMin= lb;         % Decision Variables Lower Bound
VarMax= ub;         % Decision Variables Upper Bound

%% ACOR Parameters

% MaxIt=1000;          % Maximum Number of Iterations

% nPop=10;            % Population Size (Archive Size)

nSample=40;         % Sample Size

q=0.5;              % Intensification Factor (Selection Pressure)

zeta=1;             % Deviation-Distance Ratio

%% Initialization

% Create Empty Individual Structure
empty_individual.Position=[];
empty_individual.Cost=[];

% Create Population Matrix
pop=repmat(empty_individual,nPop,1);

% Initialize Population Members
parfor i=1:nPop
    
    % Create Random Solution
    pop (i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Evaluation
    pop (i).Cost=feval(CostFunction,pop (i).Position);
    
end

% Sort Population
[~, SortOrder]=sort([pop.Cost]);
pop=pop(SortOrder);

% Update Best Solution Ever Found
BestSol=pop(1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

% Solution Weights
w=1/(sqrt(2*pi)*q*nPop)*exp(-0.5*(((1:nPop)-1)/(q*nPop)).^2);

% Selection Probabilities
p=w/sum(w);

% change to while loop
BestCostPrevious = Inf;
r = 0;
it = 1;

%% ACOR Main Loop

while it <= MaxIt && r <= impr
    
    % Means
    s=zeros(nPop,nVar);
    for l=1:nPop
        s(l,:)=pop (l).Position;
    end
    
    % Standard Deviations
    sigma=zeros(nPop,nVar);
    for l=1:nPop
        D=0;
        for k=1:nPop
            D=D+abs(s(l,:)-s(k,:));
        end
        sigma(l,:)=zeta*D/(nPop-1);
    end
    
    % Create New Population Array
    newpop=repmat(empty_individual,nSample,1);
    for t=1:nSample
        
        % Initialize Position Matrix
        newpop (t).Position=zeros(VarSize);
        
        % Solution Construction
        for i=1:nVar
            
            % Select Gaussian Kernel
            l=RouletteWheelSelection(p);
            
            % Generate Gaussian Random Variable
            newpop (t).Position(i)=s(l,i)+sigma(l,i)*randn;
            newpop (t).Position=max(newpop (t).Position,VarMin);
            newpop (t).Position=min(newpop (t).Position,VarMax);
            
        end
        
        % Evaluation
        newpop (t).Cost=CostFunction(newpop (t).Position);
        
    end
    
    % Merge Main Population (Archive) and New Population (Samples)
    pop=[pop
         newpop]; % #ok
     
    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Delete Extra Members
    pop=pop(1:nPop);
    
    % Update Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost
    BestCost(it)=BestSol.Cost;
    
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

function j=RouletteWheelSelection(P)

    rn=rand;
    C=cumsum(P);
    j=find(rn<=C,1,'first');

end