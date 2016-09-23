% Project Code: YPEA117
% Project Title: Implementation of Harmony Search in MATLAB
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

function [BestSol] = hs(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
% Harmony Search
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

%% Harmony Search Parameters

hms  = nPop;                        % Harmony Memory Size
nNew = 20;                          % Number of New Harmonies
hmcr = 0.9;                         % Harmony Memory Consideration Rate
par  = 0.1;                         % Pitch Adjustment Rate

fw      = 0.02*(VarMax-VarMin);     % Fret Width (Bandwidth)
fw_damp = 0.995;                    % Fret Width Damp Ratio

%% Initialization

% Empty Harmony Structure
empty_harmony.Position = [];
empty_harmony.Cost     = [];

% Initialize Harmony Memory
harmem = repmat(empty_harmony,hms,1);

% Create Initial Harmonies
for i = 1:hms
    harmem (i).Position = unifrnd(VarMin,VarMax,VarSize);
    harmem (i).Cost     = feval(CostFunction,harmem(i).Position);
end

% Sort Harmony Memory
[~, SortOrder] = sort([harmem.Cost]);
harmem         = harmem(SortOrder);

% Update Best Solution Ever Found
BestSol = harmem(1);

% Array to Hold Best Cost Values
BestCost = zeros(MaxIt,1);

%% Harmony Search Main Loop
r  = 0;
it = 1;
BestCostPrevious = Inf;

while it <= MaxIt && r <= impr
    
    % Initialize Array for New Harmonies
    newhar = repmat(empty_harmony,nNew,1);
    
    % Create New Harmonies
    for k = 1:nNew
        
        % Create New Harmony Position
        newhar (k).Position = unifrnd(VarMin,VarMax,VarSize);
        for j = 1:nVar
            if rand <= hmcr
                % Use Harmony Memory
                i = randi([1 hms]);
                newhar (k).Position(j) = harmem (i).Position(j);
            end
            
            % Pitch Adjustment
            if rand <= par
                % DELTA=fw*unifrnd(-1,+1);    % Uniform
                DELTA = fw*randn();            % Gaussian (Normal)
                
                newhar (k).Position(j) = newhar (k).Position(j)+DELTA;
            end
        
        end
        
        % Apply Variable Limits
        newhar (k).Position = max(newhar (k).Position,VarMin);
        newhar (k).Position = min(newhar (k).Position,VarMax);

        % Evaluation
        newhar (k).Cost = feval(CostFunction,newhar (k).Position);
        
    end
    
    % Merge Harmony Memory and New Harmonies
    harmem =[harmem
             newhar]; % #ok
    
    % Sort Harmony Memory
    [~, SortOrder] = sort([harmem.Cost]);
    harmem = harmem(SortOrder);
    
    % Truncate Extra Harmonies
    harmem = harmem(1:hms);
    
    % Update Best Solution Ever Found
    BestSol = harmem(1);
    
    % Store Best Cost Ever Found
    BestCost(it) = BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    % Damp Fret Width
    fw=fw*fw_damp;
    
     
    if BestCostPrevious - BestCost(it) > 1e-6
        r = 0;
    else
        r = r+1;  
    end
    BestCostPrevious = BestCost(it);
    it = it+1;
    
end
end