% Project Code: YPEA110
% Project Title: Implementation of Shuffled Complex Evolution (SCE-UA)
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


function [BestSol] = sce(CostFunction,nVar,lb,ub,MaxIt,impr)
%% Shuffled Complex Evolution
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

VarSize = [1 nVar];     % Unknown Variables Matrix Size

VarMin  = lb;           % Lower Bound of Unknown Variables
VarMax  = ub;           % Upper Bound of Unknown Variables
%% SCE-UA Parameters

nPopComplex = 5;                        % Complex Size
nPopComplex = max(nPopComplex, nVar+1); % Nelder-Mead Standard

nComplex = 5;                       % Number of Complexes
nPop     = nComplex*nPopComplex;    % Population Size

I = reshape(1:nPop, nComplex, []);

% CCE Parameters
cce_params.q            = max(round(0.5*nPopComplex),2);   % Number of Parents
cce_params.alpha        = 3;    % Number of Offsprings
cce_params.beta         = 5;    % Maximum Number of Iterations
cce_params.CostFunction = CostFunction;
cce_params.VarMin       = VarMin;
cce_params.VarMax       = VarMax;

%% Initialization

% Empty Individual Template
empty_individual.Position = [];
empty_individual.Cost     = [];

% Initialize Population Array
pop = repmat(empty_individual, nPop, 1);

% Initialize Population Members
parfor i = 1:nPop
    pop (i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop (i).Cost     = feval(CostFunction,pop (i).Position);
end

% Sort Population
pop = SortPopulation(pop);

% Update Best Solution Ever Found
BestSol = pop(1);

% Initialize Best Costs Record Array
BestCosts = nan(MaxIt, 1);

%% SCE-UA Main Loop
BestCostPrevious = Inf;
r  = 0;
it = 1; 

while it <= MaxIt && r <= impr
    
    % Initialize Complexes Array
    Complex = cell(nComplex, 1);
    
    % Form Complexes and Run CCE
    for j = 1:nComplex
        % Complex Formation
        Complex{j} = pop(I(j,:));
        
        % Run CCE
        Complex{j} = RunCCE(Complex{j}, cce_params);
        
        % Insert Updated Complex into Population
        pop(I(j,:)) = Complex{j};
    end
    
    % Sort Population
    pop = SortPopulation(pop);
    
    % Update Best Solution Ever Found
    BestSol = pop(1);
    
    % Store Best Cost Ever Found
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

function [pop, SortOrder] = SortPopulation(pop)

    % Get Costs
    Costs = [pop.Cost];
    
    % Sort the Costs Vector
    [~, SortOrder] = sort(Costs);
    
    % Apply the Sort Order to Population
    pop = pop(SortOrder);

end

function pop = RunCCE(pop, params)

    %% CCE Parameters
    q            = params.q;            % Number of Parents
    alpha        = params.alpha;        % Number of Offsprings
    beta         = params.beta;         % Maximum Number of Iterations
    CostFunction = params.CostFunction;
    VarMin       = params.VarMin;
    VarMax       = params.VarMax;

    nPop = numel(pop);                            % Population Size
    P    = 2*(nPop+1-(1:nPop))/(nPop*(nPop+1));   % Selection Probabilities
    
    % Calculate Population Range (Smallest Hypercube)
    LowerBound = pop (1).Position;
    UpperBound = pop (1).Position;
    parfor i = 2:nPop
        LowerBound = min(LowerBound, pop (i).Position);
        UpperBound = max(UpperBound, pop (i).Position);
    end
    
    %% CCE Main Loop

    for it = 1:beta
        
        % Select Parents
        L = RandSample(P,q);
        B = pop(L);
        
        % Generate Offsprings
        for k = 1:alpha
            
            % Sort Population
            [B, SortOrder] = SortPopulation(B);
            L = L(SortOrder);
            
            % Calculate the Centroid
            g = 0;
            for i = 1:q-1
                g = g + B (i).Position;
            end
            g = g/(q-1);
            
            % Reflection
            ReflectionSol          = B(end);
            ReflectionSol.Position = 2*g - B (end).Position;
            if ~IsInRange(ReflectionSol.Position, VarMin, VarMax)
                ReflectionSol.Position = unifrnd(LowerBound, UpperBound);
            end
            ReflectionSol.Cost = CostFunction(ReflectionSol.Position);
            
            if ReflectionSol.Cost < B (end).Cost
                B(end) = ReflectionSol;
            else
                % Contraction
                ContractionSol          = B(end);
                ContractionSol.Position = (g+B (end).Position)/2;
                ContractionSol.Cost     = feval(CostFunction,...
                                            ContractionSol.Position);
                
                if ContractionSol.Cost < B(end).Cost
                    B(end) = ContractionSol;
                else
                    B (end).Position = unifrnd(LowerBound, UpperBound);
                    B (end).Cost     = CostFunction(B (end).Position);
                end
            end
            
        end
        
        % Return Back Subcomplex to Main Complex
        pop(L) = B;
        
    end
    
end

function b = IsInRange(x, VarMin, VarMax)

    b = all(x>=VarMin) && all(x<=VarMax);

end

function L = RandSample(P, q, replacement)

    if ~exist('replacement','var')
        replacement = false;
    end

    L = zeros(q,1);
    for i = 1:q
        L(i) = randsample(numel(P), 1, true, P);
        if ~replacement
            P(L(i)) = 0;
        end
    end

end
