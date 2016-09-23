%{
Backtracking Search Optimization Algorithm (BSA)

Platform: Matlab 2013a   


Cite this algorithm as;
[1]  P. Civicioglu, "Backtracking Search Optimization Algorithm for 
numerical optimization problems", Applied Mathematics and Computation, 219, 81218144, 2013.


Copyright Notice
Copyright (c) 2012, Pinar Civicioglu
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the follbing conditions are 
met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the follbing disclaimer.
    * Redistributions in binary form must reproduce the copyright 
      notice, this list of conditions and the follbing disclaimer in 
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
% original go to 
% https://de.mathworks.com/matlabcentral/fileexchange/44842-backtracking-search-optimization-algorithm

function BestSol = bsa(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Backtracking Search Optimization Algorithm 
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


DIM_RATE = 1;

%INITIALIZATION
if numel(lb) == 1
    lb = lb*ones(1,nVar); 
    ub = ub*ones(1,nVar); 
end % this line must be adapted to your problem

pop = GeneratePopulation(nPop,nVar,lb,ub); % see Eq.1 in [1]
% Create Initial Population
fitnesspop = zeros(1,nPop);
parfor i = 1:nPop
    fitnesspop(i) = feval(CostFunction,pop(i,:));
end
historical_pop = GeneratePopulation(nPop,nVar,lb,ub); % see Eq.2 in [1]

% historical_pop  is swarm-memory of BSA as mentioned in [1].


% -------------------------------------------------------------------------
BestCostPrevious = Inf;
r   = 0;
epk = 1;
while epk <= MaxIt && r <= impr
    
    %SELECTION-I
    if rand < rand 
        historical_pop = pop; 
    end  % see Eq.3 in [1]
    
    historical_pop = historical_pop(randperm(nPop),:); % see Eq.4 in [1]
    F   = get_scale_factor; % see Eq.5 in [1], you can other F generation 
                            % strategies 
    map = zeros(nPop,nVar); % see Algorithm-2 in [1]  
    
    if rand < rand,
        for i = 1:nPop  
            u = randperm(nVar); 
            map(i,u(1:ceil(DIM_RATE*rand*nVar))) = 1; 
        end
    else
        for i = 1:nPop  
            map(i,randi(nVar)) = 1; 
        end
    end
    
    % RECOMBINATION (MUTATION+CROSSOVER)   
    offsprings = pop+(map.*F).*(historical_pop-pop);   % see Eq.5 in [1]    
    offsprings = BoundaryControl(offsprings,lb,ub); % see Algorithm-3 in [1]
    
    % SELECTON-II
    % fitnessoffsprings   = feval(CostFunction,offsprings,mydata);
    
    parfor i = 1:nPop
        fitnessoffsprings(i) = feval(CostFunction,offsprings(i,:));
    end
    
    ind                 = fitnessoffsprings<fitnesspop;
    fitnesspop(ind)     = fitnessoffsprings(ind);
    pop(ind,:)          = offsprings(ind,:);
    [globalminimum,ind] = min(fitnesspop);    
    globalminimizer     = pop(ind,:);
    
    % EXPORT SOLUTIONS 
    assignin('base','globalminimizer',globalminimizer);
    assignin('base','globalminimum',globalminimum);
    fprintf('BSA|%5.0f -----> %9.16f\n',epk,globalminimum);
    
    BestSol.Position = globalminimizer;
    BestSol.Cost     = globalminimum;
    
    if BestCostPrevious - BestSol.Cost > 1e-6
        r = 0;
    else
        r = r+1;  
    end
    BestCostPrevious = BestSol.Cost;
    epk = epk+1;

end


function pop = GeneratePopulation(nPop,nVar,lb,ub)

pop = ones(nPop,nVar);

for i = 1:nPop
    for j = 1:nVar
        pop(i,j) = rand*(ub(j)-lb(j))+lb(j);
    end
end
return

function pop = BoundaryControl(pop,lb,ub)
[nPop,nVar] = size(pop);
for i = 1:nPop
    for j = 1:nVar                
        k = rand < rand; % you can change boundary-control strategy
        if pop(i,j) < lb(j) 
            if k 
                pop(i,j) = lb(j); 
            else pop(i,j) = rand*(ub(j)-lb(j))+lb(j); 
            end
        end
        if pop(i,j) > ub(j)  
            if k 
                pop(i,j) = ub(j);  
            else
                pop(i,j) = rand*(ub(j)-lb(j))+lb(j); 
            end
        end
    end
end
return

function F = get_scale_factor 
% you can change generation strategy of scale-factor,F    
     F = 3*randn; % STANDARD brownian-walk
    % F=4*randg;  % brownian-walk    
    % F=lognrnd(rand,5*rand);  % brownian-walk              
    % F=1/normrnd(0,5);        % pseudo-stable walk (levy-like)
    % F=1./gamrnd(1,0.5);      % pseudo-stable walk (levy-like, simulates 
                               % inverse gamma distribution; 
                               % levy-distiribution)   
return