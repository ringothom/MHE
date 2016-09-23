% Project Code: YPEA108
% Project Title: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
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

function [BestEver] = cmaes(CostFunction,nVar,lb,ub,MaxIt,impr)
%% Covariance Matrix Adapting Evolutionary Strategy
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
% output:   BestEver         best position, best cost

%% Problem Settings

VarSize = [1 nVar];       % Decision Variables Matrix Size

VarMin = lb;             % Lower Bound of Decision Variables
VarMax = ub;             % Upper Bound of Decision Variables

%% CMA-ES Settings

% Population Size (and Number of Offsprings)
lambda = (4+round(3*log(nVar)))*10;

% Number of Parents
mu = round(lambda/2);

% Parent Weights
w = log(mu+0.5)-log(1:mu);
w = w/sum(w);

% Number of Effective Solutions
mu_eff = 1/sum(w.^2);

% Step Size Control Parameters (c_sigma and d_sigma);
sigma0 = 0.3*(VarMax-VarMin);
cs     = (mu_eff+2)/(nVar+mu_eff+5);
ds     = 1+cs+2*max(sqrt((mu_eff-1)/(nVar+1))-1,0);
ENN    = sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));

% Covariance Update Parameters
cc       = (4+mu_eff/nVar)/(4+nVar+2*mu_eff/nVar);
c1       = 2/((nVar+1.3)^2+mu_eff);
alpha_mu = 2;
cmu      = min(1-c1,alpha_mu*(mu_eff-2+1/mu_eff)/...
            ((nVar+2)^2+alpha_mu*mu_eff/2));
hth      = (1.4+2/(nVar+1))*ENN;

%% Initialization

ps     = cell(MaxIt,1);
pc     = cell(MaxIt,1);
C      = cell(MaxIt,1);
sigma  = cell(MaxIt,1);

ps{1}    = zeros(VarSize);
pc{1}    = zeros(VarSize);
C{1}     = eye(nVar);
sigma{1} = sigma0;

empty_individual.Position = [];
empty_individual.Step     = [];
empty_individual.Cost     = [];

M              = repmat(empty_individual,MaxIt,1);
M (1).Position = unifrnd(VarMin,VarMax,VarSize);
M (1).Step     = zeros(VarSize);
M (1).Cost     = CostFunction(M (1).Position);

BestSol           = M(1);
BestCost          = zeros(MaxIt,1);
BestEver.Cost     = Inf;
BestEver.Position = [];
r = 0;
g = 1;


%% CMA-ES Main Loop

while g <= MaxIt && r <= impr
    
    % Generate Samples
    pop = repmat(empty_individual,lambda,1);
    
    Cnow         = C{g};
    MnowPosition = M(g).Position;
    sigmanow     = sigma{g};
    parfor i = 1:lambda
        
        pop (i).Step     = mvnrnd(zeros(VarSize),Cnow);
        pop (i).Position = MnowPosition+sigmanow.*pop (i).Step;
        
        % boundary control
        pop (i).Position = max(pop (i).Position,VarMin);
        pop (i).Position = min(pop (i).Position,VarMax);
        
        pop (i).Cost     = feval(CostFunction,pop (i).Position);
    end
    % Update Best Solution Ever Found
    [~,ind_BestSol] = min([pop.Cost]);
    BestSol         = pop(ind_BestSol); 
    
    % Sort Population
    Costs          = [pop.Cost];
    [~, SortOrder] = sort(Costs);
    pop            = pop(SortOrder);
  
    % Save Results
    BestCost(g) = BestSol.Cost;
    
    % Display Results
    disp(['Iteration ' num2str(g) ': Best Cost = ' num2str(BestCost(g))]);
    
    if BestEver.Cost - BestSol.Cost > 1e-6
        BestEver = BestSol;
        r = 0;
    else
        r = r+1;  
    end
    %BestCostPrevious = BestCost(g);
    
    % Exit At Last Iteration
    if g == MaxIt
        break;
    end
        
    % Update Mean
    M (g+1).Step = 0;
    for j = 1:mu
        M (g+1).Step = M (g+1).Step+w(j)*pop (j).Step;
    end
    M (g+1).Position = M (g).Position+sigma{g}*M (g+1).Step;
    
    % boundary control
    M (g+1).Position = max(M (g+1).Position,VarMin);
    M (g+1).Position = min(M (g+1).Position,VarMax);
    
    
    M (g+1).Cost     = CostFunction(M (g+1).Position);
    if M (g+1).Cost < BestEver.Cost
        BestEver = M(g+1);
    end
    
    % Update Step Size
    ps{g+1}    = (1-cs)*ps{g}+sqrt(cs*(2-cs)*mu_eff)*...
        M (g+1).Step/chol(C{g})';
    sigma{g+1} = sigma{g}*exp (cs/ds*(norm (ps{g+1})/ENN-1))^0.3;
    
    % Update Covariance Matrix
    if norm (ps{g+1})/sqrt(1-(1-cs)^(2*(g+1))) < hth
        hs = 1;
    else
        hs = 0;
    end
    delta   = (1-hs)*cc*(2-cc);
    pc{g+1} = (1-cc)*pc{g}+hs*sqrt(cc*(2-cc)*mu_eff)*M (g+1).Step;
    C{g+1}  = (1-c1-cmu)*C{g}+c1*(pc {g+1}'*pc{g+1}+delta*C{g});
    for j=1:mu
        C{g+1} = C{g+1}+cmu*w(j)*pop (j).Step'*pop (j).Step;
    end
    
    % If Covariance Matrix is not Positive Definite or Near Singular
    [V, E] = eig(C{g+1});
    if any(diag(E)<0)
        E      = max(E,0);
        C{g+1} = V*E/V;
    end
    
    g = g+1;
end
