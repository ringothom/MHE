% Project Code: YPEA118
% Project Title: Implementation of Imperialist Competitive Algorithm (ICA)
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

function [BestSol] = ica(CostFunction,nVar,lb,ub,MaxIt,nPop,impr)
%% Imperialist Competitive Algorithm
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
VarMin  = lb;         % Lower Bound of Variables
VarMax  = ub;         % Upper Bound of Variables

%% ICA Parameters

nEmp  = 5;           % Number of Empires/Imperialists
alpha = 1;            % Selection Pressure
beta  = 1.5;          % Assimilation Coefficient

pRevolution = 0.05;   % Revolution Probability
mu          = 0.1;    % Revolution Rate

zeta  = 0.2;          % Colonies Mean Cost Coefficient

%% Globalization of Parameters and Settings

global ProblemSettings;
ProblemSettings.CostFunction = CostFunction;
ProblemSettings.nVar         = nVar;
ProblemSettings.VarSize      = VarSize;
ProblemSettings.VarMin       = VarMin;
ProblemSettings.VarMax       = VarMax;

global ICASettings;
ICASettings.MaxIt       = MaxIt;
ICASettings.nPop        = nPop;
ICASettings.nEmp        = nEmp;
ICASettings.alpha       = alpha;
ICASettings.beta        = beta;
ICASettings.pRevolution = pRevolution;
ICASettings.mu          = mu;
ICASettings.zeta        = zeta;

%% Initialization

% Initialize Empires
emp = CreateInitialEmpires();

% Array to Hold Best Cost Values
BestCost = zeros(MaxIt,1);
BestCostPrevious = Inf;
r = 0;
it = 1;

%% ICA Main Loop

while it <= MaxIt && r <= impr
    
    % Assimilation
    emp = AssimilateColonies(emp);
    
    % Revolution
    emp = DoRevolution(emp);
    
    % Intra-Empire Competition
    emp = IntraEmpireCompetition(emp);
    
    % Update Total Cost of Empires
    emp = UpdateTotalCost(emp);
    
    % Inter-Empire Competition
    emp = InterEmpireCompetition(emp);
    
    % Update Best Solution Ever Found
    imp               = [emp.Imp];
    [~, BestImpIndex] = min([imp.Cost]);
    BestSol           = imp(BestImpIndex);
    
    % Update Best Cost
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

function emp=CreateInitialEmpires()

    global ProblemSettings;
    global ICASettings;

    CostFunction = ProblemSettings.CostFunction;
%     nVar         = ProblemSettings.nVar;
    VarSize      = ProblemSettings.VarSize;
    VarMin       = ProblemSettings.VarMin;
    VarMax       = ProblemSettings.VarMax;
    
    nPop  = ICASettings.nPop;
    nEmp  = ICASettings.nEmp;
    nCol  = nPop-nEmp;
    alpha = ICASettings.alpha;
    
    empty_country.Position = [];
    empty_country.Cost     = [];
    
    country = repmat(empty_country,nPop,1);
    
    parfor i=1:nPop
        country (i).Position = unifrnd(VarMin,VarMax,VarSize);
        country (i).Cost     = feval(CostFunction,country (i).Position); 
    end

    
    costs          = [country.Cost];
    [~, SortOrder] = sort(costs);
    country        = country(SortOrder);
    
    imp = country(1:nEmp);
    col = country(nEmp+1:end);
    
    
    empty_empire.Imp       = [];
    empty_empire.Col       = repmat(empty_country,0,1);
    empty_empire.nCol      = 0;
    empty_empire.TotalCost = [];
    
    emp = repmat(empty_empire,nEmp,1);
    
    % Assign Imperialists
    for k=1:nEmp
        emp (k).Imp = imp(k);
    end
    
    % Assign Colonies
    P = exp(-alpha*[imp.Cost]/max([imp.Cost]));
    P = P/sum(P);
    for j=1:nCol
        
        k = RouletteWheelSelection(P);
        
        emp (k).Col=[emp(k).Col
                    col(j)];
        
        emp (k).nCol = emp (k).nCol+1;
    end
    
    emp = UpdateTotalCost(emp);
    
end

function emp = AssimilateColonies(emp)

    global ProblemSettings;
    CostFunction = ProblemSettings.CostFunction;
    VarSize      = ProblemSettings.VarSize;
    VarMin       = ProblemSettings.VarMin;
    VarMax       = ProblemSettings.VarMax;
    
    global ICASettings;
    beta = ICASettings.beta;
    
    nEmp=numel(emp);
    for k=1:nEmp
        Colnow = emp(k).Col;
        ImpPositionnow = emp(k).Imp.Position;
        parfor i=1:emp(k).nCol
            
%             emp (k).Col(i).Position = emp (k).Col(i).Position ...
%                 + beta*rand (VarSize).*(emp (k).Imp.Position-...
%                 emp (k).Col(i).Position);
            
            Colnow(i).Position = Colnow(i).Position ...
                + beta*rand (VarSize).*(ImpPositionnow-...
                Colnow(i).Position);
            
%             emp (k).Col(i).Position = max(emp (k).Col(i).Position,VarMin);
%             emp (k).Col(i).Position = min(emp (k).Col(i).Position,VarMax);
            Colnow(i).Position = max(Colnow(i).Position,VarMin);
            Colnow(i).Position = min(Colnow(i).Position,VarMax);

            Colnow(i).Cost = feval(CostFunction,Colnow(i).Position);
            
        end
        emp(k).Col = Colnow;
    end

end

function emp = DoRevolution(emp)

    global ProblemSettings;
    CostFunction = ProblemSettings.CostFunction;
    nVar         = ProblemSettings.nVar;
    VarSize      = ProblemSettings.VarSize;
    VarMin       = ProblemSettings.VarMin;
    VarMax       = ProblemSettings.VarMax;
    
    global ICASettings;
    pRevolution = ICASettings.pRevolution;
    mu          = ICASettings.mu;
    
    nmu   = ceil(mu*nVar);
    sigma = 0.1*(VarMax-VarMin);
    
    nEmp  = numel(emp);
    for k = 1:nEmp
        
        NewPos = emp (k).Imp.Position + sigma.*randn(VarSize);
        
        %boundary control
        NewPos = max(NewPos,VarMin);
        NewPos = min(NewPos,VarMax);
        
        jj                  = randsample (nVar,nmu)';
        NewImp              = emp (k).Imp;
        NewImp.Position(jj) = NewPos(jj);
        NewImp.Cost         = CostFunction(NewImp.Position);
        if NewImp.Cost < emp (k).Imp.Cost
            emp (k).Imp = NewImp;
        end
        
        for i = 1:emp (k).nCol
            if rand <= pRevolution

                NewPos = emp (k).Col(i).Position + sigma.*randn(VarSize);
                
                jj = randsample (nVar,nmu)';
                
                emp (k).Col(i).Position(jj) = NewPos(jj);

                emp(k).Col(i).Position = max(emp(k).Col(i).Position,...
                    VarMin);
                emp(k).Col(i).Position = min(emp(k).Col(i).Position,...
                    VarMax);

                emp(k).Col(i).Cost = CostFunction(emp(k).Col(i).Position);

            end
        end
    end

end

function emp = InterEmpireCompetition(emp)

    if numel(emp)==1
        return;
    end

    global ICASettings;
    alpha = ICASettings.alpha;

    TotalCost = [emp.TotalCost];
    
    [~, WeakestEmpIndex] = max(TotalCost);
    WeakestEmp           = emp(WeakestEmpIndex);
    
    P = exp(-alpha*TotalCost/max(TotalCost));
    P(WeakestEmpIndex) = 0;
    P = P/sum(P);
    if any(isnan(P))
        P(isnan(P)) = 0;
        if all(P==0)
            P(:) = 1;
        end
        P = P/sum(P);
    end
        
    if WeakestEmp.nCol>0
        [~, WeakestColIndex] = max([WeakestEmp.Col.Cost]);
        WeakestCol           = WeakestEmp.Col(WeakestColIndex);

        WinnerEmpIndex = RouletteWheelSelection(P);
        WinnerEmp      = emp(WinnerEmpIndex);

        WinnerEmp.Col(end+1) = WeakestCol;
        WinnerEmp.nCol       = WinnerEmp.nCol+1;
        emp(WinnerEmpIndex)  = WinnerEmp;

        WeakestEmp.Col(WeakestColIndex) = [];
        WeakestEmp.nCol                 = WeakestEmp.nCol-1;
        emp(WeakestEmpIndex)            = WeakestEmp;
    end
    
    if WeakestEmp.nCol==0
        
        WinnerEmpIndex2 = RouletteWheelSelection(P);
        WinnerEmp2      = emp(WinnerEmpIndex2);
        
        WinnerEmp2.Col(end+1) = WeakestEmp.Imp;
        WinnerEmp2.nCol       = WinnerEmp2.nCol+1;
        emp(WinnerEmpIndex2)  = WinnerEmp2;
        
        emp(WeakestEmpIndex)  = [];
    end
    
end

function emp=IntraEmpireCompetition(emp)

    nEmp = numel(emp);
    
    for k=1:nEmp
        for i=1:emp (k).nCol
            if emp (k).Col(i).Cost<emp (k).Imp.Cost
                imp = emp (k).Imp;
                col = emp (k).Col(i);
                
                emp (k).Imp    = col;
                emp (k).Col(i) = imp;
            end
        end
    end

end

function emp = UpdateTotalCost(emp)

    global ICASettings;
    zeta = ICASettings.zeta;
    
    nEmp = numel(emp);
    
    for k=1:nEmp
        if emp(k).nCol>0
            emp(k).TotalCost = emp(k).Imp.Cost+zeta*...
                                mean([emp(k).Col.Cost]);
        else
            emp(k).TotalCost = emp(k).Imp.Cost;
        end
    end

end

function i=RouletteWheelSelection(P)

    r = rand;
    
    C = cumsum(P);
    
    i = find(r<=C,1,'first');

end



