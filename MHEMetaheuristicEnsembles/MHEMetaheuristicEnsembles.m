%% INPUT
path = '';    % must contain folders 'matlabfiles' and 'predictions'
cl   = 12;    % number of clusters for parallel computing

%% Prepair 

% Set paths
pathPred = strcat(path, '/predictions');

% Get data sets names
cd(pathPred);
dir pathPred;
set = ls;
set = set(3:size(set,1),1:size(set,2));

% for parallel computing
cd(path)
parpool('local',12,'AttachedFiles',{'fn_brier.m','fn_hmeasure.m',...
      'fn_diff.m','fn_ia.m'})

%% Loop over data sets
cd(pathPred)

for i = 1:size(set,1)
    
  ds = deblank(set(i,1:end));
  
  fprintf(['Calculate ensemble weights for ',ds,'\n'])
  
  % get predictions
  cd([pathPred,'\',ds]);
  predTrain = csvread([ds,'-predTrain.csv'],1,1);
  
  % get observations
  obsTrainNum = csvread([ds,'-obsTrainNum.csv'],1,1);
  
  % global metaheuristic variables
  nvars = size(predTrain,2);
  lb=0;  %lower bound
  ub=1;   %upper bound
  
  %% ACO
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = aco(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-aco-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = aco(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-aco-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = aco(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-aco-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = aco(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-aco-ia.mat'],'ens')
  
  %% ABC 
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = abc(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-abc-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = abc(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-abc-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = abc(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-abc-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = abc(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-abc-ia.mat'],'ens')
  
  %% BBO
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bbo-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bbo-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bbo-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bbo-ia.mat'],'ens')

  %% BSA
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bsa(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bsa-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bsa(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bsa-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bsa(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bsa-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = bsa(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-bsa-ia.mat'],'ens')

  %% ICA
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ica(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ica-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ica(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ica-ia.mat'],'ens')
  
  %% CA
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ca(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ca-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ca(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ca-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ca(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ca-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ca(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ca-ia.mat'],'ens')
  
  %% CMAES
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = cmaes(ObjectiveFunction, nvars,...
      lb,ub,1000,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-cmaes-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = cmaes(ObjectiveFunction, nvars,...
      lb,ub,1000,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-cmaes-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = cmaes(ObjectiveFunction, nvars,...
      lb,ub,1000,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-cmaes-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = cmaes(ObjectiveFunction, nvars,...
      lb,ub,1000,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-cmaes-ia.mat'],'ens')
  
  %% DE
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = de(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-de-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = de(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-de-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = de(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-de-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = de(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-de-ia.mat'],'ens')

  %% ES1P1
  
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = es1p1(ObjectiveFunction, nvars,...
      lb,ub,1000,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-es1p1-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = es1p1(ObjectiveFunction, nvars,...
      lb,ub,1000,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-es1p1-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = es1p1(ObjectiveFunction, nvars,...
      lb,ub,1000,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-es1p1-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = es1p1(ObjectiveFunction, nvars,...
      lb,ub,1000,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-es1p1-ia.mat'],'ens')
  
  %% FA
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = fireflyalgorithm(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-endsemble-fa-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = fireflyalgorithm(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-endsemble-fa-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = fireflyalgorithm(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-endsemble-fa-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = fireflyalgorithm(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-endsemble-fa-ia.mat'],'ens')

  %% GA
  % from accuracy measures
  % optimization options
  options = gaoptimset('PopulationSize',25,'Generations',1000,...
    'PopulationType','doubleVector',...
    'UseParallel', true, 'Vectorized', 'off',...
    'StallGenLimit',20,'TolFun',1e-6);

  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = ga(ObjectiveFunction, nvars,...
      [],[],[],[], lb, ub,[], options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ga-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = ga(ObjectiveFunction, nvars,...
      [],[],[],[], lb, ub,[], options);
  ens.time    = toc;
  ens.cost    = -cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ga-hm.mat'],'ens')
  
  % from diversity measures
  % optimization options
  options = gaoptimset('PopulationSize',25,'Generations',1000,...
    'PopulationType','doubleVector',...
    'UseParallel', true, 'Vectorized', 'off',...
    'StallGenLimit',100,'TolFun',1e-6);

  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = ga(ObjectiveFunction, nvars,...
      [],[],[],[], lb, ub,[], options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ga-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = ga(ObjectiveFunction, nvars,...
      [],[],[],[], lb, ub,[], options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ga-ia.mat'],'ens')
  
  %% HS
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = hs(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-hs-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = hs(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-hs-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = hs(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-hs-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = hs(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-hs-ia.mat'],'ens')
  
  %% ICA
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ica(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ica-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ica(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ica-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ica(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ica-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = ica(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-ica-ia.mat'],'ens')
  
  %% PSO
  % optimization options
  options = optimoptions('particleswarm','SwarmSize',25,...
    'MaxIter',1000,'StallIterLimit',20,...
    'TolFun',1e-6,...
    'UseParallel', true, 'Vectorized', 'off');
   
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = particleswarm(ObjectiveFunction, nvars,...
      lb,ub, options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-pso-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = particleswarm(ObjectiveFunction, nvars,...
      lb,ub, options);
  ens.time    = toc;
  ens.cost    = -cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-pso-hm.mat'],'ens')
  
  
  % from diversity measures
  % optimization options
  options = optimoptions('particleswarm','SwarmSize',25,...
    'MaxIter',1000,'StallIterLimit',100,...
    'TolFun',1e-6,...
    'UseParallel', true, 'Vectorized', 'off');

  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = particleswarm(ObjectiveFunction, nvars,...
      lb,ub, options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-pso-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = particleswarm(ObjectiveFunction, nvars,...
      lb,ub, options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-pso-ia.mat'],'ens')

  %% SA
  % calculate weights
  x0   = 1/size(predTrain,2)*ones(1,size(predTrain,2));
  lbnd = zeros(1,size(predTrain,2));  %lower bound
  ubnd = ones(1,size(predTrain,2));   %upper bound
   
  % from accuracy measures
  % optimization options
  options = saoptimset('MaxIter',1000,...
      'StallIterLimit',20,'TolFun',1e-6);
  
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = simulannealbnd(ObjectiveFunction, x0,...
      lbnd,ubnd, options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sa-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = simulannealbnd(ObjectiveFunction,x0,...
      lbnd,ubnd, options);
  ens.time    = toc;
  ens.cost    = -cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sa-hm.mat'],'ens')
  
  % from diversity measures
  % optimization options
  options = saoptimset('MaxIter',1000,...
      'StallIterLimit',100,'TolFun',1e-6);
  
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = simulannealbnd(ObjectiveFunction, x0,...
      lbnd,ubnd, options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sa-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [weights, cost] = simulannealbnd(ObjectiveFunction, x0,...
      lbnd,ubnd, options);
  ens.time    = toc;
  ens.cost    = cost;
  ens.weights = weights/sum(weights);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sa-ia.mat'],'ens')
  
  %% SCE
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = sce(ObjectiveFunction, nvars,...
      lb,ub,1000,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sce-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = sce(ObjectiveFunction, nvars,...
      lb,ub,1000,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sce-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = sce(ObjectiveFunction, nvars,...
      lb,ub,1000,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sce-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = sce(ObjectiveFunction, nvars,...
      lb,ub,1000,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-sce-ia.mat'],'ens')
  
  %% TLBO
  % from accuracy measures
  fprintf('     Fitness function: Brier\n')
  cd(path);
  ObjectiveFunction = @(x) fn_brier(x,predTrain,obsTrainNum);
  tic
  [BestSol] = tlbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-tlbo-brier.mat'],'ens')
  
  fprintf('     Fitness function: H measure\n')
  cd(path);
  ObjectiveFunction = @(x) -fn_hmeasure(x,predTrain,obsTrainNum);
  tic
  [BestSol] = tlbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,20);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-tlbo-hm.mat'],'ens')
  
  % from diversity measures
  fprintf('     Fitness function: Difficulty\n')
  cd(path);
  ObjectiveFunction = @(x) fn_diff(x,predTrain,obsTrainNum);
  tic
  [BestSol] = tlbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-tlbo-diff.mat'],'ens')
  
  fprintf('     Fitness function: Interrater Agreement\n')
  cd(path);
  ObjectiveFunction = @(x) fn_ia(x,predTrain,obsTrainNum);
  tic
  [BestSol] = tlbo(ObjectiveFunction, nvars,...
      lb,ub,1000,25,100);
  ens.time    = toc;
  ens.cost    = BestSol.Cost;
  ens.weights = BestSol.Position/sum(BestSol.Position);
  cd([pathPred,'\',ds]);
  save([ds,'-ensemble-tlbo-ia.mat'],'ens')

end
