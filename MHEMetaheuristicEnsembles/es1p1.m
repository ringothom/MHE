%   This algorithm implements the simplest of all "evolution strategies"
%   (ES), the (1+1)-ES. In each iteration, one parent is used to create 
%   one offspring using a Gaussian mutation operator (A random gaussian 
%   variable with mean zero and standard deviation 'sigma').
%   Bibliography:
%
%   - KALYANMOY, Deb. "Multi-Objective optimization using evolutionary
%     algorithms". John Wiley & Sons, Ltd. Kanpur, India. 2001.
%
% -------------------------------------------------------
% | Developed by:   Gilberto Alejandro Ortiz Garcia     |
% |                 gialorga@gmail.com                  |
% |                 Universidad Nacional de Colombia    |
% |                 Manizales, Colombia.                |
% -------------------------------------------------------
%
%   Date: 20 - Sep - 2011
%
%   Note: This is a re-design as a function developed by R.Thomschke. For 
%   the original go to 
%   https://de.mathworks.com/matlabcentral/fileexchange/35800-1+1-evolution-strategy--es-.

function [BestSol] = es1p1(f, nVar, lb, ub, MaxIt, impr)
%%  (1+1)-Evolutionary Strategy
%
%   INPUT DATA:
%
%   - f:     Function to minimize (handle function)
%   - nVar:  number of variables
%   - lb:    lower bound
%   - ub:    upper bound
%   - MaxIt: Maximum number of iterations (positive integer number)
%   - impr:  maximum number of iterations without improvement
%
%   OUTPUT DATA:
%
%   - BestSol: best position, best cost
%

%% Beginning
VarMin  = lb;                     % lower bound
VarMax  = ub;                     % upper bound

xkm1    = ones(1,nVar);

sigma   = ones(nVar,1);           % mutation strength

n       = length(xkm1);           % 'n' states
xk      = zeros(n,MaxIt);         % Pre-allocate space for 'xk'

xk(:,1) = xkm1;                   % initial guessing
fx      = f(xk(:,1)');            % evaluate function with 'xk'

%% Main Loop
BestCostPrevious = Inf;
r  = 0;
k = 1;

while ((k < MaxIt) && (r <= impr))
  y  = xk(:,k) + sigma.*randn(n,1); % create mutated solution 'y'
 
  % boundary control
  y = max(y,VarMin);
  y = min(y,VarMax);
  
  
  fy = f(y');                        % evaluate function with mutated solution 'y'
  if (fy < fx)
    xk(:,k+1) = y;                  % update value of xkm1
    fx        = fy;                 % update value of f(xkm1)
  else
    xk(:,k+1) = xk(:,k);            % retain value of xkm1
  end
  
  disp(['Iteration ' num2str(k) ': Best Cost = ' ...
        num2str(fx)]);
    
  BestSol.Position = xk(:,k+1);
  BestSol.Cost     = fx;
  

  
  
  if BestCostPrevious - fx > 1e-6
        r = 0;
  else
        r = r+1;  
  end
  BestCostPrevious = fx;
  k = k+1;                        % update counter
end

end
%% END