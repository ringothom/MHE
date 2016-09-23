function hm = fn_hmeasure(x,pred,obs)
    % fitness function: H measure 
    % input: weights vector, prediction matrix, observations vector
    % output: H measure (maximize for maximizing accuracy)
    x = x/sum(x);
    weightPred = bsxfun(@times,pred,x);
    sumPred = sum(weightPred,2);
    hm = hmeasure(obs,sumPred);
    hm = hm.H;
end

function [results] = hmeasure(true_class,scores,varargin)

%HMEASURE Computes the H-measure and other classification performance metrics 
%
% H = HMEASURE(L,S,sr) takes as input the true class labels L and the 
% scores S obtained from deploying one or more classifiers to a given 
% dataset, and outputs several classification performance metrics, as well 
% as the Receiver Operating Characteristic curve.
% 
% H = HMEASURE(L,S) takes as input a column vector of true labels L and a 
% matrix S of scores, where each column corresponds to a classifier, and 
% each row to a datapoint. Class labels must be 0s and 1s only. It is 
% understood that class 0 objects tend to receive lower scores than class 1 
% objects, but to ensure consistency the signs of the scores of classifiers 
% that achieve AUC < 0.5 are reversed (and a warning is produced). 
% The output H is a structure where each field reports a certain 
% performance metric for each of the classifiers employed, in sequence. 
%
% H = HMEASURE(L,S,sr) additionally takes as input a severity ratio sr, 
% which represents how much more severe misclassifying a class 0 instance 
% is than misclassifying a class 1 instance. For instance, sr = 2 implies 
% that the cost of a False Positive is twice as large than that of a False 
% Negative. By default it is set to be reciprocal of relative class 
% frequency, i.e., sr = pi1/pi0, so that misclassifying the rare class is 
% considered a graver mistake (see Ref. [1] for details).
%
% [H,ROC,ROCH] = HMEASURE(L,S) additionally outputs the Receiver Operating
% Characteristic (ROC) curves for each classifier, in the form of an array
% of structures, where each structure has fields X and Y (see example). 
% It also outputs the convex hull thereof (ROCH).
%
% [H,ROC,ROCH,wH,wAUC] = HMEASURE(L,S) additionally outputs the implied 
% distribution over normalised costs by the H-measure (common across 
% classifiers) and the AUC implied cost distributions for each classifier
% (see ref [1] for an explanation). Output wH is a structure with fields 
% X and Y, and wAUC is an array of such structures, one per classifier.
% 
% NOTES: the measures currently implemented are the Area under the ROC 
% Curve (AUC), the H-measure, the Area under the Convex Hull of the ROC 
% Curve (AUCH), the Gini coefficient, the Kolmogorov-Smirnoff statistic, 
% the Minimum Weighted Loss (MWL), the Error Rate (ER), the F-measure, 
% Precision, Recall, as well as the four types of misclassification counts,
% True Positives (TP), False Positives  (FP), True Negatives (TN) and 
% False Negatives (FN). See references at the end of the help file.
%
% REFERENCES: 
% 
% [1]: Hand, D.J. and Anagnostopoulos, C. 2012. A better Beta for the H 
% measure of classification performance. Preprint, 
% <a href="http://arxiv.org/abs/1202.2564">arXiv:1202.2564v1</a>
% 
% [2]: Hand, D.J. 2009. Measuring classifier performance: a coherent 
% alternative to the area under the ROC curve. Machine Learning,77, 103-123
%
% [3]: Hand, D.J. 2010. Evaluating diagnostic tests: the area under the ROC 
% curve and the balance of errors. Statistics in Medicine, 29, 1502-1510.
%
% EXAMPLE:
%	% in this synthetic example, the class labels are:
%	y = [zeros(5,1);ones(5,1)];
%
%	% create some fake scores for two classifiers
%	% first classifier features lower scores for class 1
%	sA = [1.8,0.9,1.6,0.5,1,0.1,0.2,2.6,-0.4,-0.1]';
%	
%	% second classifier has ties
%	sB = [1,2,3,1,6,6,2,9,9,8]';
%	
%	scores = [sA,sB];
%	[results,ROC,ROCH,wH,wAUC] = hmeasure(y,scores);
%
%	% the remaining code reproduces the plot from the paper 
%	figure; subplot(2,2,1); hold on;
%	plot(ROC{1}.X,ROC{1}.Y,'b'); plot(ROC{2}.X,ROC{2}.Y,'r'); 
%	plot(0:0.1:1,0:0.1:1,'k--');
%	legstr = {'Classifier 1','Classifier 2','Trivial'};
%	legend(legstr,'Location','SouthEast');
%	plot(ROCH{1}.X,ROCH{1}.Y,'b:'); plot(ROCH{2}.X,ROCH{2}.Y,'r:'); 
%	title('ROC (continuous) and ROCH (dotted)');
%	
%	subplot(2,2,2); hold on; title('AUC cost weight distributions');
%	stem(wAUC{1}.X,wAUC{1}.Y,'b'); stem(wAUC{2}.X,wAUC{2}.Y,'r');
%	legend({'Classifier 1','Classifier 2'});
%	
%	subplot(2,2,3); hold on; plot(wH.X,wH.Y,'k');
%	title('H cost weight distributions'); legend({'All classifiers'});
%	
%	subplot(2,2,4); hold on;
%	plot(ksdensity(s1(y==0)),'b'); plot(ksdensity(s1(y==1)),'b:');
%	plot(ksdensity(s2(y==0)),'r'); plot(ksdensity(s2(y==1)),'r:');
%	legstr = {'s_A, class 0','s_B, class 1','s_B, class 0','s_B, class 1'}
%	legend(legstr);
	
%   Copyright 2012 Christoforos Anagnostopoulos, David Hand.
%   $Version: 0.1 $  $Date: 2012/05/19$

results = [];
% out = [];



%% catch errors in class vector 
% if iscell(true_class)
%     warning('Please enter class as a column vector, not a cell array')
%     return;
% end
% 
% if any(isnan(true_class))
%     warning('Error due to missing values in class labels');
%     return;
% end
% 
% % make sure class vector is a vector
% if size(true_class,1) < size(true_class,2)
%     true_class = true_class';
%     warning('Class row vector has been transposed to a column vector');
% end

classes = unique(true_class);
% no_of_classes = max(size(classes));

% if no_of_classes > 2
%     warning('More than two classes present, but code can only handle binary classification.');
%     return;
% end

% if no_of_classes == 1
%     warning('Only one class is present in the dataset.');
%     return;
% end

true_class = true_class==classes(2);

%% catch errors in score matrix
[n,k] = size(scores);
% if n < k
%     warning('Ensure that the score matrix is correct - currently has more columns (=classifiers) than rows (=datapoints)');
% end

% if n ~= size(true_class,1)
%     warning('Class vector provided has different number of entries than respective classifier scores.');
%     return
% end
 
% if any(any(isnan(scores)))
%     warning('Missing entries detected in score matrix. ...
%               Respective rows will be disregarded.');
% end
% if k > 1
% 	incomplete_cases = any(isnan(scores'));
% else
% 	incomplete_cases = isnan(scores);
% end
% scores = scores(~incomplete_cases,:);

n1 = sum(true_class);
n0 = n-n1;
pi0 = n0/n;
pi1 = n1/n;
if isempty(varargin)
    SR = pi1/pi0;
else
    SR = varargin{1};
end
for j = 1:k
    s = scores(:,j);
%     sc = sort(s);
   
    [F1,F0,~,~,S] = getScoreDistributions(true_class,s);


%     H = 0;

    % restrict to upper convex hull by considering ROC above diagonal only
    upper = max([1-F0,1-F1],[],2);
    if isequal(upper,1-F0)
	chull_points = [S;1];
    else
        chull_points = sort(convhull(1-F0,upper,'simplify',true),...
            'descend');
        chull_points(end) = [];
    end
    
    G0 = 1-F0(chull_points);
    G1 = 1-F1(chull_points);
    hc = size(chull_points,1);
%     sG0 = [0;G0(2:hc) - G0(1:(hc-1))];
%     sG1 = [0;G1(2:hc) - G1(1:(hc-1))];
% 
%     % get sorted scoring densities
%     s_class0 = sort(s(true_class==0));
%     s_class1 = sort(s(true_class==1));



    % Calculate the LHshape1 value
    cost = 1:(hc+1);
    b0 = 1:(hc+1);
    b1 = 1:(hc+1);
    
    % extract shape
    if (SR > 0)
      shape1 = 2;
      shape2 = 1+(shape1-1)*1/SR;
    end
    if (SR < 0)
      shape1 = pi1+1;
      shape2 = pi0+1;
    end
    cost(1) = 0;
    cost(hc+1) = 1;

    b00 = beta(shape1,shape2);
    b10 = beta(1+shape1,shape2);
    b01 = beta(shape1,1+shape2);


    b0(1) = betacdf(cost(1), 1+shape1, shape2)*b10/b00;

    b1(1) = betacdf(cost(1), shape1, (1+shape2))*b01/b00;

    b0(hc+1) = betacdf(cost(hc+1), 1+shape1, shape2)*b10/b00;

    b1(hc+1) = betacdf(cost(hc+1), shape1, 1+shape2)*b01/b00;

    
    cost(2:hc) = pi1*(G1(2:hc)-G1(1:hc-1))./(pi0*(G0(2:hc)-G0(1:hc-1))+...
        pi1*(G1(2:hc)-G1(1:hc-1)));
    
    b0(2:hc) = betacdf(cost(2:hc), 1+shape1, shape2)*b10/b00;
    
    b1(2:hc) = betacdf(cost(2:hc), shape1, (1+shape2))*b01/b00;
    
    %%% NB: can become massively faster
%     for i = 2:hc
%       cost(i) = pi1*(G1(i)-G1(i-1)) / (pi0*(G0(i)-G0(i-1)) + ...
%                   pi1*(G1(i)-G1(i-1)));
% 
%       b0(i) = betacdf(cost(i), 1+shape1, shape2)*b10/b00;
% 
%       b1(i) = betacdf(cost(i), shape1, (1+shape2))*b01/b00;
%     end


%     
%     LHshape1_parts = pi0*(ones(52,1)-G0)'.*(b0(2:hc+1)-b0(1:hc)) + ...
%                         pi1*G1'.*(b1(2:hc+1)-b1(1:hc));
%     LHshape1 = sum(LHshape1_parts);

    LHshape1 = 0;
    for i = 1:hc
      LHshape1 = LHshape1 + pi0*(1-G0(i))*(b0((i+1))-b0(i)) + pi1*G1(i)*...
          (b1((i+1))-b1(i));
    end
    
%     b0_parfor = b0(2:hc+1);
%     b1_parfor = b1(2:hc+1);
%     parfor i = 1:hc
%         LHshape1_parts(i) = pi0*(1-G0(i))*(b0_parfor(i)-b0(i)) + ...
%                       pi1*G1(i)*(b1_parfor(i)-b1(i));
%     end
%     LHshape1 = sum(LHshape1_parts);

    B0 = betacdf(pi1, (1+shape1), shape2)*b10/b00;

    B1 = betacdf(1, shape1, (1+shape2))*b01/b00 - betacdf(pi1, shape1, ...
        (1+shape2))*b01/b00;

    H = 1 - LHshape1/(pi0*B0 + pi1*B1);

    % output results
    results.H(j) = H;


end
end

function [F1,F0,s1,s0,S] = getScoreDistributions(y,s)

% 	n = size(y,1);
	n1 = sum(y);
	n0 = sum(1-y);
	% sc is a column whose ith entry is the ith sorted unique score        
	sc = unique(s);
	
%         class_unique = y();
        S = size(sc,1);
%         s1 = zeros(S,1);
%         s0 = zeros(S,1);
    ind_now = zeros(S,1);
    for i = 1:S
		% this is the index of appearances of sc(i) in the original...
        % score vector
		ind_now(i) = find(s==sc(i));
% 		s1(i) = sum(y(ind_now));
% 		s0(i) = sum(1-y(ind_now));
    end
    
    s1 = y(ind_now);
	s0 = 1-y(ind_now);
    
	s1 = s1./n1;
	s0 = s0./n0;
	s1 = [0;s1;1-sum(s1)];
	s0 = [0;s0;1-sum(s0)];
    	F1 = cumsum(s1);
	F0 = cumsum(s0);
end

