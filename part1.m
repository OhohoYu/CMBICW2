close all;
clear all;


% Sample size, means
sample_size = 25;
mu1 = 1;
mu2 = 1.5;

% Stochastic component 
mu_comp = 0;
std_comp = 0.25;

y1 = 0.25.*randn(sample_size,1) + mu1;
y2 = 0.25.*randn(sample_size,1) + mu2;

mean_y1 = mean(y1);
mean_y2 = mean(y2);
std_y1 = std(y1);
std_y2 = std(y2);

test_vect = [(mean_y1 - mu1)^2 < 0.1, (mean_y2 - mu2)^2 < 0.1, (std_y1 - std_comp)^2 < 0.1, (std_y2 - std_comp)^2 < 0.1];
if(all(test_vect))
    disp("Stds and means similar to original components");
end

[hypothesis,p_val,conf_interval,test_stats] = ttest2(y1,y2);
if (hypothesis == 1)
    disp("Null Hypothesis rejected");
end

% Q.1 c)i)
y = [y1;y2];
design_matrix = [repmat([0 1],sample_size,1); repmat([1 0],sample_size,1)];
[n,p] = size(design_matrix);
% Sample size is now 50 and column space is 2 because we know there are 2
% columns in the design matrix, each linearly independent of the other

% Q.1 c)ii)
projection_matrix = design_matrix * pinv(design_matrix' * design_matrix) * design_matrix';
% To show that P_x has the key property (orthogonality) we compare the
% angle of y_hat and e_hat to find the 
trace_y = trace(projection_matrix);

% Q.1 c)iii)
y_hat = projection_matrix * y;

% Q.1 c)iv)
r_x = eye(n) - projection_matrix;

% Q.1 c)v)
% Dimension of e_hat is any dimension perpendicular to the plane covered by
% the column space of X. Therefore it can be any dimenion (max dimension =
% 50) - dimension of X (2). dim(e_hat) = 50-2 = 48
e_hat = r_x * y;

% Q.1 c)vi)
% Angle between 2 vectors: cos(theta) = a.b / ||a|| ||b||
theta = acos(e_hat' * y_hat / (norm(e_hat) * norm(y_hat)));
if(abs(theta) < 10)
    result = sprintf("Angle is %d , which is very close to 0, therefore almost perpendicular to column space of X",theta); 
    disp(result);
end

% Q.1 c)vii)
beta_hat = pinv(design_matrix' * design_matrix) * design_matrix' * y;
if((beta_hat - [mu2,mu1])^2 < 0.1)
    disp("Beta very close to true values");
end
disp("Beta hat:");
disp(beta_hat);

% Q.1 c)viii)
intermediate = e_hat' * e_hat / (n-p);
sigma = sqrt(intermediate);
if ((sigma - std_comp)^2 < 0.05)
    disp("Estimated standard deviation is close to true values");
end

% Q.1 c)ix)
covariance_mat = intermediate * pinv(design_matrix' * design_matrix);
indep_mat = (covariance_mat > 0);
if(all(diag(indep_mat)))
    disp("Columns are independent");
end

% Q.1 c)x)
% The values for both groups are equal
hyp_vector = [1;-1];



