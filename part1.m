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
y = [y2;y1];
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
    fprintf("Angle is %d , which is very close to 0, therefore almost perpendicular to column space of X",theta); 
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
hyp_vector = [1,-1]';

% Because the hypothesis vector is a single dimension, we can calculate the
% f-statistic easily


% Q.1 c)xi)

orth_space = null(hyp_vector');
X0 = (orth_space' * design_matrix')';

[y_hat_hypothesis, projection_hypothesis, beta_hat_hyp, error_hat_hyp, dim_hyp] = calculatePredicted(X0, y);

% Calculating the error (additional error in C(X) between y_hat and
% y_hat_hyp)

error_hypothesis = norm(y_hat-y_hat_hypothesis);

t_df_statistic = (hyp_vector' * beta_hat) / (hyp_vector' * covariance_mat * hyp_vector);

v_1 = trace(projection_matrix - projection_hypothesis);
v_2 = trace(r_x);


numerator = (y' * (eye(n) - projection_hypothesis)* y - y'* r_x * y)/v_1;
denominator = (y' * r_x * y) / v_2;
diff = numerator / denominator;

f_statistic = (diff) / (numerator/ v_2);

% Q.1 c)xii)
t_statistic = (hyp_vector' * beta_hat) / (sqrt(hyp_vector' * covariance_mat * hyp_vector));

% Q.1 c)xiv)
true_error = y - y_hat;
projection_error = projection_matrix * true_error;

% Q.1 c)xv)
projection_e_error = r_x * true_error;


% Q.1 d)i
X_d = [ones(sample_size,1),repmat([0 1],sample_size,1); ones(sample_size,1),repmat([1 0],sample_size,1)];
[~,proj_d,~,~,~] = calculatePredicted(X_d,y);

% Q.1 d)iii)
hyp_d = [0,1,-1]';
[t_d,f_d] = calculateStatistics(X_d,y, hyp_d);

% Q.1 e)i)
X_e = [repmat([1 0], sample_size,1);repmat([1 1], sample_size, 1)];
[~,proj_e,beta_hat_e,~,p_e] = calculatePredicted(X_e,y);

% Q.1 e)ii)
hyp_e = [0, 1]';
[t_e, f_e] = calculateStatistics(X_e,y,hyp_e);

% Q.2 a)
% NEW SAMPLES
new_y1 = 0.25.*randn(sample_size,1) + mu1;
new_y2 = 0.25.*randn(sample_size,1) + mu2;

[~,~,~,test_stats_2] = ttest(new_y1,new_y2);

% Q.2 b)

X_2b = [X_e, [eye(sample_size);eye(sample_size)]];
[~,~,~,e_hat_2b,p_2b] = calculatePredicted(X_2b,y);

% Q.2 b)ii)
% Save time rather than writing all the zeros
hyp_2b = zeros(sample_size + 2, 1);
hyp_2b(2) = 1;

[t_stat_2b, f_stat_2b] = calculateStatistics(X_2b,[new_y2;new_y1],hyp_2b);


function [y_hat, projection, beta_hat, error_hat, p] = calculatePredicted(X,Y)
    projection = X * pinv(X' * X) * X';
    beta_hat = pinv(X' * X) * X' * Y;
    y_hat  = projection' * Y;
    [n,~] = size(X);
    p = trace(projection);
    r_x = (eye(n) - projection);
    error_hat = r_x * Y;
end

function [t_stat, f_stat] = calculateStatistics(X, Y, hyp)
    [n,~] = size(X);
    [~,proj,beta_hat,e_hat,p] = calculatePredicted(X,Y);
    disp(beta_hat);
    sigma_sq = e_hat' * e_hat / (n - p);
    orth_space = null(hyp');
    X0 = (orth_space' * X')';
    [~, proj_hyp, ~, ~, ~] = calculatePredicted(X0,Y);
    std_beta = sigma_sq  * pinv(X' * X);
    t_stat = (hyp' * beta_hat) / (sqrt(hyp' * std_beta * hyp));
    v_1 = trace(proj - proj_hyp);
    v_2 = trace(eye(n) - proj);
    numerator = ((Y' * (eye(n) - proj_hyp) * Y) - (Y' * (eye(n) - proj) * Y)) / v_1;
    denominator = (Y' * (eye(n) - proj) * Y) / v_2;
    f_stat = numerator / denominator;
end
