%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CS 129 - Final Project - Target-based Prediction of Robotic Arm Input Angles %%
%%%%%%%%%%%%%    Bartosz Kaczmarski, Victoria Ou, Daniel Cha     %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars;
close all;
clc;

%% Construct the F and J_F functions
% Define the number of degrees of freedom
nDOF = 7;

% Construct the symbolic expressions for F and J_F
[F, J_F, q] = symbolicF(nDOF);

% Convert to function handles
numF   = matlabFunction(F, 'vars', {q});
numJ_F = matlabFunction(J_F, 'vars', {q});

%% Data set filenames
trainSetFileName = 'LaundryTableProcessed.csv';  % File name for the training data set
cvSetFileName   = 'LaundryTable2Processed.csv';  % File name for the CV data set
testSetFileName   = 'ObjectTableProcessed.csv';  % File name for the CV data set

%% Import training set data, generate polynomial features and perform feature scaling
% Set maximum polynomial order of the input features
beta = 3;

% Import training set, generate polynomial features up to order beta and perform feature scaling
[X, y, m, mu, sigma] = trainSet_BetaOrder_Scale(trainSetFileName, beta);

%% Define the neural network                       
% User-defined neural network architecture parameters
L                 = 4;           % Number of neural network layers
hidden_layer_size = 10;           % Number of units in each hidden layer = H
hidden_layer_N    = L - 2;       % Number of hidden layers (useful parameter)

% Neural network parameters defined by X and nDOF
input_layer_size  = size(X, 2);  % Number of units in the input layer (excluding the bias unit)
nOut              = nDOF - 1;    % Dimension of the angle output q 
                                 % (the map F and Jacobian of F are not functions of q7)

% Minimization parameters
lambda = 0.0003;  % Regularization parameter
N_iter = 5000;    % Number of iterations

% Other parameters
zeta = pi;                       % Scaling factor for the output layer
gradientChecking = false;        % Set to true if gradient checking is to be performed
learningCurves   = true;         % Set to true if learning curves are to be generated

%% Train the deep neural network
[Theta, Jcost_vector] = trainDNN(input_layer_size, hidden_layer_size, hidden_layer_N, nOut,...
                                 X, y, lambda, zeta, N_iter, numF, numJ_F, gradientChecking);                           
%%%%%%%%%%%%%%%%%%%%%%
%% Model evaluation %%
%%%%%%%%%%%%%%%%%%%%%%
%% Training set %%
% Compute the average error, the error vector (both as distance deviation),
% the average J_error and the x predictions
[avgError_train, error_train, avgJ_Error_train, J_error_train, xPred_train] = computeSetError(Theta, X, zeta, numF);

%% Cross-validation set %%
% Import CV set, generate higher-order CV features up to order beta and perform feature scaling
% using the mu and sigma obtained for the training set features
[Xval, yval, m_val] = cvSet_BetaOrder_Scale(cvSetFileName, beta, mu, sigma);

%% Compute the average error, the error vector (both as distance deviation),
% the average J_error and the x predictions
[avgError_val, error_val, avgJ_Error_val, J_error_val, xPred_val] = computeSetError(Theta, Xval, zeta, numF);

%% Test set %%
% Import test set, generate higher-order CV features up to order beta and perform feature scaling
% using the mu and sigma obtained for the training set features
[Xtest, ytest, m_test] = cvSet_BetaOrder_Scale(testSetFileName, beta, mu, sigma);

% Compute the average error, the error vector (both as distance deviation),
% the average J_error and the x predictions
[avgError_test, error_test, avgJ_Error_test, J_error_test, xPred_test] = computeSetError(Theta, Xtest, zeta, numF);

%% Artificial test set %%
Xtest_arti = [];
x_step = 0.02; y_step = 0.02; z_step = 0.02;
x_bounds = [-0.3, 0.45]; y_bounds = [-0.15, 0.5]; z_bounds = [-0.25, 0.4];
for x = x_bounds(1):x_step:x_bounds(2)
    for y = y_bounds(1):y_step:y_bounds(2)
        for z = z_bounds(1):z_step:z_bounds(2)
            new_x = [x, y, z];
            Xtest_arti = [Xtest_arti; new_x];
        end
    end
end

% Generate polynomial features
Xtest_arti = generateHigherOrder(Xtest_arti, beta);
    
% Scale the features by the mu and sigma vectors computed for the
% training set (see "trainSet_BetaOrder_Scale.m" for explanation why
% the first 3 features are not modified)
Xtest_arti(:,4:end) = bsxfun(@minus, Xtest_arti(:,4:end), mu(4:end));
Xtest_arti(:,4:end) = bsxfun(@rdivide, Xtest_arti(:,4:end), sigma(4:end));

% Compute the average error, the error vector (both as distance deviation),
% the average J_error and the x predictions
[avgError_test_arti, error_test_arti, avgJ_Error_test_arti, J_error_test_arti, xPred_test_arti] = ...
    computeSetError(Theta, Xtest_arti, zeta, numF);

%% Learning curve generation
m_step = 10; % Step size for learning curve construction
if learningCurves
    m_vec = [1:m_step:m, m]; % Append the maximum number of examples as well
    i = 1;
    % Train the model for all m in m_vec, compute the train and CV J-errors
    for m_learn = m_vec
        if m_learn < m
            Xtrain_learn = X(1:m_learn,:);

            [Theta_learn, Jcost_vector_learn] = trainDNN(input_layer_size, hidden_layer_size, hidden_layer_N, nOut,...
                                                   Xtrain_learn, y, lambda, zeta, N_iter, numF, numJ_F, gradientChecking);
            [avgErrorTrain_learn, errorTrain_learn, avgJ_ErrorTrain_learn, J_errorTrain_learn, xPredTrain_learn] = computeSetError(Theta_learn, Xtrain_learn, zeta, numF);
            [avgErrorVal_learn, errorVal_learn, avgJ_ErrorVal_learn, J_errorVal_learn, xPredVal_learn] = computeSetError(Theta_learn, Xval, zeta, numF);

            avgJErrorTrainVec(i) = avgJ_ErrorTrain_learn;
            avgJErrorValVec(i)   = avgJ_ErrorVal_learn;
        else
            load('L_4_H_10_beta_3_lam_0.0003_zeta_3.1416_ITER_5000_train_LaundryTableProcessed_cv_LaundryTable2Processed.mat');
            avgJErrorTrainVec(length(m_vec)) = avgJ_Error_train;
            avgJErrorValVec(length(m_vec)) = avgJ_Error_val;
        end
        i = i + 1;
    end
end

%% Save MATLAB workspace
workspace_filename = strcat("L_", string(L), "_H_", string(hidden_layer_size), "_beta_",...
                            string(beta), "_lam_", string(lambda), "_zeta_", string(zeta),...
                            "_ITER_", string(N_iter), "_train_", trainSetFileName(1:end - 4),...
                            "_cv_", cvSetFileName(1:end - 4), ".mat");
save(workspace_filename);

%%%%%%%%%%%%%%
%% Plotting %%
%%%%%%%%%%%%%%
%% Plot the training set data and predictions
close all
figure(1);
tile = tiledlayout(1,2);
nexttile
scatter3(X(:,1), X(:,2), X(:,3), 40, 'k')
hold on;
scatter3(xPred_train(:,1), xPred_train(:,2), xPred_train(:,3), 30, 'rs')
axis equal;
xlim([-0.15,0.3]);
ylim([0,0.3]);
zlim([0,0.3]);
PlotFormatting(strcat("Training set: $\textrm{Mean}_k\left\{\left\|\hat{\mathbf{x}}^{(k)} - \mathbf{x}^{(k)}\right\|\right\} \approx", string(avgError_train), "$"),...
                '$x_1$', '$x_2$', '$x_3$', true, {'$\mathbf{x}_{\textrm{train}}$ data','$\hat{\mathbf{x}}_{\textrm{train}}$ predictions'}, 'northwest', 20);
view([-34.2677, 22.1874]);

%% Plot the CV set data and predictions
nexttile
scatter3(Xval(:,1), Xval(:,2), Xval(:,3), 40, 'k')
hold on;
scatter3(xPred_val(:,1), xPred_val(:,2), xPred_val(:,3), 30, 'rs')
axis equal;
xlim([-0.15,0.3]);
ylim([0,0.3]);
zlim([0,0.3]);
PlotFormatting(strcat("CV set: $\textrm{Mean}_k\left\{\left\|\hat{\mathbf{x}}^{(k)} - \mathbf{x}^{(k)}\right\|\right\} \approx", string(avgError_val), "$"),...
                '$x_1$', '$x_2$', '$x_3$', true, {'$\mathbf{x}_{\textrm{cv}}$ data','$\hat{\mathbf{x}}_{\textrm{cv}}$ predictions'}, 'northwest', 20)
view([-34.2677, 22.1874]);


%% Plot the J-cost value as a function of the number of iterations  
figure(2)
loglog(1:N_iter, Jcost_vector)
PlotFormatting('Cost as a function of number of iterations',...
               'Number of iterations', '$J$', '', false, {''}, '', 16);
           
%% Plot the Test set data and predictions
figure(3)
scatter3(Xtest(:,1), Xtest(:,2), Xtest(:,3), 40, 'k')
hold on;
scatter3(xPred_test(:,1), xPred_test(:,2), xPred_test(:,3), 30, 'rs')
axis equal;
xlim([-0.15,0.3]);
ylim([0,0.3]);
zlim([0,0.3]);
PlotFormatting(strcat("Test set: $\textrm{Mean}_k\left\{\left\|\hat{\mathbf{x}}^{(k)} - \mathbf{x}^{(k)}\right\|\right\} \approx", string(avgError_test), "$"),...
                '$x_1$', '$x_2$', '$x_3$', true, {'$\mathbf{x}_{\textrm{test}}$ data','$\hat{\mathbf{x}}_{\textrm{test}}$ predictions'}, 'northwest', 20)
view([-34.2677, 22.1874]);       

%% Plot the artificial test set predictions
figure(4)
hold on;
scatter3(xPred_test_arti(:,1), xPred_test_arti(:,2), xPred_test_arti(:,3), 1, 'ro','MarkerFaceColor','k','MarkerEdgeColor','r','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1)
axis equal;
xlim([-0.3,0.45]);
ylim([-0.15,0.5]);
zlim([-0.25,0.4]);
PlotFormatting("Reachable space $\hat{\mathcal{X}}$ of the CV-minimized model $M^*$",...
                '$x_1$', '$x_2$', '$x_3$', false, {''}, '', 20)
view([-34.2677, 22.1874]); 

%% Plot learning curves
if learningCurves
    figure(3)
    hold on;
    plot(m_vec(2:end), avgJErrorTrainVec(2:end))
    plot(m_vec(2:end), avgJErrorValVec(2:end))
    set(gca, 'YScale', 'log');
    PlotFormatting('Learning Curves',...
                   '$m$ (Number of training examples)', '$J_\textrm{train}$, $J_\textrm{cv}$', '', true, {'$J_\textrm{train}$', '$J_\textrm{cv}$'}, 'southwest', 16);
end

