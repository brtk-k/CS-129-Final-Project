%% Auxillary function akin to roboticArm_DNN.m to facilitate prototyping
% Import training set, generate polynomial features up to order beta and perform feature scaling
[X, y, m, mu, sigma] = trainSet_BetaOrder_Scale(trainSetFileName, beta);

%% Define the neural network
hidden_layer_N    = L - 2;       % Number of hidden layers (useful parameter)

% Neural network parameters defined by X and nDOF
input_layer_size  = size(X, 2);  % Number of units in the input layer (excluding the bias unit)
nOut              = nDOF - 1;    % Dimension of the angle output q 
                                 % (the map F and Jacobian of F are not functions of q7)

% Other parameters
gradientChecking = false;        % Set to true if gradient checking is to be performed
learningCurves   = false;        % Set to true if learning curves are to be generated

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

% Compute the average error, the error vector (both as distance deviation),
% the average J_error and the x predictions
[avgError_val, error_val, avgJ_Error_val, J_error_val, xPred_val] = computeSetError(Theta, Xval, zeta, numF);

%% Learning curve generation
if learningCurves
    m_step = 50;
    m_vec = [1:m_step:m, m];
    i = 1;
    for m_learn = m_vec
        Xtrain_learn = X(1:m_learn,:);
        [Theta_learn, Jcost_vector_learn] = trainDNN(input_layer_size, hidden_layer_size, hidden_layer_N, nOut,...
                                               Xtrain_learn, y, lambda, zeta, N_iter, numF, numJ_F, gradientChecking);
        [avgErrorTrain_learn, errorTrain_learn, avgJ_ErrorTrain_learn, J_errorTrain_learn, xPredTrain_learn] = computeSetError(Theta_learn, Xtrain_learn, zeta, numF);
        [avgErrorVal_learn, errorVal_learn, avgJ_ErrorVal_learn, J_errorVal_learn, xPredVal_learn] = computeSetError(Theta_learn, Xval, zeta, numF);

        avgJErrorTrainVec(i) = avgJ_ErrorTrain_learn;
        avgJErrorValVec(i)   = avgJ_ErrorVal_learn;
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
PlotFormatting(strcat("Training set: $\textrm{Mean}_k\left\{\left\|\hat{\mathbf{x}}^{(k)} - \mathbf{x}^{(k)}\right\|\right\} \approx", string(avgError_train), "$"),...
                '$x_1$', '$x_2$', '$x_3$', true, {'$\mathbf{x}_{\textrm{train}}$ data','$\hat{\mathbf{x}}_{\textrm{train}}$ predictions'}, 'northwest', 20);
            
%% Plot the CV set data and predictions
nexttile
scatter3(Xval(:,1), Xval(:,2), Xval(:,3), 40, 'k')
hold on;
scatter3(xPred_val(:,1), xPred_val(:,2), xPred_val(:,3), 30, 'rs')
axis equal;
PlotFormatting(strcat("CV set: $\textrm{Mean}_k\left\{\left\|\hat{\mathbf{x}}^{(k)} - \mathbf{x}^{(k)}\right\|\right\} \approx", string(avgError_val), "$"),...
                '$x_1$', '$x_2$', '$x_3$', true, {'$\mathbf{x}_{\textrm{cv}}$ data','$\hat{\mathbf{x}}_{\textrm{cv}}$ predictions'}, 'northwest', 20)

%% Plot the J-cost value as a function of the number of iterations  
figure(2)
plot(1:N_iter, Jcost_vector)
PlotFormatting('Cost as a function of number of iterations',...
               'Number of iterations', '$J$', '', false, {''}, '', 16);
           
%% Plot learning curves
if learningCurves
    figure(3)
    hold on;
    plot(m_vec, avgJErrorTrainVec)
    plot(m_vec, avgJErrorValVec)
    PlotFormatting('Learning Curves',...
                   'Number of training examples', '$J$', '', true, {'$J_\textrm{train}$', '$J_\textrm{cv}$'}, 'northeast', 16);
end
