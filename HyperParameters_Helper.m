%% Hyperparameter sweep helper function
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

%% Set number of iterations
N_iter = 2000;

%% Define the hyperparameter lists
L_list = [3, 4, 5];
H_list = [5, 7, 10, 20];
beta_list = [1, 2, 3, 4, 5];
lambda_list = [0.00003, 0.0001, 0.0003, 0.0006, 0.001, 0.003];
zeta_list = pi;

%% Iterate through the different hyperparameter sets
for L = L_list
    for hidden_layer_size = H_list
        for beta = beta_list
            for lambda = lambda_list
                for zeta = zeta_list
                    disp(['L = ', num2str(L)]);
                    disp(['H = ', num2str(hidden_layer_size)]);
                    disp(['beta = ', num2str(beta)]);
                    disp(['lambda = ', num2str(lambda)]);
                    disp(['zeta = ', num2str(zeta)]);
                    run("roboticArm_DNN_ForRunning.m");
                end
            end
        end
    end
end
