%% Hyperparameter tuning error visualization on the cross-validation set
%% based on the training results for all models in the hyperparameter sweep
clearvars;
close all;
clc;

%% Move to the directory with workspace data files
cd HyperparameterSweepData

%% Extract the relevant file names in the directory
matFiles = dir('*.mat');

matFileNames = cell(length(matFiles),1);
for i = 1:length(matFiles)
    matFileNames{i} = matFiles(i).name;
end

reg_indices = regexp(matFileNames,'L_\d+_H_\d+_beta_\d+_lam_((\d+\.)|(\d+e-))\d+_zeta_\d+\.\d+_ITER_\d+_\w+\.mat');
filename_indices = find(cell2mat(reg_indices));

dataFileNames = matFileNames(filename_indices);

% Move back up to the original directory
cd ..\

%% Load the hyperparameter and J-error data from the relevant data files
L_vec = [];
H_vec = [];
beta_vec = [];
lambda_vec = [];
zeta_vec = [];
    
for i = 1:length(dataFileNames)
    load(dataFileNames{i});
    L_vec = [L_vec, L];
    H_vec = [H_vec, hidden_layer_size];
    beta_vec = [beta_vec, beta];
    lambda_vec = [lambda_vec, lambda];
    zeta_vec = [zeta_vec, zeta];
    
    JError_train(i) = avgJ_Error_train; 
    JError_val(i)   = avgJ_Error_val;
end

%% Process the data to facilitate plotting the average CV error
%% as a function of the different hyperparameters
[L_list, minJ_L]           = minCurve(L_vec, JError_val);
[H_list, minJ_H]           = minCurve(H_vec, JError_val);
[beta_list, minJ_beta]     = minCurve(beta_vec, JError_val);
[lambda_list, minJ_lambda] = minCurve(lambda_vec, JError_val);
[zeta_list, minJ_zeta]     = minCurve(zeta_vec, JError_val);

%% Plot the average CV error as a function of the varied hyperparameters
close all;
figure('Renderer', 'painters', 'Position', [600 400 900 620])
tiledlayout('flow');
nexttile
plot(L_vec, JError_val, 'o','MarkerSize',3);
hold on;
plot(L_list, minJ_L, 's-','MarkerSize',8);
set(gca, 'YScale', 'log')
PlotFormattingHyperparam('', '$L$', '$\bar{J}_\textrm{cv}$', '', true, {'CV Error for all models','Minimum error curve'}, 'northwest', 16, true)
ylim([1e-5, 1e-2]);
yticks([1e-5, 1e-4, 1e-3,1e-2])
%%
nexttile
plot(H_vec, JError_val, 'o','MarkerSize',3);
hold on;
plot(H_list, minJ_H, 's-','MarkerSize',8);
set(gca, 'YScale', 'log')
PlotFormattingHyperparam('', '$H$', '$\bar{J}_\textrm{cv}$', '', false, '', '', 16, false)
ylim([1e-5, 1e-2]);
yticks([1e-5, 1e-4, 1e-3,1e-2])
%%
nexttile
plot(beta_vec, JError_val, 'o','MarkerSize',3);
hold on;
plot(beta_list(1:5), minJ_beta(1:5), 's-','MarkerSize',8);
set(gca, 'YScale', 'log')
PlotFormattingHyperparam('', '$\beta$', '$\bar{J}_\textrm{cv}$', '', false, '', '', 16, true)
xlim([1, 5]);
ylim([1e-5, 1e-2]);
yticks([1e-5, 1e-4, 1e-3,1e-2])
%%
nexttile
loglog(lambda_vec, JError_val, 'o','MarkerSize',3);
hold on;
plot(lambda_list, minJ_lambda, 's-','MarkerSize',8);
PlotFormattingHyperparam('', '$\lambda$', '$\bar{J}_\textrm{cv}$', '', false, '', '', 16, false)
xlim([1e-5, 0.01]);
ylim([1e-5, 1e-2]);
xticks([1e-5, 1e-4, 1e-3,1e-2]);
yticks([1e-5, 1e-4, 1e-3,1e-2])

sgtitle('Average CV error $\bar{J}_\textrm{cv}$ as a function of $\{L,H,\beta,\lambda\}$ for all 360 models','Interpreter','latex','FontSize',16);




