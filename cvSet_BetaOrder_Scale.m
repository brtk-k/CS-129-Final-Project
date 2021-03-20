% Function importing the cross-validation set data, 
% generating the additional polynomial features up to order beta, and 
% scaling the features using the mu and sigma from the training set
% feature scaling (can also be used for the test set data)
function [X, y, m] = cvSet_BetaOrder_Scale(cvSetFileName, beta, mu_train, sigma_train)
    % Import .csv data
    data_cv = readtable(cvSetFileName);
    
    % Construct X and y
    X = table2array(data_cv(:, 8:10));   % Extract the endpoint data
    y = table2array(data_cv(:, 1:6));    % Extract the q1-q6 angle data 
    m = size(X, 1); % Number of training examples

    % Generate polynomial features
    X = generateHigherOrder(X, beta);
    
    % Scale the features by the mu and sigma vectors computed for the
    % training set (see "trainSet_BetaOrder_Scale.m" for explanation why
    % the first 3 features are not modified)
    X(:,4:end) = bsxfun(@minus, X(:,4:end), mu_train(4:end));
    X(:,4:end) = bsxfun(@rdivide, X(:,4:end), sigma_train(4:end));
end