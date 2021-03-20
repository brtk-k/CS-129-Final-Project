% Function importing the training set data, generating the additional
% polynomial features up to order beta, and scaling the features
function [X, y, m, mu, sigma] = trainSet_BetaOrder_Scale(trainSetFileName, beta)
    % Import .csv data
    data_train = readtable(trainSetFileName);
    
    % Construct X and y
    X = table2array(data_train(:, 8:10));   % Extract the endpoint data
    y = table2array(data_train(:, 1:6));    % Extract the q1-q6 angle data 
    m = size(X, 1); % Number of training examples

    % Generate polynomial features
    X = generateHigherOrder(X, beta);

    % Perform feature scaling
    [X_normal, mu, sigma] = featureNormalize(X);  % Compute normalized features, together with their mu and sigma

    % Only normalize the higher-order features, because the first-order features are the Euclidean points 
    % used explicitly in the cost function, so they should not be transformed (they are on the same order
    % of magnitude as the scaled polynomial features anyways)
    X(:,4:end) = X_normal(:,4:end);
end