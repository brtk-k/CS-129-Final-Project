%% Function from the "featureNormalize.m" file on Coursera coding assignments
function [X_norm, mu, sigma] = featureNormalize(X)
    % Mean-normalization
    mu = mean(X);
    X_norm = bsxfun(@minus, X, mu);
    
    % Scale by standard deviation
    sigma = std(X_norm);
    X_norm = bsxfun(@rdivide, X_norm, sigma);
end
