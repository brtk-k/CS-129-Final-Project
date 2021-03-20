% Function computing the mean Euclidean distance deviation (avgError),
% the Euclidean distance deviation vector (error), the average J-error 
% (avgJ_Error), the J-error vector (J_error) and the endpoint predictions x
function [avgError, error, avgJ_Error, J_error, xPred] = computeSetError(Theta, X, zeta, numF)
    % Compute the q and x predictions, distance errors and J-errors for all examples in the set
    qPred = predictAngles_DNN(Theta, X, zeta);
    xPred = numF(qPred')';
    error = vecnorm(xPred - X(:,1:3), 2, 2);
    J_error = vecnorm(xPred - X(:,1:3), 2, 2).^2;

    % Compute the average distance error
    avgError = mean(error);
    avgJ_Error = 1/(2*size(J_error, 1))*sum(J_error);
end