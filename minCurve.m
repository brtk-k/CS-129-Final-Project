% Function computing the minimal curve for a hyperparameter data vector
function [paramList, minJ_Curve] = minCurve(param_vec, JError_val)
    paramList = unique(param_vec);
    for i = 1:length(paramList)
        minJ_Curve(i) = min(JError_val(param_vec == paramList(i)));
    end
end