%% Function modified for the general DNN case based on the "predict.m" file on Coursera coding assignments
function q = predictAngles_DNN(Theta, X, zeta)
    % Number of training examples
    m = size(X, 1);

    % Compute the q-prediction via feed-forward propagation
    hidden_layer_N = size(Theta, 1) - 1;
    h = cell(hidden_layer_N, 1);
    h{1} = sigmoid([ones(m, 1) X] * Theta{1}');
    for i = 2:hidden_layer_N
        h{i} = sigmoid([ones(m, 1) h{i - 1}] * Theta{i}');
    end
    q = zeta*(sigmoid([ones(m, 1) h{end}] * Theta{end}') - 1/2);
end