%% Function modified and generalized based on the "Neural Network Learning.ipynb" notebook on Coursera
function [Theta, Jcost_vector] = trainDNN(input_layer_size, hidden_layer_size, hidden_layer_N, nOut, X, y, lambda, zeta, N_iter, numF, numJ_F, gradientChecking)
    %% Randomly initialize all Theta weight matrices for the L-layer NN
    initial_Theta = cell(hidden_layer_N + 1, 1);
    initial_Theta{1} = randInitializeWeights(input_layer_size, hidden_layer_size);
    for i = 2:hidden_layer_N
        initial_Theta{i} = randInitializeWeights(hidden_layer_size, hidden_layer_size);
    end
    initial_Theta{hidden_layer_N + 1} = randInitializeWeights(hidden_layer_size, nOut);

    % Unroll parameters
    initial_nn_params = [];
    for i = 1:hidden_layer_N + 1
        initial_nn_params = [initial_nn_params; initial_Theta{i}(:)];
    end

    %% Gradient checking (uncomment this section for gradient checking)
    if gradientChecking
        checkGradients_DNN(input_layer_size, hidden_layer_size, nOut, hidden_layer_N, size(X, 1), lambda, zeta, numF, numJ_F)
    end

    %% DNN Model training
    % Define the cost function                                     
    costFunction = @(p) DNN_CostFunction(p, input_layer_size, hidden_layer_size, ...
                                         nOut, hidden_layer_N, X, y, lambda, zeta, numF, numJ_F);

    % Set up the routine options
    options = optimset('MaxIter', N_iter);

    % Minimize the cost function using the conjugate gradient routine (function
    % fmincg() from Coursera)
    [nn_params, Jcost_vector] = fmincg(costFunction, initial_nn_params, options);
    
    % Extract all Theta matrices for the L-layer NN from the unrolled vector nn_params
    Theta = cell(hidden_layer_N + 1, 1);
    Theta{1} = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                      hidden_layer_size, (input_layer_size + 1));
    for i = 2:hidden_layer_N
        Theta{i} = reshape(nn_params(1 + (i - 2)*(hidden_layer_size * (hidden_layer_size + 1)) + hidden_layer_size * (input_layer_size + 1):(i - 1)*(hidden_layer_size * (hidden_layer_size + 1)) + (hidden_layer_size * (input_layer_size + 1))), ...
                     hidden_layer_size, (hidden_layer_size + 1));
    end
    Theta{hidden_layer_N + 1} = reshape(nn_params(1 + (hidden_layer_N - 1)*hidden_layer_size * (hidden_layer_size + 1) + (hidden_layer_size * (input_layer_size + 1)):end), ...
                     nOut, (hidden_layer_size + 1));   
end