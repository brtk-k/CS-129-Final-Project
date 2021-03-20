%% Function modified and generalized from the "checkNNGradients.m" file on Coursera coding assignments
function checkGradients_DNN(input_layer_size, hidden_layer_size, num_labels, hidden_layer_N, m, lambda, zeta numF2, numJ_F2)
    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0;
    end

    % Randomly initialize weights
    Theta = cell(hidden_layer_N + 1, 1);
    Theta{1} = randInitializeWeights(input_layer_size, hidden_layer_size);
    for i = 2:hidden_layer_N
        Theta{i} = randInitializeWeights(hidden_layer_size, hidden_layer_size);
    end
    Theta{hidden_layer_N + 1} = randInitializeWeights(hidden_layer_size, num_labels);

    % Randomly initialize data
    X = 0.5*rand(m, input_layer_size);
    y = zeta*(rand(m, num_labels) - 1/2);

    % Unroll parameters
    nn_params = [];
    for i = 1:hidden_layer_N + 1
        nn_params = [nn_params; Theta{i}(:)];
    end

    % Short hand for cost function
    costFunc = @(p) nnCostFunction_Deep(p, input_layer_size, hidden_layer_size, ...
                                   num_labels, hidden_layer_N, X, y, lambda, zeta, numF2, numJ_F2);

    [cost, grad] = costFunc(nn_params);
    numgrad = computeNumericalGradient(costFunc, nn_params);

    % Visually examine the two gradient computations.  The two columns
    % you get should be very similar. 
    disp([numgrad grad]);
    fprintf(['The above two columns you get should be very similar.\n' ...
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

    % Evaluate the norm of the difference between two solutions.  
    % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    % in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = norm(numgrad-grad)/norm(numgrad+grad);

    fprintf(['If your backpropagation implementation is correct, then \n' ...
             'the relative difference will be small (less than 1e-9). \n' ...
             '\nRelative Difference: %g\n'], diff);
end
