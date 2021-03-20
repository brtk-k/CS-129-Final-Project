%% Function modified and generalized based on the "nnCostFunction" function on Coursera coding assignments
function [J, grad] = DNN_CostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, hidden_layer_N, ...
                                   X, y, lambda, zeta, numF, numJ_F)
    %% Initialization
    % Initialize the Theta cell array
    Theta = cell(hidden_layer_N + 1, 1);
    
    % Extract the Theta matrices from the nn_params unrolled vector
    Theta{1} = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                      hidden_layer_size, (input_layer_size + 1));
    for i = 2:hidden_layer_N
        Theta{i} = reshape(nn_params(1 + (i - 2)*(hidden_layer_size * (hidden_layer_size + 1)) + hidden_layer_size * (input_layer_size + 1):(i - 1)*(hidden_layer_size * (hidden_layer_size + 1)) + (hidden_layer_size * (input_layer_size + 1))), ...
                     hidden_layer_size, (hidden_layer_size + 1));
    end
    Theta{hidden_layer_N + 1} = reshape(nn_params(1 + (hidden_layer_N - 1)*hidden_layer_size * (hidden_layer_size + 1) + (hidden_layer_size * (input_layer_size + 1)):end), ...
                     num_labels, (hidden_layer_size + 1));        
    
    % Initialize the cell array of Theta gradients
    Theta_grad = cell(hidden_layer_N + 1, 1);
    for i = 1:hidden_layer_N + 1
        Theta_grad{i} = zeros(size(Theta{i}));
    end
    
    % Extract number of examples
    m = size(X, 1);                               

    %% Feedforward propagation of the parameters
    a = cell(hidden_layer_N + 2, 1); % Initalize cell array of a{i} matrices
    z = cell(hidden_layer_N + 2, 1); % Initalize cell array of z{i} matrices

    a{1} = [ones(m, 1), X];
    for i = 2:hidden_layer_N + 1
        z{i} = a{i - 1}*Theta{i - 1}';
        a{i} = [ones(m, 1), sigmoid(z{i})];
    end
    z{hidden_layer_N + 2} = a{hidden_layer_N + 1}*Theta{hidden_layer_N + 1}';
    a{hidden_layer_N + 2} = zeta*(sigmoid(z{hidden_layer_N + 2}) - 1/2);

    % Compute the unregularized cost function value
    J_unreg = 0;
    for k = 1:m
        J_unreg = J_unreg + 1/(2*m)*norm(numF(a{end}(k,:)') - X(k,1:3)')^2;
    end

    % Compute the regularized terms
    J_regterms = 0;
    for i = 1:hidden_layer_N + 1
        J_regterms = J_regterms + sum(sum(Theta{i}(:,2:end).^2));
    end
    J_regterms = lambda/(2*m)*J_regterms;

    % Compute the regularized cost function value
    J = J_unreg + J_regterms;
    
    %% Backpropagation (gradient computation)
    % Compute the J-gradients w.r.t. the Theta matrices
    delta_k = cell(hidden_layer_N + 2, 1);      % Initialize the example-wise delta cell array
    Theta_grad_k = cell(hidden_layer_N + 1, 1); % Initialize the example-wise Theta_grad cell array
    for k = 1:m
        % Construct the delta array for k-th example
        delta_k{hidden_layer_N + 2} = ((numF(a{end}(k,:)') - X(k,1:3)')'*numJ_F(a{end}(k,:)')*diag(zeta*sigmoidGradient(z{end}(k,:))))';
        for i = hidden_layer_N + 1:-1:2
            delta_k{i} = (delta_k{i+1}'*Theta{i}(:,2:end)*diag(sigmoidGradient(z{i}(k,:))))';
        end
        
        % Accumulate the Theta gradients Theta_grad{i}: {Theta1_grad,...,Theta(L-1)_grad} over all examples k: {1,...,m}
        for i = 1:hidden_layer_N + 1
            Theta_grad_k{i} = delta_k{i + 1}*a{i}(k,:);
            Theta_grad{i} = Theta_grad{i} + 1/m*Theta_grad_k{i};
        end
    end
    
    % Regularize the Theta gradients
    for i = 1:hidden_layer_N + 1
        Theta{i}(:,1) = 0;
        Theta_grad{i} = Theta_grad{i} + lambda/m*Theta{i};
    end

    % Unroll the gradients into a vector output
    grad = [];
    for i = 1:hidden_layer_N + 1
        grad = [grad; Theta_grad{i}(:)];
    end
end