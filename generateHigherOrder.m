% Function appending higher-order features to X, up to order beta 
% (where beta > 1). The function operates under any number of Euclidean
% dimensions for the base features.
function X = generateHigherOrder(X, beta)
    m = size(X, 1); % Number of examples
    space_dim = size(X, 2); % Dimension of Euclidean space
    expo = cell(space_dim, 1); % Initialize a cell array of 2D exponent matrices
    
    % Compute the higher-order features up to order = beta
    for order = 2:beta
        [expo{1}, expo{2}, expo{3}] = ndgrid(0:order);
        exp_poly = [expo{1}(expo{1} + expo{2} + expo{3} == order), expo{2}(expo{1} + expo{2} + expo{3} == order), expo{3}(expo{1} + expo{2} + expo{3} == order)];
        for i = 1:size(exp_poly, 1)
            x_new = ones(m, 1);
            for j = 1:size(exp_poly, 2)
                x_new = x_new.*X(:, j).^exp_poly(i, j);
            end
            X = [X, x_new];
        end
    end
end