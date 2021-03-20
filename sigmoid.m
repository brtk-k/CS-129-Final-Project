% Function computing the sigmoid of the input z
function sigma = sigmoid(z)
    sigma = 1./(1 + exp(-z));
end
