% Function computing the sigmoid gradient evaluated at z
function sigmaGradient = sigmoidGradient(z)
    sigmaGradient = sigmoid(z).*(1 - sigmoid(z));
end