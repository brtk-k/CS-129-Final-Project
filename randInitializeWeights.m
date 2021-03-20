%% Function adapted from the "randInitializeWeights" function in Coursera "Neural Network Learning.ipynb" notebook
function Theta = randInitializeWeights(L_in, L_out)           
    % Randomly initialize the weights to small values
    epsilon_init = 0.12;
    Theta = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end