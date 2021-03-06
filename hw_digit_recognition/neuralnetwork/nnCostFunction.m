function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Computing cost by feedforwarding
A1 = [ones(m, 1) X]';

Z2 = Theta1*A1;
A2 = sigmoid(Z2);
A2 = [ones(1,m); A2];

A3 = sigmoid(Theta2*A2);

Y = zeros(num_labels, m);
for c = 1:m
    Y(y(c, 1), c) = 1;
end

% Summation of regularized theta values (excluding first coulmns of each theta matrix)
reg = (lambda/(2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Computing regularized cost
J = -1*sum(sum(Y .* log(A3) + (1 .- Y).*log(1 .- A3)))/m + reg;

% backpropagation
D3 = A3 - Y;
D2 = (Theta2' * D3) .* sigmoidGradient([ones(1, m); Z2]);

Theta1_grad = (D2(2:end, :) * A1')/m + (lambda/m)*[zeros(hidden_layer_size,1) Theta1(:, 2:end)];
Theta2_grad = (D3 * A2')/m + (lambda/m)*[zeros(num_labels,1) Theta2(:, 2:end)];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
