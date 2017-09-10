function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));


z = sigmoid(X*theta);
c = theta(2:end);
a = lambda*(c'*c)/(2*m);

J = -1*sum(y.*log(z) + (1.-y).*log(1.-z))/m + a;

b = (lambda/m).*theta;
b(1) = 0;

grad = (X'*(z-y))./m + b;


% =============================================================

grad = grad(:);

end
