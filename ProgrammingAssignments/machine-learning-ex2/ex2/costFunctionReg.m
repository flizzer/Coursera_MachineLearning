function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

   
h = sigmoid(X * theta);
   
#vectorized cost function
unregularizedCost = ((-y' * log(h)) - ((1-y)' * log(1-h)))/m;

#remember, we don't want to penalize theta zero and Octave starts indexing array elements at 1 so we need to set that element to 0
theta(1) = 0;

#must use a transpose on order to multiply the dimensions
sumOfSquares = theta' * theta;
   
costRegularizationTerm = (lambda/(2*m))*sumOfSquares;
   
J = unregularizedCost + costRegularizationTerm;
   
#vectorized gradient descent partial derivatives
errors = (h-y);
errorsMatrix = X'*errors;
unregularizedGradient = (1/m)*errorsMatrix;
   
#compute the regularization term for gradient descent
gradientRegularization = (lambda/m) * theta;
   
#now sum up the gradient descent calculation and the regularization term
grad = unregularizedGradient + gradientRegularization;

% =============================================================

end
