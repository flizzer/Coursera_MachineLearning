function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%compute the hypothesis vector
h = X*theta;

%compute the squared errors
squaredErrors = (h - y).^2;

%since this is a vector, sum the errors
sumOfSquaredErrors = sum(squaredErrors);

%finish the equation by multiplying by 1/2m
J = (1/(2*m))*sumOfSquaredErrors;


% =========================================================================

end
