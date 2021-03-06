function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% Computing squared distance between all examples and all K
m = size(X,1);

for i = 1:m
  dis = ones(K,1)*X(i,:) - centroids;
  dis = sum(dis.^2, 2);
  [x, idx(i)] = min(dis);
end
% =============================================================

end
