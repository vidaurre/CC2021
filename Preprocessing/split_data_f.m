function [Y1,Y2] = split_data_f(X,T,s)

two_dim = length(size(X))==2;

if two_dim
    X = reshape(X,[T(1), length(T), size(X,2)]);
end

[~,N,p] = size(X);
Y1 = zeros(size(X));
Y2 = zeros(size(X));

for j = 1:p
    for n = 1:N
        x1 = smooth(X(:,n,j),s,'lowess');
        b = (x1' * X(:,n,j)) /  (x1' * x1);
        x2 = X(:,n,j) - b * x1;
        Y1(:,n,j) = x1;
        Y2(:,n,j) = x2;
    end
end

if two_dim
    Y1 = reshape(Y1,[T(1)*length(T), size(X,3)]);
    Y2 = reshape(Y2,[T(1)*length(T), size(X,3)]);
end

end