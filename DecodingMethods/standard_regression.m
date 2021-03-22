function [accuracy,K] = standard_regression(X,Y,T,options)
% options.K is the number of PCA components, assuming X are PCA components
% Y is time by classes, with 0 or 1

if ~all(T==T(1)), error('All elements of T must be equal here'); end
ttrial = T(1); p = size(X,2); q = size(Y,2); N = length(T); 
X = zscore(X); 

responses = reshape(Y,[ttrial N q]);
responses = permute(responses(1,:,:),[2 3 1]); % N x q

if nargin < 4, options = struct(); end
if ~isfield(options,'K'), K = p; 
else, K = options.K;
end 
if ~isfield(options,'time'), time = 1:ttrial; 
else, time = options.time;
end 

% Make the CV folds
if isfield(options,'c')
    NCV = options.c.NumTestSets;
    if isfield(options,'NCV'), options = rmfield(options,'NCV'); end
elseif isfield(options,'NCV')
    NCV = options.NCV;
    options = rmfield(options,'NCV');
else %default to 10
    NCV = 10;
end
if ~isfield(options,'c')
    group = zeros(N,1);
    for j = 1:q
        rj = responses(:,j);
        group(rj==1) = j;
    end
    c2 = cvpartition(group,'KFold',NCV);
else
    c2 = options.c; 
end
c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = find(c2.training(icv));
    c.test{icv} = find(c2.test(icv));
end; clear c2

X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);

accuracy = zeros(length(time),q*(q-1)/2,length(K),NCV);

for icv = 1:NCV
    Xtrain = X(:,c.training{icv},:);
    Ytrain = squeeze(Y(1,c.training{icv},:));
    Xtest = X(:,c.test{icv},:);
    Ytest = squeeze(Y(1,c.test{icv},:));    
    
    acc = zeros(length(time),q,q,length(K));
    for it = 1:length(time)
        t = time(it);
        x = permute(Xtrain(t,:,:),[2 3 1]);
        xtest = permute(Xtest(t,:,:),[2 3 1]);
        for j1 = 1:q-1
            ind1 = Ytrain(:,j1) == 1;
            ind1t = Ytest(:,j1) == 1;
            for j2 = j1+1:q
                ind2 = Ytrain(:,j2) == 1;
                ind2t = Ytest(:,j2) == 1;
                N1 = sum(ind1); N2 = sum(ind2); N12 = N1+N2; 
                Nte1 = sum(ind1t); Nte2 = sum(ind2t); Nte = Nte1+Nte2;
                xstar = [ones(N12,1) [x(ind1,:); x(ind2,:)]];
                ystar = [-ones(N1,1); ones(N2,1)];
                C0 = (xstar' * xstar);
                R = 0.01 * eye(size(C0,1)); R(1,1) = 0;
                %R = 0.1 * C0(end,end) * eye(size(C0,1)); R(1,1) = 0;
                C0 = C0 + R;
               
                for ii = 1:length(K)
                    k = K(ii);
                    C = C0(1:(k+1),1:(k+1));
                    w = C \ (xstar' * ystar);
                    yhat1 = [ones(Nte1,1) xtest(ind1t,1:k)] * w; % decision boundary; correct is <1
                    yhat2 = [ones(Nte2,1) xtest(ind2t,1:k)] * w; % decision boundary; correct is >1
                    acc(it,j1,j2,ii) = (sum(yhat1<0) + sum(yhat2>0)) / Nte;
                    acc(it,j2,j1,ii) = acc(it,j1,j2,ii);
                end
            end
        end
        
    end
    
    disp(['CV, iteration ' num2str(icv)])
    
    acc = permute(acc,[1 4 2 3]);
    acc = acc(:,:,triu(true(q),1));
    acc = permute(acc,[1 3 2]);
    
    accuracy(:,:,:,icv) = acc; 
    
end

accuracy = squeeze(mean(accuracy,4));
    
end
