function [accuracy,K] = standard_LDA(X,Y,T,options)
% options.K is the number of PCA components, assuming X are PCA components

if ~all(T==T(1)), error('All elements of T must be equal here'); end
ttrial = T(1); p = size(X,2); q = size(Y,2); N = length(T); 

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
    if NCV > 10, NCV = 10; end
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

% train HMM on training data
parfor icv = 1:NCV
    Ntr = length(c.training{icv}); 
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
                Nte1 = sum(ind1t); Nte2 = sum(ind2t);
                Nte = Nte1 + Nte2;  
                Mu1 = sum(x(ind1,:)) / Nte1;
                Mu2 = sum(x(ind2,:)) / Nte2;
                xstar = [x(ind1,:); x(ind2,:)];
                Mu0 = sum(xstar) / Nte; 
                C0 = (xstar' * xstar) / (Ntr -1);
                C0 = C0 + 0.1 * C0(end,end) * eye(size(C0,1));
                
                for ii = 1:length(K)
                    k = K(ii);
                    mu0 = Mu0(:,1:k); 
                    C = C0(1:k,1:k);
                    mu1 = Mu1(:,1:k); 
                    mu2 = Mu2(:,1:k); 
                    w = C \ (mu1 - mu2)'; 
                    w0 = mu0 * w;
                    yhat1 = sum(xtest(ind1t,1:k) .* w',2) - w0'; % decision boundary; correct is >1
                    yhat2 = sum(xtest(ind2t,1:k) .* w',2) - w0'; % decision boundary; correct is <1
                    acc(it,j1,j2,ii) = (sum(yhat1>0) + sum(yhat2<0)) / Nte;
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
