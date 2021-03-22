function accuracy = standard_regression_cross(X,Y,T,options)
% Get the Temporal Generalization Matrix with the decoding accuracies
% (generalising across time points)

if nargin < 4, options = struct(); end
if ~isfield(options,'method'), method = 'cvridge';
else, method = options.method;
end
if ~isfield(options,'alpha'), alpha = [0.01 0.1 1 10 100 1000 10000 100000];
else, alpha = options.alpha;
end
if ~isfield(options,'permuteTrials'), permuteTrials = 0;
else, permuteTrials = options.permuteTrials;
end
if ~isfield(options,'timeNorm'), timeNorm = 0;
else, timeNorm = options.timeNorm;
end

%NA = length(alpha);

if ~all(T==T(1)), error('All elements of T must be equal here'); end
if length(size(X))==3
    ttrial = T(1); p = size(X,3); q = size(Y,3); N = length(T);
    if permuteTrials
        r = randperm(N); X = X(:,r,:); Y = Y(:,r,:);
    end
    if timeNorm % channel normalisation for each time point, to focus in channel differences
        for t = 1:ttrial
            X(t,:,:) = zscore(permute(X(t,:,:),[2 3 1]));
        end
    end
    X = reshape(X,[ttrial*N p]);
    Y = double(reshape(Y,[ttrial*N q]));
else
    ttrial = T(1); p = size(X,2); q = size(Y,2); N = length(T);
    if permuteTrials
        X = reshape(X,[ttrial N p]);
        Y = double(reshape(Y,[ttrial N q]));
        r = randperm(N);  X = X(:,r,:); Y = Y(:,r,:);
        X = reshape(X,[ttrial*N p]);
        Y = double(reshape(Y,[ttrial*N q]));
    end
    if timeNorm % channel normalisation for each time point, to focus in channel differences
        X = reshape(X,[ttrial N p]);
        for t = 1:ttrial
            X(t,:,:) = zscore(permute(X(t,:,:),[2 3 1]));
        end
        X = reshape(X,[ttrial*N p]);
    end
end

uY = unique(Y(:));
X = zscore(X);

if q==2 || length(uY)==2
    classification = true;
    if q==2
        if any( Y(:)~=1  & Y(:)~= 0  )
            error('Wrong format for Y')
        end
        if any( sum(Y,2)~=1 )
            error('Wrong format for Y')
        end
        Y = Y(:,1) - Y(:,2);
    end
    responses = reshape(Y,[ttrial N]);
    responses = responses(1,:)';
elseif q > 2
    error('Not yet working for more than 2 classes')
else
    classification = false;
end

if classification
    c1 = sum(Y==-1); c2 = sum(Y==1); c12 = length(Y);
    Y(Y==-1) = - c12 / c1;  Y(Y==+1) = c12 / c2;
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
    if classification
        c2 = cvpartition(responses,'KFold',NCV);
    else
        c2 = cvpartition(N,'KFold',NCV);
    end
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
Y = double(reshape(Y,[ttrial N]));

accuracy = NaN(ttrial,ttrial,NCV);

for icv = 1:NCV
    
    Ntr = length(c.training{icv}); Nte = length(c.test{icv});
    Xtrain_orig = X(:,c.training{icv},:);
    Ytrain_orig = Y(:,c.training{icv});
    Xtest_orig = X(:,c.test{icv},:);
    Ytest_orig = Y(:,c.test{icv});
    if classification
        r = responses(c.training{icv});
    end
    for t = 1:ttrial
        Xtrain = [ones(Ntr,1) reshape(Xtrain_orig(t,:,:),[Ntr p] )] ;
        Ytrain = reshape(Ytrain_orig(t,:),[Ntr 1] ) ;
        Xtest = [ones(Nte*ttrial,1) reshape(Xtest_orig,[Nte*ttrial p])];
        Ytest = Ytest_orig;
        if strcmp(method,'cvridge')
            C1 = Xtrain' * Xtrain;
            C2 = Xtrain' * Ytrain;
            if classification
                best_alpha = CV_alpha(Xtrain,Ytrain,alpha,r,NCV);
            else
                best_alpha = CV_alpha(Xtrain,Ytrain,alpha,[],NCV);
            end
            R = best_alpha * eye(p+1); R(1,1) = 0;
            b = (C1 + R) \ C2;
        else
            %PriorMdl = bayeslm(p,'ModelType','semiconjugate');
            %PosteriorMdl = estimate(PriorMdl,Xtrain,Ytrain);
            %b = mean(PosteriorMdl.BetaDraws,2);
            b = bayes_linear_fit(Xtrain,Ytrain);
        end
        Yhat = reshape(Xtest * b,[ttrial Nte]);
        if classification
            accuracy(t,:,icv) = mean(sign(Yhat)==sign(Ytest),2);
        else
            for t2 = 1:ttrial
                accuracy(t,t2,icv) = corr(Yhat(t2,:)',Ytest(t2,:)');
            end
            %e = (Yhat-Ytest).^2;
            %e0 = (Ytest-repmat( mean(Ytest,2),[1 Nte] ) ).^2;
            %accuracy(t,:,icv) = 1 - sum(e,2) ./  sum(e0,2);
        end
        %if t==60; keyboard; end
    end
    disp(['ICV: ' num2str(icv)])
end

accuracy = mean(accuracy,3);

end



