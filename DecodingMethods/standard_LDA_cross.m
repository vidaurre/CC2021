function accuracy = standard_LDA_cross(X,Y,T,options)
% Get the Temporal Generalization Matrix with the decoding accuracies
% (generalising across time points)

if nargin < 4, options = struct(); end
if ~isfield(options,'alpha'), alpha = [0.001 0.01 0.1 linspace(0.2,1.0,9)];
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
    error('This is for classification')
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
        Xtrain = reshape(Xtrain_orig(t,:,:),[Ntr p] ) ;
        Ytrain = reshape(Ytrain_orig(t,:),[Ntr 1] ) ;
        Xtest = reshape(Xtest_orig,[Nte*ttrial p]);
        Ytest = Ytest_orig;
        
        m1 = mean(Xtrain(Ytrain==-1,:));
        m2 = mean(Xtrain(Ytrain==1,:));
        S = Xtrain' * Xtrain / N;
        
        best_alpha = CV_alpha_LDA(Xtrain,Ytrain,alpha,r,NCV);
        
        w = ((1 - best_alpha) * S  + best_alpha * eye(p)) \ (m2 - m1)';
        Yhat = -ones(ttrial*Nte,1);
        cl = Xtest * w;
        Yhat(cl>0) = 1;
        
        Yhat = reshape(Yhat,[ttrial Nte]);
        accuracy(t,:,icv) = mean(sign(Yhat)==sign(Ytest),2);
        
    end
    disp(['ICV: ' num2str(icv)])
end

accuracy = mean(accuracy,3);

end





