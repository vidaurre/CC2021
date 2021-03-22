function [acc_state,acc_time,Gamma] = tudacv_xstate(X,Y,T,options)
%
% Performs cross-validation of the TUDA model across states
% (the words decoder and state are used below indistinctly)
%
% INPUT
%
% X: Brain data, (time by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
%               For binary classification problems, Y is (time by 1) and
%               has values -1 or 1
%               For multiclass classification problems, Y is (time by classes) 
%               with indicators values taking 0 or 1. 
%           If the stimulus is the same for all trials, Y can have as many
%           rows as trials, e.g. (trials by q) 
% T: Length of series or trials
% options: structure with the training options - see documentation in
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%  - options.NCV, containing the number of cross-validation folds (default 10)
%  - options.c      an optional CV fold structure as returned by cvpartition
%
% Note that the estimation of Gamma is *not* cross-validated
%
% OUTPUT
%
% acc_state: (KxK) cross-validated explained variance if Y is continuous,
%           classification accuracy if Y is categorical (one value),
%           where acc(i,j) corresponds to the accuracy of state i when
%           state j is active. 
% acc_time: (TxK) cross-validated explained variance time point by time point 
%            for each state
% Gamma: State time courses 
%
% Author: Diego Vidaurre, OHBA, University of Oxford 

N = length(T); q = size(Y,2); ttrial = T(1); K = options.K;
if ~all(T==T(1)), error('All elements of T must be equal for cross validation'); end 

if size(Y,1) == length(T) % one value per trial
    responses = Y;
else
    responses = reshape(Y,[ttrial N q]);
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
end

options.Nfeatures = 0;
Ycopy = Y;
if size(Ycopy,1) == N 
    Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
end
[X,Y,T,options] = preproc4hmm(X,Y,T,options); % this demeans Y if necessary
ttrial = T(1); p = size(X,2); q_star = size(Y,2); 
classifier = options.classifier;
classification = ~isempty(classifier);
if classification, Ycopy = round(Ycopy); end
if q_star~=q && strcmp(options.distribution,'logistic')
    Ycopy = multinomToBinary(Ycopy);
    q = size(Ycopy,2);   
    responses = Ycopy(cumsum(T),:);
end
if strcmp(classifier,'LDA') || options.encodemodel
    error('LDA not implemented in tudacv_xstate'); 
end
%if strcmp(classifier,'LDA') || options.encodemodel
%    options.intercept = false; %this necessary to avoid double addition of intercept terms
%end

if isfield(options,'CVmethod')
    warning('No use for CVmethod here')
    options = rmfield(options,'CVmethod');
else
end
class_totals = (sum(Ycopy==1)./ttrial);
if q_star == (q+1)
    class_totals = class_totals(2:end); %remove intercept term
end 
if size(unique(class_totals))>1
    warning(['Note that Y is not balanced; ' ...
        'cross validation folds will not be balanced and predictions will be biased'])
end
if isfield(options,'c')
    NCV = options.c.NumTestSets;
    if isfield(options,'NCV'), options = rmfield(options,'NCV'); end
elseif isfield(options,'NCV')
    NCV = options.NCV; 
    options = rmfield(options,'NCV');
else
    %default to hold one-out CV unless NCV>10:
    NCV = max([0,class_totals]);
    if NCV > 10 || NCV < 1, NCV = 10; end
    
end
if isfield(options,'verbose') 
    verbose = options.verbose; options = rmfield(options,'verbose');
else, verbose = 1; 
end
if isfield(options,'accuracyType')
    accuracyType = options.accuracyType;
    options = rmfield(options,'accuracyType');
else
    accuracyType = 'COD';
end
options.verbose = 0; 

if ~isfield(options,'c')
    % this system is thought for cases where a trial can have more than 
    % 1 category, and potentially each column can have more than 2 values,
    % but there are not too many categories
    if classification 
        tmp = zeros(N,1);
        for j = 1:q
            rj = responses(:,j);
            uj = unique(rj);
            for jj = 1:length(uj)
                tmp(rj == uj(jj)) = tmp(rj == uj(jj)) + (q+1)^(j-1) * jj;
            end
        end
        uy = unique(tmp);
        group = zeros(N,1);
        for j = 1:length(uy)
            group(tmp == uy(j)) = j;
        end
        c2 = cvpartition(group,'KFold',NCV);
    else % Response is treated as continuous - no CV stratification
        c2 = cvpartition(N,'KFold',NCV);
    end
else
   c2 = options.c; options = rmfield(options,'c');
end
c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = find(c2.training(icv));
    c.test{icv} = find(c2.test(icv));
end; clear c2

[~,Gamma] = tudatrain(X,Y,T,options);
options.cyc = 1; 

X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q_star]);
Gamma = reshape(Gamma,[ttrial N K]);

% Get Gamma and the Betas for each fold
Betas = zeros(p,q_star,K,NCV); 
if strcmp(options.classifier,'regression'), options.classifier = ''; end
for icv = 1:NCV
    Ntr = length(c.training{icv});  
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial q_star] ) ;
    Ttr = T(c.training{icv});
    options.Gamma = reshape(Gamma(:,c.training{icv},:),[Ntr*ttrial K] ) ;
    tuda = tudatrain(Xtrain,Ytrain,Ttr,options,1); % Gamma never gets updated
    Betas(:,:,:,icv) = tudabeta(tuda);
    if verbose
        fprintf(['\nCV iteration: ' num2str(icv),' of ',int2str(NCV),'\n'])
    end
end

% Perform the prediction 
Ypred = zeros(ttrial,N,q_star,K);
for icv = 1:NCV
    Nte = length(c.test{icv});
    Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
    for k = 1:K
        Ypred(:,c.test{icv},:,k) = reshape( (Xtest * Betas(:,:,k,icv)) ,...
            [ttrial Nte q_star]);
    end
end

% Get Gamma and the Betas for each fold
Betas2 = zeros(p,q_star,K,NCV); 
for icv = 1:NCV
    Ntr = length(c.training{icv});  
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial q_star] ) ;
    Gamma2 = reshape(Gamma(:,c.training{icv},:),[Ntr*ttrial K] ) ;
    for k = 1:K
        Betas2(:,:,k,icv) = ( (Xtrain .* repmat(Gamma2(:,k),1,size(Xtrain,2)))' * Xtrain ) \ ...
            ( (Xtrain .* repmat(Gamma2(:,k),1,size(Xtrain,2)))' * Ytrain ) ;
    end
    if verbose
        fprintf(['\nCV iteration: ' num2str(icv),' of ',int2str(NCV),'\n'])
    end
end
Ypred2 = zeros(ttrial,N,q_star,K); 
for icv = 1:NCV
    Ntr = length(c.training{icv});
    Xtrain = reshape(X(:,c.training{icv},:),[ttrial*Ntr p]);
    for k = 1:K
        Ypred2(:,c.training{icv},:,k) = Ypred2(:,c.training{icv},:,k) + reshape( (Xtrain * Betas2(:,:,k,icv)) ,...
            [ttrial Ntr q_star]) / (NCV-1);
    end
end
CV2 = zeros(250,K);
for k = 1:K
    CV2(:,k) = 1 -  sum(((Ypred(:,:,1,k) - Y)).^2 ,2) ./ sum((( Y)).^2 ,2);
end

if strcmp(options.distribution,'logistic')
    for k = 1:K
        if q_star==q % denotes binary logistic regression
            Ypred(:,:,:,k) = log_sigmoid(Ypred(:,:,:,k));
        else %multivariate logistic regression
            Ypred(:,:,:,k) = multinomLogRegPred(Ypred(:,:,:,k));
        end
    end
end

acc_state = zeros(K,K,q);
acc_time = zeros(ttrial,K,q);

if classification
    % Time point by time point
    Y = reshape(Ycopy,[ttrial*N q]);
    Y = continuous_prediction_2class(Ycopy,Y); % get rid of noise we might have injected
    for k = 1:K
        Ypred_temp = reshape(Ypred(:,:,:,k),[ttrial*N q]);
        Ypred_star_temp = reshape(continuous_prediction_2class(Ycopy,Ypred_temp),[ttrial N q]);
        Ypred_temp = zeros(N,q);
        for j = 1:N % getting the most likely class for all time points in trial
            if q == 1 % binary classification, -1 vs 1
                Ypred_temp(j) = sign(mean(Ypred_star_temp(:,j,1)));
            else
                [~,cl] = max(mean(permute(Ypred_star_temp(:,j,:),[1 3 2])));
                Ypred_temp(j,cl) = 1;
            end
        end
        % acc is cross-validated classification accuracy
        Ypred_star_temp = reshape(Ypred_star_temp,[ttrial*N q]);
        if q == 1
            tmp = abs(Y - Ypred_star_temp) < 1e-4;
        else
            tmp = sum(abs(Y - Ypred_star_temp),2) < 1e-4;
        end
        acc_time(:,k) = squeeze(mean(reshape(tmp, [ttrial N]),2));
        for k2 = 1:K
            acc_state(k,k2) = (acc_time(:,k)' * Gamma(:,k2)) / sum(Gamma(:,k2));
        end
    end
else
    for k = 1:K
        Ypred_star_temp = reshape(Ypred(:,:,:,k), [ttrial N q]);
        for t = 1:ttrial
            y = permute(Y(t,:,:),[2 3 1]);
            if strcmp(accuracyType,'COD')
                acc_time(t,k,:) = 1 - sum((y - permute(Ypred_star_temp(t,:,:),[2 3 1])).^2) ./ sum(y.^2);
            elseif strcmp(accuracyType,'Pearson')
                acc_time(t,k,:) = diag(corr(y,permute(Ypred_star_temp(t,:,:),[2 3 1])));
            end
            for k2 = 1:K
                for j = 1:q
                    acc_state(k,k2,j) = (acc_time(:,k,j)' * Gamma(:,k2))  / sum(Gamma(:,k2));
                end
            end
        end
    end     
end

Gamma = reshape(Gamma,[ttrial*N K]);


end


function Y_out = multinomToBinary(Y_in)
Y_out=zeros(length(Y_in),length(unique(Y_in)));
for i=1:length(Y_in)
    Y_out(i,Y_in(i))=1;
end
end

