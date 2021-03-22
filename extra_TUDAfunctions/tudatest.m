function [pvals,perms,Gammahat,pvals_std,perms_std] = tudatest(X,Y,T,options,perms,perms_std)
% Run permutation testing for TUDA as well as standard decoding.
%
% For TUDA there are three different tests: 
% 1. Is the unconstrained model more accurate than the constrained model?
% 2. Is the constrained model significantly predictive? 
% 3. Is the unconstrained model significantly predictive? 
%
% The unconstrained model is TUDA as presented in Vidaurre et al 2019, Cer Cortex
% (that is, allowing for between-trial differences in the timing of things).
% The constrained model is still data-driven in the sense that the
% inference decides when we jump from one state to the next, but the
% transitions occur at the same time point in all trials. 
%
% Standard decoding is done time point by time point. 
% 
% OUTPUTS
% pvals: (time x 3) p-values for each time point in the trial and each of
%   the tests mentioned above
% perms: basic statistics from permutation testing 
%     (No. perms x time points in trial x 2)
%       perms(:,:,1) is for the constrained model, 
%       perms(:,:,2) is for the unconstrained model.
% Gammahat: State time courses for TUDA 
% pvals_std: p-values for standard decoding, one per time point, obtained
%       from permutation testing
% perms_std: basic statistics from permutation testing applied to standard
%       decoding (No. perms by time points in trial)
%   
% Diego Vidaurre, University of Oxford / Aarhus University (2020) 

N = length(T); ttrial = T(1);

if isfield(options,'Nperms')
    Nperms = options.Nperms; options = rmfield(options,'Nperms');
else
    Nperms = 1000;
end

% Check options and put data in the right format
[X,Y,T,options] = preproc4hmm(X,Y,T,options);
classifier = options.classifier;
sequential = options.sequential;
parallel_trials = options.parallel_trials; 
p = size(X,2); q = size(Y,2);
if q > 1, error('Y can only have one column here'); end
adjustBeta = 0;
do_stddecoding = (nargout > 3);

% init HMM, only if trials are temporally related
if ~isfield(options,'Gamma')
    if parallel_trials
        if sequential
            GammaInit = cluster_decoding(X,Y,T,options.K,'fixedsequential');
        else
            GammaInit = cluster_decoding(X,Y,T,options.K,'regression','',...
                options.Pstructure,options.Pistructure);
        end
        Gamma = permute(repmat(GammaInit,[1 1 N]),[1 3 2]);
        Gamma = reshape(Gamma,[length(T)*size(GammaInit,1) options.K]);
    else
        error('parallel_trials are required')
    end
else
   error('Init Gamma cannot be supplied')
end
options = rmfield(options,'sequential');
if ~isfield(options,'Gamma_constraint')
    options.Gamma_constraint = [];
end
if isfield(options,'accuracyType')
    accuracyType = options.accuracyType;
    options = rmfield(options,'accuracyType');
else
    accuracyType = 'SquaredErr';
end

% Put X and Y together
Ttmp = T;
T = T + 1;
Z = zeros(sum(T),q+p,'single');
for j = 1:N
    t1 = (1:T(j)) + sum(T(1:j-1));
    t2 = (1:Ttmp(j)) + sum(Ttmp(1:j-1));
    if strcmp(classifier,'LDA') || (isfield(options,'encodemodel') && options.encodemodel)
        error('Not yet implemented')
        %Z(t1(2:end),1:p) = X(t2,:);
        %Z(t1(1:end-1),(p+1):end) = Y(t2,:);
    else
        Z(t1(1:end-1),1:p) = X(t2,:);
        Z(t1(2:end),(p+1):end) = Y(t2,:);        
    end
end 

% Run TUDA inference
options.S = -ones(p+q);
if strcmp(classifier,'LDA') || (isfield(options,'encodemodel') && options.encodemodel)
    options.S(p+1:end,1:p) = 1;
else
    options.S(1:p,p+1:end) = 1;
end

%switch off parallel as not implemented for some models
if strcmp(classifier,'logistic') || strcmp(classifier,'LDA')
    options.useParallel = 0;
end
options.decodeGamma = 0;

% 0. In case parallel_trials is false and no Gamma was provided
if isempty(GammaInit) && ~parallel_trials
    error('Only works for parallel trials')
end

% init tuda and options
options = checkoptions(options,Z,T,0);
tuda = struct('train',struct());
tuda.K = options.K;
tuda.train = options;
tuda.train.ndim = size(Z,2); 
tuda.train.Gamma_constraint = options.Gamma_constraint;
tuda = hmmhsinit(tuda,Gamma,T);
%[~, residuals] = obsinit (struct('X',Z),T,tuda,Gamma);

% intermediate variables
orders = formorders(options.order);
X = formautoregr(Z,T,orders,1,1,0);
XXGXX = cell(options.K,1);
for k = 1:options.K
    XXGXX{k} = (X' .* repmat(Gamma(:,k)',size(X,2),1)) * X;
end
%residuals = reshape(residuals,[ttrial N size(residuals,2)]);

tuda0 = tuda; 
Gamma_constr = Gamma;
if nargin<5
    perms = zeros(Nperms,ttrial,2); Nperms0 = 0;
    if do_stddecoding
        perms_std = zeros(Nperms,ttrial);
    end
else
    perms0 = perms; Nperms0 = size(perms0,1); 
    perms = zeros(Nperms+Nperms0,ttrial,2);
    perms(1:Nperms0,:,:) = perms0;
    if do_stddecoding
        perms0_std = perms_std; 
        perms_std = zeros(Nperms+Nperms0,ttrial);
        perms_std(1:Nperms0,:) = perms0_std;
    end
end
optopt = optimoptions(@lsqlin);
optopt.Display = 'off';

for j = 1:Nperms
    Z = reshape(Z,[(ttrial+1) N size(Z,2)]);
    % if Nperms0 > 0, the unpermuted was done in the first pass
    if j > 1 || (Nperms0 > 0) 
        r = randperm(N);
        %residuals(:,:,(p+1):end) = residuals(:,r,(p+1):end);
        Z(:,:,(p+1):end) = Z(:,r,(p+1):end);
        
    end
    %residuals = reshape(residuals,[ttrial*N size(residuals,3)]);
    Z = reshape(Z,[(ttrial+1)*N size(Z,3)]);
    % train
    [tuda,Y] = obsinit (struct('X',Z),T,tuda0,Gamma_constr);
    tuda = obsupdate(T,Gamma_constr,tuda,Y,X,XXGXX);
    beta = squeeze(tudabeta(tuda));
    if adjustBeta
        beta = adjustbetas(X(:,1:end-1),Y(:,end),Gamma_constr,beta,optopt);
        for k = 1:size(beta,2)
            tuda.state(k).W.Mu_W(1:p,p+1:end) = beta(:,k);
            tuda.state(k).W.S_W = zeros(p+1,p+1,p+1);
        end
    end
    Gamma = hsinference(Z,T,tuda,Y,[],X);
    Y = reshape(Y(:,end),[ttrial N]);
    if j==1
        Gammahat = Gamma;
    end
    % predict
    Yhat = zeros(ttrial*N,1); Yhat_constr = zeros(ttrial*N,1); 
    for k = 1:options.K
        Yhat = Yhat + (X(:,1:p) * beta(:,k)) .* Gamma(:,k);
        Yhat_constr = Yhat_constr + (X(:,1:p) * beta(:,k)) .* Gamma_constr(:,k);        
    end
    Yhat = reshape(Yhat,[ttrial N]);
    Yhat_constr = reshape(Yhat_constr,[ttrial N]);
    if strcmp(accuracyType,'Pearson')
        perms(j+Nperms0,:,1) = diag(corr(Yhat_constr',Y'));
        perms(j+Nperms0,:,2) = diag(corr(Yhat',Y'));
    elseif strcmp(accuracyType,'SquaredErr')
        perms(j+Nperms0,:,1) = sum((Yhat_constr-Y).^2,2);
        perms(j+Nperms0,:,2) = sum((Yhat-Y).^2,2);
    elseif strcmp(accuracyType,'AbsErr') 
        perms(j+Nperms0,:,1) = sum(abs(Yhat_constr-Y),2);
        perms(j+Nperms0,:,2) = sum(abs(Yhat-Y),2);    
    elseif strcmp(accuracyType,'Accuracy')
        perms(j+Nperms0,:,1) = 1 - mean(abs( sign(Yhat_constr) - sign(Y) ) / 2,2);
        perms(j+Nperms0,:,2) = 1 - mean(abs( sign(Yhat) - sign(Y) ) / 2,2);  
    else % COD
        m = repmat(mean(Y,2),1,N); m = sum((m-Y).^2,2);
        perms(j+Nperms0,:,1) = 1 - sum((Yhat_constr-Y).^2,2) ./ m;
        perms(j+Nperms0,:,2) = 1 - sum((Yhat-Y).^2,2) ./ m;
    end
    % Standard decoding
    if do_stddecoding
        X = reshape(X,[ttrial N (p+1)]);
        m = mean(Y(1,:)); m = sum((Y(1,:)-m).^2);
        for t = 1:ttrial
            x = permute(X(t,:,1:p),[2 3 1]);
            y = Y(t,:)';
            b = (x' * x) \ (x' * y);
            yhat = x*b; 
            if strcmp(accuracyType,'Pearson')
                perms_std(j+Nperms0,t) = corr(y,yhat);
            elseif strcmp(accuracyType,'SquaredErr')
                perms_std(j+Nperms0,t) = -sum((y-yhat).^2);
            elseif strcmp(accuracyType,'AbsErr') 
                perms_std(j+Nperms0,t) = -sum(abs(y-yhat));
            elseif strcmp(accuracyType,'Accuracy')
                perms_std(j+Nperms0,t) = 1 - mean(abs( sign(yhat) - sign(y) ) / 2);  
            else
                perms_std(j+Nperms0,t) = 1 - sum((y-yhat).^2) / m;
            end
        end
        X = reshape(X,[ttrial*N (p+1)]);
    end
    disp(num2str(j))
end
    
pvals = zeros(ttrial,3); Nperms = Nperms + Nperms0;
pvals_std = zeros(ttrial,1);

for t = 1:ttrial
   if strcmp(accuracyType,'Pearson') || strcmp(accuracyType,'COD')
       d = perms(:,t,2) - perms(:,t,1);
       pvals(t,1) = sum(d(1) <= d) / (Nperms+1);
       d = perms(:,t,1);
       pvals(t,2) = sum(d(1) <= d) / (Nperms+1);
       d = perms(:,t,2);
       pvals(t,3) = sum(d(1) <= d) / (Nperms+1);
       if do_stddecoding
           d = perms_std(:,t);
           pvals_std(t) = sum(d(1) <= d) / (Nperms+1);
       end
   else
       d = perms(:,t,1) - perms(:,t,2);
       pvals(t,1) = sum(d(1) <= d) / (Nperms+1);
       d = perms(:,t,1);
       pvals(t,2) = sum(d(1) >= d) / (Nperms+1);
       d = perms(:,t,2);
       pvals(t,3) = sum(d(1) >= d) / (Nperms+1);
       if do_stddecoding
           d = perms_std(:,t);
           pvals_std(t) = sum(d(1) >= d) / (Nperms+1);
       end
   end
end
    
end

% 
% y = residuals(:,end);
% v = unique(y);
% g = zeros(options.K,3);
% for jj = 1:3
%    g(:,jj) = sum(Gamma( y==v(jj),:)) / size(Gamma,1) ;
%    g(:,jj) = g(:,jj)/ sum(g(:,jj));
% end
% corr(g)



function betahat = adjustbetas(X,Y,Gamma,beta,options)

K = size(Gamma,2);
meanbeta = mean(sum(beta));
p = size(beta,1);
betahat = zeros(p,K);

for k = 1:K
   ind = Gamma(:,k)==1;
   x = X(ind,:); 
   y = Y(ind);
   betahat(:,k) = lsqlin(x,y,[],[],ones(1,p),meanbeta,[],[],[],options);
end

end


