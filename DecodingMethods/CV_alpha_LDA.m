function best_alpha = CV_alpha_LDA(X,Y,alpha,responses,NCV)

[N,p] = size(X); 
NA = length(alpha);

if nargin < 5, NCV = 10; end
if nargin<4 || isempty(responses)
    c2 = cvpartition(N,'KFold',NCV);
else
    c2 = cvpartition(responses,'KFold',NCV);
end
c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = find(c2.training(icv));
    c.test{icv} = find(c2.test(icv));
end; clear c2

accuracy = zeros(NA,NCV);
for icv = 1:NCV
    Xtrain = X(c.training{icv},:); 
    Ytrain = Y(c.training{icv});
    Xtest = X(c.test{icv},:); Nte = size(Xtest,1); 
    Ytest = Y(c.test{icv});
    m1 = mean(Xtrain(Ytrain==-1,:));  
    m2 = mean(Xtrain(Ytrain==1,:));
    S = Xtrain' * Xtrain / N;
    for ialpha = 1:NA
        w = ((1 - alpha(ialpha)) * S  + alpha(ialpha) * eye(p)) \ (m2 - m1)';
        Yhat = -ones(Nte,1);
        cl = Xtest * w;
        Yhat(cl>0) = 1;
        accuracy(ialpha,icv) = mean(sign(Yhat)==sign(Ytest));
    end
end

accuracy = mean(accuracy,2);
[~,I] = max(accuracy);
best_alpha = alpha(I);

end

