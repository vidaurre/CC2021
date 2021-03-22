function [pval,perms] = testPLF_timevarying(X,Y,T,Gamma,Nperm)
% is PLF stronger between than across classes, at the peak time of each
% state?

N = length(T); ttrial = T(1);
if length(size(X))==3
    p = size(X,3);
else
    p = size(X,2);
    X = reshape(X,[ttrial N p]);
end
if length(size(Gamma))==2
    K = size(Gamma,2);
    Gamma = reshape(Gamma,[ttrial N K]);
else
    K = size(Gamma,3);
end
if size(Y,1) == (ttrial*N)
   Y = reshape(Y,ttrial,N); Y = Y(1,:)'; 
end
ucl = unique(Y); 
if length(Y)~=N || length(ucl)~=2
   error('Incorrect format for Y') 
end

if nargin < 4, Nperm = 1000; end
N1 = sum(Y==ucl(1));
N2 = sum(Y==ucl(2));
cl0 = Y;

pval = zeros(K,1); 
perms = zeros(Nperm,K);

for j = 1:p
    Ph = zeros(ttrial,N);
    for n = 1:N
        h = hilbert(X(:,n,j));
        s = angle(h);
        Ph(:,n) = exp(1i * s);
    end
    for k = 1:K
        if k == 1 % before it decays
            [~,t] = find(Gamma(:,j,1)<0.999,1); t = t-1; 
            if t==0, t = 1; end
        elseif k==K % as soon as it becomes dominant
            [~,t] = find(Gamma(:,j,K)>0.999,1); t = t+1; 
            if isempty(t) || t>ttrial, t = ttrial; end
        else
            [~,t] = max(Gamma(:,j,k)); 
        end
        Y = cl0;
        for r = 1:Nperm
            if r > 1
                Y = Y(randperm(N));
            end
            ind1 = Y==ucl(1);
            ind2 = Y==ucl(2);
            plf1 = abs(sum( Ph(t,ind1) )) / N1;
            plf2 = abs(sum( Ph(t,ind2) )) / N2;
            perms(r,k) = perms(r,k) + (plf1 + plf2) / 2;
        end
    end
end

for k = 1:K
    pval(k) = sum(perms(1,k) <= perms(:,k)) / (Nperm+1);
end

end