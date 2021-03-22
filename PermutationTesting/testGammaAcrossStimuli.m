function [pvals,changepoints] = testGammaAcrossStimuli (Gamma,Y,T,Nperm,Fs)
% Can Gamma predict the stimulus?

if nargin<5, Fs = 1; end

N = length(T); ttrial = T(1); 

if size(Y,1) == sum(T)
   Y = reshape(Y,ttrial,N); Y = Y(1,:)'; 
end

if length(size(Gamma))==2
    K = size(Gamma,2);
    Gamma = reshape(Gamma,[ttrial N K]);
else
    K = size(Gamma,3);
end

changepoints = zeros(K-1,1);
pvals = zeros(K-1,1);

for k = 2:K 
    for j = 1:N
       [~,t0] = max(Gamma(:,j,k-1)); 
       t = find(Gamma(t0:end,j,k) > Gamma(t0:end,j,k-1),1);
       if isempty(t), changepoints(j,k-1) = ttrial / Fs; 
       else, changepoints(j,k-1) = (t+t0-1) / Fs;
       end
    end
    pvals(k-1) = permtestcorr(changepoints(:,k-1),Y,Nperm);
end

end
