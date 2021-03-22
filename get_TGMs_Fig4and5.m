%% Obtain the TGMs for for the original, non-oscillatory, oscillatory and power signal.
% This part is designed to run on a cluster, with one job per pair of images and subject
% so when calling this script, we should have specified s, j1 and j2.
% s ranges from 1 to 15 (subjects), and j1,j2 range from 1 to 118 (images)

datadir = '/vols/Scratch/HCP/rfMRI/diego/Radek/data/';
outdata = '/vols/Scratch/HCP/rfMRI/diego/Radek/out/';

if s<10
    str1 = ['subj0' num2str(s)];
else
    str1 = ['subj' num2str(s)];
end

options_sr = struct();
options_sr.alpha = [0.0100 0.1000 1 10 100 1000 10000];

dat1 = load([datadir 'by_condition/' str1 '_cond' num2str(j1) '_orig.mat']);
dat2 = load([datadir 'by_condition/' str1 '_cond' num2str(j2) '_orig.mat']);
dat1a = load([datadir 'by_condition/' str1 '_cond' num2str(j1) '_nonoscl.mat']);
dat2a = load([datadir 'by_condition/' str1 '_cond' num2str(j2) '_nonoscl.mat']);
dat1b = load([datadir 'by_condition/' str1 '_cond' num2str(j1) '_oscl.mat']);
dat2b = load([datadir 'by_condition/' str1 '_cond' num2str(j2) '_oscl.mat']);
dat1c = load([datadir 'by_condition/' str1 '_cond' num2str(j1) '_pow.mat']);
dat2c = load([datadir 'by_condition/' str1 '_cond' num2str(j2) '_pow.mat']);

X = [dat1.X; dat2.X];
Y = [-ones(size(dat1.X,1),1); +ones(size(dat2.X,1),1)];
T = [dat1.T; dat2.T];

Accuracy_orig = single(standard_regression_cross(X,Y,T,options_sr));

X = [dat1a.X; dat2a.X];
Y = [-ones(size(dat1a.X,1),1); +ones(size(dat2a.X,1),1)];
T = [dat1a.T; dat2a.T];

Accuracy_nonoscl = single(standard_regression_cross(X,Y,T,options_sr));

X = [dat1b.X; dat2b.X];
Y = [-ones(size(dat1b.X,1),1); +ones(size(dat2b.X,1),1)];
T = [dat1b.T; dat2b.T];

Accuracy_oscl = single(standard_regression_cross(X,Y,T,options_sr));

X = [dat1c.X; dat2c.X];
Y = [-ones(size(dat1c.X,1),1); +ones(size(dat2c.X,1),1)];
T = [dat1c.T; dat2c.T];

Accuracy_pow = single(standard_regression_cross(X,Y,T,options_sr));


save([outdata 'by_condition/' str1 '_cond' num2str(j1) '_' num2str(j2) '.mat'],...
    'Accuracy_nonoscl','Accuracy_oscl','Accuracy_orig','Accuracy_pow');


%% This part used to merge all interim results into one single file per subject,
% so when calling this part we need to specify s, between 1 and 15.

outdata = '../scratch/out/Radek/';

if s<10
    str1 = ['subj0' num2str(s)];
else
    str1 = ['subj' num2str(s)];
end

q = 118; ttrial = 250; qq = q * (q-1) / 2;

Accuracy_orig_crossT = zeros(ttrial,ttrial,q);
Accuracy_nonoscl_crossT = zeros(ttrial,ttrial,q);
Accuracy_oscl_crossT = zeros(ttrial,ttrial,q);
Accuracy_pow_crossT = zeros(ttrial,ttrial,q);

Accuracy_orig_mean_crossT = zeros(ttrial,ttrial);
Accuracy_nonoscl_mean_crossT = zeros(ttrial,ttrial);
Accuracy_oscl_mean_crossT = zeros(ttrial,ttrial);
Accuracy_pow_mean_crossT = zeros(ttrial,ttrial);

Accuracy_orig_crossq = zeros(q,q,ttrial);
Accuracy_nonoscl_crossq = zeros(q,q,ttrial);
Accuracy_oscl_crossq = zeros(q,q,ttrial);
Accuracy_pow_crossq = zeros(q,q,ttrial);

Accuracy_orig_mean_crossq = zeros(ttrial,1);
Accuracy_nonoscl_mean_crossq = zeros(ttrial,1);
Accuracy_oscl_mean_crossq = zeros(ttrial,1);
Accuracy_pow_mean_crossq = zeros(ttrial,1);

for j1 = 1:q-1
    for j2 = j1+1:q
        
        out = load([outdata 'by_condition/' str1 '_cond' num2str(j1) '_' num2str(j2) '.mat'],...
            'Accuracy_orig','Accuracy_nonoscl','Accuracy_oscl','Accuracy_pow');
        
        Accuracy_orig_crossT(:,:,j1) = Accuracy_orig_crossT(:,:,j1) + out.Accuracy_orig / (q-1);
        Accuracy_nonoscl_crossT(:,:,j1) = Accuracy_nonoscl_crossT(:,:,j1) + out.Accuracy_nonoscl / (q-1);
        Accuracy_oscl_crossT(:,:,j1) = Accuracy_oscl_crossT(:,:,j1) + out.Accuracy_oscl / (q-1);
        Accuracy_pow_crossT(:,:,j1) = Accuracy_pow_crossT(:,:,j1) + out.Accuracy_pow / (q-1);
        
        Accuracy_orig_crossT(:,:,j2) = Accuracy_orig_crossT(:,:,j2) + out.Accuracy_orig / (q-1);
        Accuracy_nonoscl_crossT(:,:,j2) = Accuracy_nonoscl_crossT(:,:,j2) + out.Accuracy_nonoscl / (q-1);
        Accuracy_oscl_crossT(:,:,j2) = Accuracy_oscl_crossT(:,:,j2) + out.Accuracy_oscl / (q-1);
        Accuracy_pow_crossT(:,:,j2) = Accuracy_pow_crossT(:,:,j2) + out.Accuracy_pow / (q-1);
        
        Accuracy_orig_mean_crossT = Accuracy_orig_mean_crossT + out.Accuracy_orig / qq;
        Accuracy_nonoscl_mean_crossT = Accuracy_nonoscl_mean_crossT + out.Accuracy_nonoscl / qq;
        Accuracy_oscl_mean_crossT = Accuracy_oscl_mean_crossT + out.Accuracy_oscl / qq;
        Accuracy_pow_mean_crossT = Accuracy_pow_mean_crossT + out.Accuracy_pow / qq;
        
        Accuracy_orig_crossq(j1,j2,:) = diag(out.Accuracy_orig);
        Accuracy_nonoscl_crossq(j1,j2,:) = diag(out.Accuracy_nonoscl);
        Accuracy_oscl_crossq(j1,j2,:) = diag(out.Accuracy_oscl);
        Accuracy_pow_crossq(j1,j2,:) = diag(out.Accuracy_pow);
        
        Accuracy_orig_mean_crossq = Accuracy_orig_mean_crossq + diag(out.Accuracy_orig) / qq;
        Accuracy_nonoscl_mean_crossq = Accuracy_nonoscl_mean_crossq + diag(out.Accuracy_nonoscl) / qq;
        Accuracy_oscl_mean_crossq = Accuracy_oscl_mean_crossq + diag(out.Accuracy_oscl) / qq;
        Accuracy_pow_mean_crossq = Accuracy_pow_mean_crossq + diag(out.Accuracy_pow) / qq;
        
    end
end


save([outdata 'crossDecAcc_' str1 '.mat'],...
    'Accuracy_orig_crossT','Accuracy_nonoscl_crossT','Accuracy_oscl_crossT','Accuracy_pow_crossT',...
    'Accuracy_orig_mean_crossT','Accuracy_nonoscl_mean_crossT','Accuracy_oscl_mean_crossT','Accuracy_pow_mean_crossT',...
    'Accuracy_orig_crossq','Accuracy_nonoscl_crossq','Accuracy_oscl_crossq','Accuracy_pow_crossq',...
    'Accuracy_orig_mean_crossq','Accuracy_nonoscl_mean_crossq','Accuracy_oscl_mean_crossq','Accuracy_pow_mean_crossq') %,...


%% Show TGMs for for the original, non-oscillatory, oscillatory and power signal. 

M1 = zeros(250); % original signal
M2 = zeros(250); % non-oscillatory
M3 = zeros(250); % oscillatory
M4 = zeros(250); % power

% Aggregate across subjects
for subj = subjs
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    
    load(['out/crossDecAcc_' str1 '.mat']) % saved in the previous block
    
    M = mean(Accuracy_orig_crossT,3); M = M(end:-1:1,:);
    M1 = M1 + M / N;
    M = mean(Accuracy_nonoscl_crossT,3); M = M(end:-1:1,:);
    M2 = M2 + M / N;
    M = mean(Accuracy_oscl_crossT,3); M = M(end:-1:1,:);
    M3 = M3 + M / N;
    M = mean(Accuracy_pow_crossT,3); M = M(end:-1:1,:);
    M4 = M4 + M / N;
end

% plot the four TGMs
figure(300+subj); clf(300+subj);
imagesc(M1);colorbar; caxis([0.4 0.7]); axis square
hold on; plot([1 250],[250 1],'k','LineWidth',2); hold off
L = 0:50:250; L = L(2:end); L = L/250;
ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'xticklabel',L,'ytick',[1 50 100 150 200 250],'yticklabel',[L(end:-1:1) 0])
title('Original signal TGM')
set(gca,'FontSize',14)

figure(400+subj); clf(400+subj);
imagesc(M2);colorbar; caxis([0.4 0.7]); axis square
hold on; plot([1 250],[250 1],'k','LineWidth',2); hold off
L = 0:50:250; L = L(2:end); L = L/250;
ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'xticklabel',L,'ytick',[1 50 100 150 200 250],'yticklabel',[L(end:-1:1) 0])
title('1/f TGM')
set(gca,'FontSize',14)

figure(500+subj); clf(500+subj);
imagesc(M3);colorbar; caxis([0.4 0.7]); axis square
hold on; plot([1 250],[250 1],'k','LineWidth',2); hold off
L = 0:50:250; L = L(2:end); L = L/250;
ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'xticklabel',L,'ytick',[1 50 100 150 200 250],'yticklabel',[L(end:-1:1) 0])
title('Oscillations TGM')
set(gca,'FontSize',14)

figure(600+subj); clf(600+subj);
M = mean(Accuracy_pow_crossT,3); M = M(end:-1:1,:);
imagesc(M4);colorbar; caxis([0.4 0.7]); axis square
hold on; plot([1 250],[250 1],'k','LineWidth',2); hold off
L = 0:50:250; L = L(2:end); L = L/250;
ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'xticklabel',L,'ytick',[1 50 100 150 200 250],'yticklabel',[L(end:-1:1) 0])
%title('Power TGM')
set(gca,'FontSize',14)


%% How well can we predict the whole signal TGM as a function
% of the non-oscillatory signal and the oscillation?

beta1 = [];
beta2 = [];
beta3 = [];
r2 = zeros(5,N);
fstats = zeros(2,N); 
pvals = zeros(2,N);

for subj = 1:15
    
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    load(['out/crossDecAcc_' str1 '.mat'])
    
    x = Accuracy_orig_crossT(:);  
    x1 = Accuracy_nonoscl_crossT(:);
    x2 = Accuracy_oscl_crossT(:);
    x3 = Accuracy_pow_crossT(:);
    D = [ones(length(x1),1) x1 x2 x3]; %clear x1 x2
    [B,BINT,R,RINT,STATS] = regress(x,D); % all
    beta1 = [beta1 B(2:4)];
    r2(1,subj) = STATS(1);
    yhat = D * B; 
    RSS1 = norm(yhat-mean(x))^2;
        
    D = [ones(length(x1),1) x1 x2]; %clear x1 x2
    [B,BINT,R,RINT,STATS] = regress(x,D); % no power
    yhat = D * B; 
    RSS2 = norm(yhat-mean(x))^2;  
    beta2 = [beta2 B(2:3)];
    r2(2,subj) = STATS(1);
    
    D = [ones(length(x1),1) x1 x3]; %clear x1 x2
    [B,BINT,R,RINT,STATS] = regress(x,D); % no phase
    yhat = D * B; 
    RSS3 = norm(yhat-mean(x))^2;  
    beta3 = [beta3 B(2:3)];
    r2(3,subj) = STATS(1);
    
    D = [ones(length(x1),1) x1]; %clear x1 x2
    [B,BINT,R,RINT,STATS] = regress(x,D); % no phase
    yhat = D * B; 
    RSS4 = norm(yhat-mean(x))^2;  
    r2(4,subj) = STATS(1);
    
    D = [ones(length(x1),1) x2]; %clear x1 x2
    [B,BINT,R,RINT,STATS] = regress(x,D); % no phase
    yhat = D * B; 
    RSS5 = norm(yhat-mean(x))^2;  
    r2(5,subj) = STATS(1);
    
    fstats(1,subj) = (RSS2 - RSS1) / (RSS1 / (250 - 250*249/2 - 3));
    fstats(2,subj) = (RSS3 - RSS1) / (RSS1 / (250 - 250*249/2 - 3));
    pvals(1,subj) = fpval(fstats(1,subj),1,250 - 3 ); % Significance probability for regression
    pvals(2,subj) = fpval(fstats(2,subj),1,250 - 3 ); % 250*249/2 Significance probability for regression
     
end

(r2(1,1:15) - r2(3,1:15))
(r2(1,1:15) - r2(3,1:15)) ./ r2(3,1:15)

mean((r2(1,1:15) - r2(3,1:15)))
mean((r2(1,1:15) - r2(3,1:15)) ./ r2(3,1:15))

%% How oscillations can affect accuracy?
% Show the decrease of accuracy to levels below baseline just off the diagonal

    M1 = zeros(250);
    m1 = zeros(1,100);

for subj = 1:15
    
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    
    load(['out/crossDecAcc_' str1 '.mat'])
    
    M = mean(Accuracy_oscl_crossT,3); M = M(end:-1:1,:);
    M1 = M1 + M / N;
    
    I = false(250); for t = 1:100, I((250-100)+t-1,t) = true; end
    m = M(I);
    m1 = m1 + m / N;
    
end

figure(400); clf(400);
imagesc(M1);colorbar; caxis([0.4 0.7]); axis square
hold on;
plot([1 250],[250 1],'k','LineWidth',2);
plot([1 100],[250-100 250],':k','LineWidth',4);
hold off
L = 0:50:250; L = L(2:end); L = L/250;
%ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'ytick',[250-99 250-49 250-9],'yticklabel',[100/250 50/250 10/250],...
    'xtick',[10 50 100],'xticklabel',[10/250 50/250 100/250])
%title('Oscillations TGM')
%xlim([10 100]); ylim([250-99 250-9]);
xlim([1 100]); ylim([250-100 250]);
set(gca,'FontSize',20)

figure(500+subj); clf(500+subj);
plot((1:100)/250,m1,'k','LineWidth',4)
hold on;
plot([1 100]/250, [0.5 0.5],':k','LineWidth',4)
[~,im] = max(m);
plot([im im]/250, [0.35 0.75],'k')
hold off
%xlabel('Time (s)'); ylabel('Accuracy')
set(gca,'FontSize',20)



%% Vertical and horizontal bars of higher accuracy in the cross-decoding matrix


M1 = zeros(250); % original signal
M2 = zeros(250); % non-oscillatory
M3 = zeros(250); % oscillatory
M4 = zeros(250); % power

for subj = 1:15
    
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    
    load(['out/crossDecAcc_' str1 '.mat'])

    M = mean(Accuracy_orig_crossT,3); M = M(end:-1:1,:);
    M1 = M1 + M / length(subjs);
    
    M = mean(Accuracy_nonoscl_crossT,3); M = M(end:-1:1,:);
    M2 = M2 + M / length(subjs);
    
    M = mean(Accuracy_oscl_crossT,3); M = M(end:-1:1,:);
    M3 = M3 + M / length(subjs);
    
    M = mean(Accuracy_pow_crossT,3); M = M(end:-1:1,:);
    M4 = M4 + M / length(subjs);

end


m = diag(M1(250:-1:1,:)); [~,im] = max(m); 

m1 = M1(:,im); m2 = M2(:,im); m3 = M3(:,im);
[b1,~,~,~,stats1] = regress(m1,[ones(250,1) m2 m3]); stats1(1)
m1 = M1(250-im+1,:)'; m2 = M2(250-im+1,:)'; m3 = M3(250-im+1,:)';
[b2,~,~,~,stats2] = regress(m1,[ones(250,1) m2 m3]); stats2(1)

% TGM_whole_t150_te, model tested at t=150
figure(509);clf(509)
imagesc(M1);colorbar; caxis([0.5 0.55]); axis square
hold on; 
plot([1 250],[250 1],'k','LineWidth',2); 
plot([1 250],[250-im+1 250-im+1],':k','LineWidth',4); 
plot([im im],[250 1],':k','LineWidth',4); 
hold off
L = 0:50:250; L = L(2:end); L = L/250; 
%ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'xticklabel',L,'ytick',[1 50 100 150 200 250],'yticklabel',[L(end:-1:1) 0])
%title('Original signal TGM')
set(gca,'FontSize',20)

% TGM_whole_t150_tr, model trained at t=150
figure(510);clf(510)
r = 250:-1:1;
hold on
plot((1:250)/250,[M1(r,im) M2(r,im) M3(r,im) ],'LineWidth',3)
plot((1:250)/250,[(b1(1) + b1(2)*M2(r,im) + b1(3)*M3(r,im)) ],':','Color',[0 0.25 1],'LineWidth',3)
hold off
ylim([0.38 0.71])
%xlabel('Generalisation time (s)'); ylabel('Accuracy')
%title('Model tested at t=150ms')
%l=legend('Original signal','1/f','Oscillations','Linear combination: 1/f + Oscillations');
%set(l,'FontSize',16)
set(gca,'FontSize',20)

figure(511);clf(511)
hold on
plot((1:250)/250,[M1(250-im+1,:)' M2(250-im+1,:)' M3(250-im,:)']','LineWidth',3)
plot((1:250)/250,[(b2(1) + b2(2)*M2(250-im+1,:)' + b2(3)*M3(250-im+1,:)') ],':','Color',[0 0.25 1],'LineWidth',3)
hold off
ylim([0.39 0.71])
%xlabel('Generalisation time (s)'); ylabel('Accuracy')
%title('Model trained at t=150ms')
%l=legend('Original signal','1/f','Oscillations','Linear combination: 1/f + Oscillations');
%set(l,'FontSize',16)
set(gca,'FontSize',20)

figure(512); clf(512);
imagesc(M2);colorbar; caxis([0.5 0.6]); axis square
hold on; plot([1 250],[250 1],'k','LineWidth',2); hold off
L = 0:50:250; L = L(2:end); L = L/250;
ylabel('Training time (s)'); xlabel('Generalization time (s)')
set(gca,'xticklabel',L,'ytick',[1 50 100 150 200 250],'yticklabel',[L(end:-1:1) 0])
%title('1/f TGM')
set(gca,'FontSize',18)
hold on; 
plot([1 250],[250 1],'k','LineWidth',2); 
plot([1 250],[250-im 250-im],':k','LineWidth',4); 
plot([im im],[250 1],':k','LineWidth',4); 
hold off 
xlim([1 100]); ylim([250-100 250]);


