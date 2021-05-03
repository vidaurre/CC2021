
addpath(genpath('../../HMM-MAR')) % Set to your local HMM-MAR directory
addpath('./DecodingMethods')
addpath('../PermutationTesting')


% Cichy's data is available on 
%   http://userpage.fu-berlin.de/rmcichy/fusion_project_page/main.html
datadir = '../data/RadekCichy/MEG_118objects/by_condition/'; % set to your local data directory
beh = load('118_visual_stimuli_Khaligh_et_al_2018.mat'); % struct with the behaviourals


do_size = 0; % 1 for size, 0 for animacy

changepoints = []; Stim = []; NN = []; % some things we'll need for the group-lvel testing

for subj = 1:15
    
    %%% Load the data
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    X2 = []; Y = []; stim = []; T = []; ttrial = 250; Nj = 0; 
    for j = 1:118
        % load the phasic component only
        dat = load([datadir str1 '_cond' num2str(j) '_oscl.mat']);
        X2 = [X2; dat.X];
        N = size(dat.X,1) / ttrial;
        % encode the behavioural as...
        if do_size % ... 1,2,3 size categories
            if beh.visual_stimuli(j).small_size
                Y = [Y; ones(size(dat.X,1),1)];
                stim = [stim; 1*ones(N,1)];
            elseif beh.visual_stimuli(j).medium_size
                Y = [Y; 2*ones(size(dat.X,1),1)];
                stim = [stim; 2*ones(N,1)];
            elseif beh.visual_stimuli(j).large_size
                Y = [Y; 3*ones(size(dat.X,1),1)];
                stim = [stim; 3*ones(N,1)];
            end
        else % ... animate object or not
            if beh.visual_stimuli(j).animate
                Y = [Y; ones(size(dat.X,1),1)];
                stim = [stim; 1*ones(N,1)];
            else
                Y = [Y; -ones(size(dat.X,1),1)];
                stim = [stim; -1*ones(N,1)];
            end
        end
        Nj = Nj + N;
        T = [T; ttrial*ones(N,1)];
        %disp(num2str(j))
    end
    Stim = [Stim; stim]; NN = [NN; Nj]; 
    
    %%% Standardise and get the first principal components
    X2 = zscore(X2);
    [A,Xpca,e] = pca(X2); e = cumsum(e)/sum(e);
    X2 = Xpca(:,1:116); e(116)
    clear Xpca
    
    %%% generate surrogates for TUDA (panel B)
    % basic options
    options = struct();
    options.K = 8; % no. of states 
    options.inittype = 'sequential'; % state 1 is followed by 2, then by 3...
    options.verbose = 1;
    options.useParallel = 0;
    options.Nperms = 1000; % if too expensive, set to 100 (but don't trust p-values)
    options.accuracyType = 'COD'; % coefficient of determination as base statistic
    % Provide some initial state time courses, heuristically initialised
    gc = cluster_decoding(X2,Y,T,options.K,'fixedsequential');
    gc2 = zeros(250,options.K); gc3 = zeros(250,options.K);
    for k = 1:options.K-1, gc2(:,k) = gc(:,k) + gc(:,k+1); end
    for k = 2:options.K, gc3(:,k) = gc(:,k-1) + gc(:,k); end
    gc2 = gc2 + gc3;
    gc2(gc2(:)>1) = 1;
    options.Gamma_constraint = gc2;
    % run
    [~,perms_oscl,~,~,perms_oscl_std] = tudatest(X2,Y,T,options);
    % save
    if do_size
        save(['out/predict_size_' str1 '.mat'],'perms_oscl_std','perms_oscl')
    else
        save(['out/predict_animate_' str1 '.mat'],'perms_oscl_std','perms_oscl')
    end
    
    %%% Cross-validation accuracy per state 
    options = rmfield(options,'Nperms'); 
    options.NCV = 10;
    % run
    [acc_state,acc_time,Gamma_tudacv] = tudacv_xstate(X2,Y,T,options);
    Gamma_tudacv=single(Gamma_tudacv);
    % save
    if do_size
        save(['out/predict_size_' str1 '.mat'],...
            'acc_state','acc_time','Gamma_tudacv', '-append')
    else
        save(['out/predict_animate_' str1 '.mat'],...
            'acc_state','acc_time','Gamma_tudacv','-append')
    end
    
    %%% Additional tests (panel C) 
    % different timing for different categories? 
    [pvalsTimingGamma,cp] = testGammaAcrossStimuli (Gamma_tudacv,Y,T,1000,250);
    changepoints = [changepoints; cp]; % save for later 
    % different phasic configurations at state peak time points between categories?
    uY = unique(Y); uYL = length(uY);
    pvalsPhaseAtMaxGamma = zeros(uYL,uYL,options.K);
    for j1 = 1:uYL-1
        for j2 = j1+1:uYL
            ind1 = Y==uY(j1); ind2 = Y==uY(j2); ind = ind1 | ind2; 
            indT = reshape(ind,[ttrial length(T)]); indT = indT(1,:);
            pvalsPhaseAtMaxGamma(j1,j2,:) = ...
                testPLF_timevarying(X(ind,:),Y(ind,:),T(indT),Gamma_tudacv(ind,:),1000); 
        end
    end
    if do_size
        save(['out/predict_size_' str1 '.mat'],...
            'pvalsTimingGamma','pvalsPhaseAtMaxGamma','-append')
    else
        save(['out/predict_animate_' str1 '.mat'],...
            'pvalsTimingGamma','pvalsPhaseAtMaxGamma','-append')
    end
    
    % Standard cross-generalised decoding w/ LDA and regression
    if do_size
        TGM_regression = standard_regression_cross(X2,Y,T);
        save(['out/predict_size_' str1 '.mat'],'TGM_regression','-append')
    else
        TGM_regression = standard_regression_cross(X2,Y,T);
        TGM_classification = standard_LDA_cross(X2,Y,T);
        save(['out/predict_animate_' str1 '.mat'],...
            'TGM_regression','TGM_classification','-append')
    end

    disp(['SUBJ ' num2str(subj) ])

end

%% tests at the group level : absolute phase modulations

pvalsTimingGamma = testGammaAcrossStimuli_group (changepoints,Stim,NN,10000);

if do_size
    save('out/group_tests_size.mat','pvalsTimingGamma')
else
    save('out/group_tests_animacy.mat','pvalsTimingGamma')
end


%% tests at the group level : relative phase modulations

% collect the permutations that we previously made
% (It's too much memory to work with all the data together)
perms = [];
for subj = 1:15
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    if do_size
        load(['out/tmp_predict_size_' str1 '.mat'],'perms_phase')
    else
        load(['out/tmp_predict_animate_' str1 '.mat'],'perms_phase')
    end
    perms = cat(2,perms,perms_phase);
end

ncontrasts = size(perms,1);

pvalsPhaseAtMaxGamma_perchan_group = zeros(options.K,p,ncontrasts);
pvalsPhaseAtMaxGamma_crosschan_group = zeros(options.K,ncontrasts);

for c = 1:ncontrasts
    [pvalsPhaseAtMaxGamma_perchan_group(:,:,c),...
        pvalsPhaseAtMaxGamma_crosschan_group(:,c)] = ...
        testPLF_timevarying_grouplevel(permute(perms(c,:,:,:),[2 3 4 1]),1000);
    % the number of permutations has to be the same as we used before
    % to create perm
end

if do_size
    save('out/group_tests_size.mat','pvalsPhaseAtMaxGamma_perchan_group',...
        'pvalsPhaseAtMaxGamma_crosschan_group','-append')
else
    save('out/group_tests_animacy.mat','pvalsPhaseAtMaxGamma_perchan_group',...
        'pvalsPhaseAtMaxGamma_crosschan_group','-append')
end



    