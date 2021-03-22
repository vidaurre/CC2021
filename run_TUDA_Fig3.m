
addpath(genpath('../../HMM-MAR')) % Set to your local HMM-MAR directory
addpath('./DecodingMethods')
addpath('../PermutationTesting')


% Cichy's data is available on 
%   http://userpage.fu-berlin.de/rmcichy/fusion_project_page/main.html
datadir = '../data/RadekCichy/MEG_118objects/by_condition/'; % set to your local data directory
beh = load('118_visual_stimuli_Khaligh_et_al_2018.mat'); % struct with the behaviourals


do_size = 0; % 1 for size, 0 for animacy

for subj = 1:15
    
    %%% Load the data
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    X2 = []; Y = []; T = []; ttrial = 250;
    for j = 1:118
        % load the phasic component only
        dat = load([datadir str1 '_cond' num2str(j) '_oscl.mat']);
        X2 = [X2; dat.X];
        % encode the behavioural as...
        if do_size % ... 1,2,3 size categories
            if beh.visual_stimuli(j).small_size
                Y = [Y; ones(size(dat.X,1),1)];
            elseif beh.visual_stimuli(j).medium_size
                Y = [Y; 2*ones(size(dat.X,1),1)];
            elseif beh.visual_stimuli(j).large_size
                Y = [Y; 3*ones(size(dat.X,1),1)];
            end
        else % ... animate object or not
            if beh.visual_stimuli(j).animate
                Y = [Y; ones(size(dat.X,1),1)];
            else
                Y = [Y; -ones(size(dat.X,1),1)];
            end
        end
        N = size(dat.X,1) / ttrial;
        T = [T; ttrial*ones(N,1)];
        %disp(num2str(j))
    end
    
    %%% Standardise and get the first principal components
    X2 = zscore(X2);
    [A,Xpca,e] = pca(X2); e = cumsum(e)/sum(e);
    X2 = Xpca(:,1:116); e(116)
    clear Xpca
    
    %%% Run permutation testing on TUDA 
    % basic options
    options = struct();
    options.K = 8; % no. of states 
    options.inittype = 'sequential'; % state 1 is followed by 2, then by 3...
    options.verbose = 1;
    options.useParallel = 0;
    options.Nperms = 100;
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
    [pvals_oscl,perms_oscl,~,pvals_oscl_std,perms_oscl_std] = tudatest(X2,Y,T,options);
    % save
    if do_size
        save(['out/predict_size_' str1 '.mat'],...
            'pvals_oscl_std','perms_oscl_std','pvals_oscl','perms_oscl')
    else
        save(['out/predict_animate_' str1 '.mat'],...
            'pvals_oscl_std','perms_oscl_std','pvals_oscl','perms_oscl')
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
    
    %%% Additional tests 
    % different timing for different categories? 
    pvalsTimingGamma = testGammaAcrossStimuli (Gamma_tudacv,Y,T,1000,250);
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