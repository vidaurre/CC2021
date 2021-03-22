% Prepare the data and separate the different components: oscillatory from
% non-oscillatory. 

% Data downloaded from http://userpage.fu-berlin.de/rmcichy/fusion_project_page/main.html
%
% for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
% do
%    tar xvzf subj${i}_sess01.tar.gz
%    rm subj${i}_sess01.tar.gz
% done

do_filtering = true; 
do_pca = false; 

datadir = '../data/RadekCichy/MEG_118objects/';

cd(datadir)
hmmdir = '../../../HMM-MAR/';
addpath(genpath(hmmdir))

for subj = 1:15
    ttrial = 250; nchannels = 306; ncond = 118; ntrial = 30;
    X = zeros(ttrial * ncond * ntrial,  nchannels);
    Y = zeros(ttrial * ncond * ntrial, ncond,'single');
    discounted_trials = 0; ind_discard = [];
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];
    end
    for cond = 1:ncond
        t0_1 = (cond-1)*(ntrial*ttrial);
        if cond<10
            str2 = ['cond000' num2str(cond)];
        elseif cond<100
            str2 = ['cond00' num2str(cond)];
        else
            str2 = ['cond0' num2str(cond)];
        end
        for trial = 1:ntrial
            t0_2 = (trial-1)*ttrial;
            ind = (1:ttrial) + t0_1 + t0_2; 
            if trial<10
                str3 = ['/trial00' num2str(trial)];
            else
                str3 = ['/trial0' num2str(trial)];
            end
            file = [datadir str1 '/sess01/' str2 str3 '.mat'];
            dat = load(file);
            F = dat.F';
            F = F - repmat(mean(F(1:100,:)),size(F,1),1);
            F = F ./ repmat(std(F(1:100,:)),size(F,1),1);
            if do_filtering
                x = filterdata(F,size(F,1),1000,[0 10]);
                x = downsampledata(x,size(F,1),250,1000);
            else
                x = downsampledata(F,size(F,1),250,1000);
            end
            if any(isnan(x(:))) 
                discounted_trials = discounted_trials + 1; 
                ind_discard = [ind_discard ind];
            end
            X(ind,:) = x(21:end-6,:);
            Y(ind,cond) = 1;
        end
    end
    X(ind_discard,:) = []; Y(ind_discard,:) = [];
    N = (ntrial * ncond) - discounted_trials;    
    T = ttrial  * ones(N,1);
    if do_pca
        [A,X,e] = pca(X);
        e = cumsum(e)/sum(e); npca = find(e>0.99,1);
        X = X(:,1:npca); A = A(:,1:npca);
        str2 = '';
    else
        A = []; e = []; str2 = '_nopca';
    end
    if do_filtering
        save([str1 str2 '.mat'],'X','Y','A','e','T','-V7.3')
    else
        save([str1 str2 '_nofilt.mat'],'X','Y','A','e','T','-V7.3')
    end
    disp(str1) % ' done - PCAS: '  num2str(npca) ]) 
end


%% Separate the data between non-oscillatory and oscillatory parts

for subj = 2
    
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];      
    end    
    
    load([datadir str1 '_nopca.mat'])

    smooth_par = 100; 
   
    [X1,X2] = split_data_f(X,T,smooth_par);
    X0 = X; 
     
    p = size(X1,2); N = length(T); ttrial = T(1); q = 118;

  
    for j = 1:q
    
        ind = Y(:,j) == 1;
        ind_star = reshape(ind,ttrial,N);
        ind_star = ind_star(1,:);
        Njj = sum(ind_star);
        T = ttrial * ones(Njj,1);
        
        X = X0(ind,:);
        save([datadir 'by_condition/' str1 '_cond' num2str(j) '_orig.mat'],'X','T');  
        
        X = X1(ind,:);
        save([datadir 'by_condition/' str1 '_cond' num2str(j) '_nonoscl.mat'],'X','T');  
        
        X = X2(ind,:);
        save([datadir 'by_condition/' str1 '_cond' num2str(j) '_oscl.mat'],'X','T');  
        
    end

    disp([str1 ' done ']) 
end


%% Get power from the oscillatory part 

addpath('../HMM-MAR/utils/preproc/')

for subj = 1:15
    
    if subj<10
        str1 = ['subj0' num2str(subj)];
    else
        str1 = ['subj' num2str(subj)];      
    end    
    
    ttrial = 250; p = 306; q = 118; 

    for j = 1:q
        
        load([datadir 'by_condition/' str1 '_cond' num2str(j) '_oscl.mat'])
        X = rawsignal2power(X,T);
        save([datadir 'by_condition/' str1 '_cond' num2str(j) '_pow.mat'],'X','T');  
    
    end

    disp([str1 ' done ']) 
end

