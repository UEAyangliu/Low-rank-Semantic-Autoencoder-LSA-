%% SAE + low rank
clc
clear
addpath library
dataset = 'APY';
%% load data
for fold=1:1
    if(strcmp(dataset, 'AWA1'))
        load F:\RSAE\xlsa17\data\AWA1\att_splits.mat
        load F:\RSAE\xlsa17\data\AWA1\res101.mat

        
        Ytr = labels(trainval_loc);
        Xtr = features(:,trainval_loc);  Xtr = NormalizeFea(Xtr);
        Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
        Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
        %     att = NormalizeFea(att);
    elseif(strcmp(dataset, 'AWA2'))
        load F:\RSAE\xlsa17\data\AWA2\att_splits.mat
        load F:\RSAE\xlsa17\data\AWA2\res101.mat

        
        Ytr = labels(trainval_loc);
        Xtr = features(:,trainval_loc);  Xtr = NormalizeFea(Xtr);
        Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts);
        Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu);
        
    elseif(strcmp(dataset, 'CUB'))
        load F:\RSAE\xlsa17\data\CUB\att_splits.mat
        load F:\RSAE\xlsa17\data\CUB\res101.mat
        alpha = 1;
        lambda = [0.5];
        Ytr = labels(trainval_loc);
        Xtr = features(:,trainval_loc);
        Xtr = NormalizeFea(Xtr')';
        Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
        Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
        
        %     att = NormalizeFea(att);
    elseif(strcmp(dataset, 'SUN'))
        load F:\RSAE\xlsa17\data\SUN\att_splits.mat
        load F:\RSAE\xlsa17\data\SUN\res101.mat
        Ytr = labels(train_loc);
        Xtr = features(:,train_loc);  
        Xtr = NormalizeFea(Xtr')';
        Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
        Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';
        %     att = NormalizeFea(att);
        
    elseif(strcmp(dataset, 'APY'))
        load F:\RSAE\xlsa17\data\APY\att_splits.mat
        load F:\RSAE\xlsa17\data\APY\res101.mat
        Ytr = labels(trainval_loc);
        Xtr = features(:,trainval_loc);  Xtr = NormalizeFea(Xtr')';
        Xts = features(:,test_seen_loc);  Xts = NormalizeFea(Xts')';
        Xtu = features(:,test_unseen_loc);  Xtu = NormalizeFea(Xtu')';

    end
    
    
    Yts = labels(test_seen_loc);
    Ytu = labels(test_unseen_loc);
    
    tr_ind = unique(Ytr,'stable');
    ts_ind = unique(Yts,'stable');
    tu_ind = unique(Ytu,'stable');
    tu_attr = att(:,tu_ind);
    ts_attr = att(:,ts_ind);
    
    %%%%% augment attributes A to Ax
    Ax = zeros([size(att,1) size(Xtr,1)]);
    for ii = 1:length(tr_ind)
        ind = find(Ytr==tr_ind(ii));
        Ax(:,ind) = repmat(att(:,tr_ind(ii)), 1, length(ind));
    end
end
%% auto tain the W
paraLambda = linspace(0.01,1000,50);
paraMiu = linspace(0.01,1000,50);


main_start = tic;
for i = 1:size(paraLambda,2)
    for j = 1:size(paraMiu,2)
        params.lambda = paraLambda(i);
        params.miu = paraMiu(j);
        [A,E,W,Y1,Y2,Y3,fun1All,fun2All,fun3All,result,obj1,obj2,obj3,obj4,predict_label] = RSAE_fun(Xtr,Ax,params,tu_attr,att,Xtu,Xts,tu_ind,ts_ind,Ytu,Yts);
    end
end
toc(main_start);
predict_label_disorder = predict_label;