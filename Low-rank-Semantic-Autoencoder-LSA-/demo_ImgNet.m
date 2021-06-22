clear all
warning off
clc

% Loading the data
addpath('library')
load('data_zsl/ImNet_2_demo_data.mat')

%% Dimension reduction
W    = (X_tr'  * X_tr + 150*eye(size(X_tr'*X_tr)))^(-1)*X_tr'*(Y);
X_tr = X_tr * W;
X_te = X_te * W;

X_tr = Y;

% [Phi, y2] = Kmeans(X_tr', 256, 100); y2 = y2(:);
%% Learn projection
for lambda   = [11:20];
    W        = SAE(X_tr', S_tr', lambda);
    fprintf('\n lambda = %.3f \n', lambda);
    % for iter = 1:20
    %         M = (W*W' +1e-5*eye(size(S_tr,2)))^(-0.5);
    %         Wn = W;
    %         W = sylvester(S_tr'*S_tr+ alpha*M, lambda * (X_tr'*X_tr), (1 + lambda) * S_tr'*X_tr);
    %         if sum(sum((W-Wn).^2)) < 1e-3
    %             break
    %         end
    % end
    %%%%% Testing, nearest neighbour classification %%%%%
    %[F --> S], projecting data from feature space to semantic space
    S_te_est = X_te * W';
    dist     =  1 -  pdist2(zscore(S_te_est),zscore(S_te_pro')', 'cosine');  % 26.3%
    HITK     = 5;
    Y_hit5   = zeros(size(dist,1),HITK);
    for i  = 1:size(dist,1)
        [~, I] = sort(dist(i,:),'descend');
        Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
    end
    
    n=0;
    for i  = 1:size(dist,1)
        if ismember(Y_te(i),Y_hit5(i,:))
            n = n + 1;
        end
    end
    zsl_accuracy = n/size(dist,1);
    fprintf('\n[1] ImageNet-2 ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);
    
    
    %[S --> F], projecting from semantic to visual space
    dist  =  1 - (pdist2(X_te,zscore(W' * S_te_pro')', 'cosine')) ;    % 27.1%
    HITK   = 5;
    Y_hit5 = zeros(size(dist,1),HITK);
    for i  = 1:size(dist,1)
        [sort_dist_i, I] = sort(dist(i,:),'descend');
        Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
    end
    n = 0;
    for i  = 1:size(dist,1)
        if ismember(Y_te(i),Y_hit5(i,:))
            n = n + 1;
        end
    end
    zsl_accuracy = n/size(dist,1);
    fprintf('[2] ImageNet-2 ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);
end
clear i n Y_hit5 zsl_accuracy sort_dist_i lambda HITK dist I