function [zsl_unseen_acc,gzsl_unseen_acc,gzsl_seen_acc,H,zsl_unseen_predict_label] = evaluationCriteria(W,tu_attr,att,Xtu,Xts,tu_ind,ts_ind,Ytu,Yts)

%% test the zsl unseen, gzsl unseen, gzsl seen, H
% zsl unseen
% distance = 'euclidean';
distance = 'cosine';

X_te_pro = tu_attr' *W;
dist =  1 - (pdist2(Xtu', X_te_pro, distance));
[~, predict_label] = max(dist, [], 2);
zsl_unseen_predict_label = mapLabel(predict_label, tu_ind);
zsl_unseen_acc = computeAcc(zsl_unseen_predict_label, Ytu, tu_ind) * 100;
% fprintf('[1.2] %s ZSL accuracy [S >>> V]: %.1f%%\n', dataset, zsl_unseen_acc);

% gzsl unseen
X_te_pro = att' *W;
dist =  1 - (pdist2(Xtu', X_te_pro, distance));
[~, predict_label] = max(dist, [], 2);
gzsl_unseen_acc = computeAcc(predict_label, Ytu, tu_ind) * 100;
% fprintf('[2.2] %s GZSL unseen->all accuracy [S >>> V]: %.1f%%\n', dataset, gzsl_unseen_acc);

% gzsl seen
X_te_pro =  att' *W;
dist =  1 - (pdist2(Xts', X_te_pro, distance));
[~, predict_label] = max(dist, [], 2);
gzsl_seen_acc = computeAcc(predict_label, Yts, ts_ind) * 100;
% fprintf('[3.2] %s GZSL seen->all accuracy [S >>> V]: %.1f%%\n', dataset, gzsl_seen_acc*100);

% H
H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc);
% disp(['GZSL: H=' num2str(H) ' [S >>> V]']);