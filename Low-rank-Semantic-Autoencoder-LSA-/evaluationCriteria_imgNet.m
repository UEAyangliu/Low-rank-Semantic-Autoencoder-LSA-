function [zsl_unseen_acc] = evaluationCriteria_imgNet(W,tu_attr,Xtu,tu_ind,Ytu)

%% test the zsl unseen, gzsl unseen, gzsl seen, H
% zsl unseen
X_te_pro = tu_attr' *W;
dist =  1 - (pdist2(Xtu', X_te_pro, 'cosine'));
[~, predict_label] = max(dist, [], 2);
zsl_unseen_predict_label = mapLabel(predict_label, tu_ind);
zsl_unseen_acc = computeAcc(zsl_unseen_predict_label, Ytu, tu_ind) * 100;
% fprintf('[1.2] %s ZSL accuracy [S >>> V]: %.1f%%\n', dataset, zsl_unseen_acc);