%{
## objective function
min norm(A,'nuclear') + lambda * norm(E,'L1')
s.t. X = A + E
     A = W' * S
     S = W * X
%}
function [A,E,W,Y1,Y2,Y3,fun1All,fun2All,fun3All,result,obj1,obj2,obj3,obj4,zsl_unseen_predict_label] = RSAE_fun(trX,S,params,tu_attr,att,Xtu,Xts,tu_ind,ts_ind,Ytu,Yts)
%% =====参数设置，初始化========================
X = trX;
obj1=0;obj2=0;obj3=0;obj4=0;zsl_unseen_predict_label=0;
% nfea：原始空间样本维度，nsamp：样本数
[nfea,nsamp] = size(X);
% nsemantic：语义空间样本维度
nsemantic = size(S,1);
% 最大的miu值
miuMax = 1e+6;
% 可调参数lambda
lambda = params.lambda;
% 参数rho，一般取1.1
rho = 1.1;
% 约束条件无穷范数的阈值
yita = 1e-3;
% 指定参数miu
miu = params.miu;
% 指定最大迭代次数
numIter = 100;
result = zeros(numIter,4);
% 指定初始化方式
initWays = 'zeros';
if strcmp(initWays,'zeros')
    A = zeros(nfea,nsamp);
    E = zeros(nfea,nsamp);
    Y1 = zeros(nfea,nsamp);
    Y2 = zeros(nfea,nsamp);
    Y3 = zeros(nsemantic,nsamp);
    W = zeros(nsemantic,nfea);
    
%     S1 = miu * S * S';
%     S2 = miu * X * X';
%     S3 = miu * (S*A'+S*X') + S*Y2' + Y3*X';
%     W = sylvester(S1,S2,S3);
elseif strcmp(initWays,'rand')
    A = rand(nfea,nsamp);
    E = rand(nfea,nsamp);
    Y1 = rand(nfea,nsamp);
    Y2 = rand(nfea,nsamp);
    Y3 = rand(nsemantic,nsamp);
    W = rand(nsemantic,nfea);
end
% 迭代次数
k = 1;
% 3个约束条件的无穷范数
fun1 = 1;
fun2 = 1;
fun3 = 1;
% 3个约束条件的无穷范数数组，每个元素代表一次迭代结果
fun1All = [];
fun2All = [];
fun3All = [];
% 是否写入文件
isWrite = true;
% 迭代循环体，fun1/2/3都小于yita时，或迭代达到最大次数时，停止循环
if isWrite
    fid = fopen('F:\RSAE\outputResults/result_apy.txt','a+');
    fprintf(fid,'lambda = %f, miu = %f\n', lambda, miu);
end
while  (fun1>=yita || fun2>=yita || fun3>=yita) && k<=numIter
    disp([num2str(k),','])
    tic
    
    % update A
    M = 0.5 * (X - E + W' * S) + (Y1 - Y2) / (2 * miu);
    [U,S_svd,V] = svd(M,'econ');
    miuNew = 1 / (2 * miu);
    [C] = threshold_wsj(S_svd,miuNew);% 求S(x)
    A = U * C * V';
    
    % update E
    temp = X - A + Y1 / miu;
    tau = lambda / miu;
    [E] = threshold_wsj(temp,tau);
    
    % update W
    S1 = miu * S * S';
    S2 = miu * X * X';
    S3 = miu * (S*A'+S*X') + S*Y2' + Y3*X';
    W = sylvester(S1,S2,S3);
    
    % update Y1,Y2 and Y3
    Y1 = Y1 + miu * (X - A - E);
    Y2 = Y2 + miu * (A - W' * S);
    Y3 = Y3 + miu * (S - W * X);
    
    % update miu
    miu = min(rho * miu, miuMax);
    
    % check convergence
    fun1 = norm(X - A -E,'inf');
    fun2 = norm(A - W' * S,'inf');
    fun3 = norm(S - W * X,'inf');
    fun1All = [fun1All fun1];
    fun2All = [fun2All fun2];
    fun3All = [fun3All fun3];
    
    % compute objective function value
    
    % obj1
    %     [~,sTmp,~] = svd(A);
    %     obj1(k) = sum(diag(sTmp)) + lambda * sum(sum(abs(E)));
    obj1(k)=0;
    
    % obj2
    %     tmp1 = (X-A-E);
    %     tmp2 = (A-W'*S);
    %     tmp3 = (S-W*X);
    %     obj2(k) = obj1(k) + trace(Y1'*tmp1) + trace(Y2'*tmp2) + trace(Y3'*tmp3) + (miu/2)*(norm(tmp1,'fro')^2 + norm(tmp2,'fro')^2 + norm(tmp3,'fro')^2);
    obj2(k)=0;
    
    % obj3
    %     [~,sTmp,~] = svd(W'*S);
    %     obj3(k) = sum(diag(sTmp)) + lambda * sum(sum(abs(X-W'*S)));
    obj3(k)=0;
    
    % obj4
    %     obj4(k) = obj3(k) + (miu/2)*norm(S-W*X+Y3/miu,'fro')^2;
    obj4(k)=0;
    
    % output the information
    [zsl_unseen_acc,gzsl_unseen_acc,gzsl_seen_acc,H,zsl_unseen_predict_label] = evaluationCriteria(W,tu_attr,att,Xtu,Xts,tu_ind,ts_ind,Ytu,Yts);
    result(k,:) = [zsl_unseen_acc,gzsl_unseen_acc,gzsl_seen_acc,H];
    
    fprintf('zsl_unseen_acc = %.1f, gzsl_unseen_acc = %.1f, gzsl_seen_acc = %.1f, H = %.1f,\n', zsl_unseen_acc,gzsl_unseen_acc,gzsl_seen_acc,H);
    time = toc;
    if isWrite
        fprintf(fid,'zsl_unseen_acc = %.1f, gzsl_unseen_acc = %.1f, gzsl_seen_acc = %.1f, H = %.1f,\n', zsl_unseen_acc,gzsl_unseen_acc,gzsl_seen_acc,H);
    end
%     if k==4
%         break
%     end
    k = k+1;
end
if isWrite
    fclose(fid);
end
disp('ok!');