%% prepaer data
clear
% k: key value; d: date; c: count
[k,d,c] = textread('data.2012-2013.filtered.100','%s %d %f','delimiter','\t');
[kname, x, kid] = unique(k);
[dname, x, did] = unique(d);
clear k;
clear d;

ff = [];
for i = 1:length(kname)
    if strcmp(kname{i},'ss上证指数ss')
        ff(1) = i;
    end
    if strcmp(kname{i},'cc金融理财cc')
        ff(2) = i;
    end
end

m = full(sparse(did,kid,c));
f = find(m(:,ff(1)) > 0);
dat = m(f,:);
dnameall = dname;
dname = dnameall(f);
szzs = dat(:,ff(1)); % Y vector
jrlc = dat(:,ff(2));
[a,b] = size(dat);
ff1 = setdiff(1:b,ff);
dat = [dat(:,ff1)]; % X matrix
%%% smooth dat
% [a,b] = size(dat);
% for i = 1:b
%     st = std(dat(:,i));
%     me = mean(dat(:,i));
%     for j = 1:a
%         if dat(j,i) > me + 3*st;
%             dat(j,i) = me + 3*st;
%         end
%         if dat(j,i) < me - 3*st;
%             dat(j,i) = me - 3*st;
%         end
%     end
% end
%%%
%%% normalize dat
% [a,b] = size(dat);
% % dat = dat./repmat(jrlc,1,b);
% % sdat = sum(dat);
% % dat = dat./repmat(sdat,a,1);
% mdat = max(dat);
% mmdat = mdat./100;
% dat = dat./repmat(mmdat,a,1);
%%%
%%% tfidf dat
% [a,b] = size(dat);
% sdat = sum(dat);
% dat = dat.*log(repmat(repmat(a,1,b)./(sdat+1),a,1));
%%%
%%% diff dat
% dat = diff(dat);
% szzs = (diff(szzs));
% dname = dname(2:end);
%%%
kname = kname(ff1);
%% experiment
windowsize = 240;
lamb = 1;
mlag = 1;
k = 3; % number of AR predictor
for i = 1:k
    w{end+1} = ['AR',num2str(i)];
end
nfea = size(dat,2); % number of manuly selected feature
cntfea = zeros(nfea+k,1);
ac_fe_1 = zeros(mlag,1); ac_fe_2 = zeros(mlag,1); ac_ar_1 = zeros(mlag,1); ac_ar_2 = zeros(mlag,1);
pred_fe = [];
true_val = [];
pred_ar = [];
for lagsize = 1:mlag
    for i = k:(a-windowsize-1-lagsize)
        % for i = 1:43
        train_fe_x = dat(i:i+windowsize,:);
        train_ar_x = log((szzs(i:i+windowsize)));
        [train_ar_x,test_ar_x] = F_arfe(szzs,i,i+windowsize,k,1);
        train_y = log((szzs(i+lagsize:i+windowsize+lagsize)));
        [train_fe_x,f] = F_fesel(train_fe_x,train_y,nfea,1);
        test_fe_x = dat(i+windowsize+1,:);
        test_fe_x = test_fe_x(:,f);
        test_y = log((szzs(i+windowsize+1+lagsize)));
        test_yp = log((szzs(i+windowsize+lagsize)));
        idx(i-k+1,lagsize) = i+windowsize+1+lagsize;
        train_fe_X = [train_fe_x,train_ar_x,ones(length(train_ar_x),1)];
        %         train_fe_X = [train_fe_x,ones(length(train_ar_x),1)];
        train_ar_X = [train_ar_x,ones(length(train_ar_x),1)];
        test_fe_X = [test_fe_x,test_ar_x,1];
        %         test_fe_X = [test_fe_x,1];
        test_ar_X = [test_ar_x,1];
        %%%% linear regression begin
        %                 beta_fe = mvregress(train_fe_X,train_y);
        %                 prefix = 'linearreg';
        beta_ar = mvregress(train_ar_X,train_y);
        %%%% linear regression end
        
        %%%% stepwise begin
        %                 [beta_fe,se,pval,inmodel,stats,nextstep,history] = ...
        %                     stepwisefit(train_fe_X(:,1:end-1),train_y,'display','off','penter',0.2);
        %                 beta_fe = [beta_fe.*inmodel';stats.intercept];
        %                 prefix = ['stepwise-p-enter-2'];
        %         [beta_ar,se,pval,inmodel,stats,nextstep,history] = ...
        %             stepwisefit(train_ar_X(:,1:end-1),train_y,'display','off','penter',0.2);
        %         beta_ar = [beta_ar.*inmodel';stats.intercept];
        %%%% stepwise end
        
        %%%% lasso begin
        %         [B_fe,info_fe] = lasso(train_fe_X(:,1:end-1),train_y,'Lambda',lamb./1000);
        [B_fe,info_fe] = lasso(train_fe_X(:,1:end-1),train_y,'CV',10,'alpha',0.5);
        idx = info_fe.IndexMinMSE;
        cntfea(find(B_fe(:,idx) > eps)) = cntfea(find(B_fe(:,idx) > eps)) + 1;
        beta_fe = [B_fe(:,idx);info_fe.Intercept(idx)];
        prefix = ['lasso-lamb-alpha-0',num2str(lamb)];
        %         [B_ar,info_ar] = lasso(train_ar_X(:,1:end-1),train_y,'Lambda',lamb./1000);
        %         beta_ar = [B_ar(:,1);info_ar.Intercept];
        %%%% lasso end
        
        %%%%% DT begin
        %-log(szzs(i+lagsize-1:i+windowsize+lagsize-1)
%         t = classregtree(train_fe_X(:,1:end-1),double((train_y)),...
%             'names',w,'prune','on');
%         prefix = ['DT-',num2str(lamb)];
%         pred_fe_test(i-k+1,lagsize) = eval(t,test_fe_X(1:end-1));
%         pred_fe_train = eval(t,train_fe_X(1:end-1));
        
        %%%%% DT end
        
        %%%% ridge begin
        %         beta_fe = ridge(train_y,train_fe_X(:,1:end-1),lamb./1000,0);
        %         t = beta_fe(1);
        %         beta_fe = [beta_fe(2:end);t];
        %         prefix = ['ridge-lamb-',num2str(lamb)];
        %         beta_ar = ridge(train_y,train_ar_X(:,1:end-1),1,0);
        %         t = beta_ar(1);
        %         beta_ar = [beta_ar(2:end);t];
        %%%% ridge end
        
        pred_fe_train = train_fe_X*beta_fe;
        pred_ar_train = train_ar_X*beta_ar;
        pred_fe_test(i-k+1,lagsize) = test_fe_X*beta_fe;
        pred_ar_test(i-k+1,lagsize) = test_ar_X*beta_ar;
        
        MODEL{i-k+1,lagsize} = beta_fe;
        mae_train_fe(i-k+1,lagsize) = mean(abs(exp(train_y)-exp(pred_fe_train)));
        mae_train_ar(i-k+1,lagsize) = mean(abs(exp(train_y)-exp(pred_ar_train)));
        mae_test_fe(i-k+1,lagsize) = abs(exp(test_y)-exp(pred_fe_test(i-k+1,lagsize)));
        mae_test_ar(i-k+1,lagsize) = abs(exp(test_y)-exp(pred_ar_test(i-k+1,lagsize)));
        
        
        %         mae_train_fe(i-k+1,lagsize) = mean(abs((train_y)-(pred_fe_train)));
        %         mae_train_ar(i-k+1,lagsize) = mean(abs((train_y)-(pred_ar_train)));
        %         mae_test_fe(i-k+1,lagsize) = abs((test_y)-(pred_fe_test(i-k+1,lagsize)));
        %         mae_test_ar(i-k+1,lagsize) = abs((test_y)-(pred_ar_test(i-k+1,lagsize)));
        
        ac_fe_1(lagsize) = ac_fe_1(lagsize) + ((test_y - test_yp)*(pred_fe_test(i-k+1,lagsize)-test_yp) > 0);
        ac_fe_2(lagsize) = ac_fe_2(lagsize) + 1;
        ac_ar_1(lagsize) = ac_ar_1(lagsize) + ((test_y - test_yp)*(pred_ar_test(i-k+1,lagsize)-test_yp) > 0);
        ac_ar_2(lagsize) = ac_ar_2(lagsize) + 1;
        
        pred_fe(end+1,lagsize) = pred_fe_test(i-k+1,lagsize)-test_yp;
        true_val(end+1,lagsize) = test_y - test_yp;
        pred_ar(end+1,lagsize) = pred_ar_test(i-k+1,lagsize)-test_yp;
        nfe(i-k+1,lagsize) = length(find(abs(beta_fe) > eps));
    end
    %     mae_ar(stepsize./10,lagsize) = mean(mae_test_ar(:,lagsize))
end
save(prefix)
%%
subplot(2,2,4)
mae = [mae_train_fe,mae_train_ar,mae_test_fe,mae_test_ar];
boxplot(mae)
ylim([0,200])
title('trainerr_{trend} | trainerr_{base}  | testerr_{trend}  |  testerr_{base}','fontsize',8)
hold on
plot([15.5,15.5],ylim)
plot([30.5,30.5],ylim)
plot([45.5,45.5],ylim)
plot(xlim,[50,50])
hold off
set(gca,'XTick',[])

subplot(2,2,1)
boxplot(mae_test_fe./mae_test_ar)
hold on
plot(xlim,[1 1]);
ylim([0,2])
title([prefix,': Dist. of MAE_{trend} / MAE_{base}'])
ylabel('MAE_{trend} / MAE_{base}')
xlabel('lag days')
hold off
for i = 1:mlag
    [h,p(i,1)] = ttest2(mae_test_fe(:,i),mae_test_ar(:,i),[],'left');
end
subplot(2,2,2)
plot(p,'LineWidth',2,'marker','s')
title('P value for MAE_{trend} < MAE_{base}','fontsize',10)
set(gca,'xtick',1:20);
xlabel('lag days','fontsize',14)
ylabel('p value','fontsize',14)

subplot(2,2,3)


ac_fe = ac_fe_1./ac_fe_2;
ac_ar = ac_ar_1./ac_ar_2;
plot(ac_fe,'b','LineWidth',2,'marker','o');
hold on
plot(ac_ar,'r','LineWidth',2,'marker','s');
hold on
plot(xlim,[0.5 0.5]);
hold off
legend({'Trend','Base'})
title('涨跌预测准确度')

ylabel('Accuracy','fontsize',14)
xlabel('lag of days','fontsize',14)

print('-dpng','-r300',[prefix,'_',num2str(nfea),'_',num2str(k),'_',num2str(windowsize)]);
%%
figure(2)
subplot(2,1,1)
boxplot(nfe)
title(['Dist. of selected feature numbers (totally ',num2str(length(test_fe_x)),')'],'fontsize',10);
xlabel('lag days')
ylabel(['#feature by ',prefix]);

subplot(2,1,2)
[s,ord] = sort(cntfea,'descend');
bar(cntfea(ord(1:15)));
xlim([0,16])
set(gca,'Xtick',1:15)
set(gca,'Xticklabel',w(ord(1:15)),'fontsize',8)

print('-dpng','-r300',[prefix,'_',num2str(nfea),'_',num2str(k),'_',num2str(windowsize),'_fea']);
%%
save([prefix,'_',num2str(nfea),'_',num2str(k),'_',num2str(windowsize)]);
