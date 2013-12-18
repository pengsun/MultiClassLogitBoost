%% parameters
w = 240;
npos = 221;
num_Tpre = 500;
T = 500;
v  = 0.1;
J = 16;
ns = 1;
%%% sample
rs = 0.95;
wrs = 1.1;
%%% feature
rf = 1.1;
%% loada data
S = load( 'S_raw.mat', '-mat','X','Y','cat_mask' );
X = S.X;
Y = S.Y;
catmask = S.cat_mask;
clear S;
%% 
err = [];
fprintf('win size: %d\n',w);
for istart = 1 : npos
  iend = istart + w - 1;
  Xtr = X(:,istart:iend);
  Ytr = Y(istart:iend);
  Xte = X(:,iend+1);
  Yte = Y(iend+1);
  
  % train
  fprintf('pos %d\n', istart);
  hboost = pAOSOLogitBoostV2();
  hboost = train(hboost, Xtr,Ytr,...
    'var_cat_mask',catmask,...
    'T',T, 'v',v, 'J',J,...
    'node_size',ns,...
    'rs',rs, 'rf',rf,'wrs',wrs); 
  
  % test
  Tpre = num_Tpre;
  F = predict(hboost, Xte, Tpre);
  % error rate
  [~,yy] = max(F);
  yy = yy - 1;
  err(end+1) = sum(yy~=Yte);
  
  %
  clear hboost;
end

%%
err_rate = sum(err==1)/numel(err);
fprintf('err rate = %d\n',err_rate);
save(sprintf('rst_S_raw_J%d.mat', J),...
  'err');