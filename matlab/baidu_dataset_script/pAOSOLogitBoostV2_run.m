%%
name = 'S';
dir_data = '.\';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pAOSOLogitBoostV2',name);
%%
num_Tpre = 500;
T = 500;
cv  = {0.05};
cJ = {2};
cns = {1};
%%% sample
crs = {0.95};
cwrs = {1.1};
%%% feature
crf = {1.1};
%%
h = batch_pAOSOLogitBoostV2();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
% sample
h.cwrs = cwrs;
h.crs = crs;
% feature
h.crf = crf;
run_all_param(h, fn_data, dir_rst);
clear h;
