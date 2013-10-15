%% 
name = 'isolet';
% dir_data = 'D:\Users\sp\data\dataset_mat';
dir_data = 'D:\Data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pCoSampVTLogitBoost_wrb0.95',name);
%%
num_Tpre = 5000;
T = 5000;
cv  = {0.1};
cJ = {20};
cns = {1};
%%% feature
crf = {0.0514};
%%% budget
crb = {1.1};
cwrb = {0.95};
%%
h = batch_pCoSampVTLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
% feature
h.crf = crf;
% budget
h.crb = crb;
h.cwrb = cwrb;
run_all_param(h, fn_data, dir_rst);
clear h;