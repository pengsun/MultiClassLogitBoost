%% 
name = 'easy1hard2';
% dir_data = 'D:\Users\sp\data\dataset_mat';
dir_data = 'D:\Data\dataset2_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pCoSampVTLogitBoost',name);
%%
num_Tpre = 500;
T = 500;
cv  = {0.1};
cJ = {2};
cns = {1};
%%% feature
crf = {1.1};
%%% budget
crb = {0.15};
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