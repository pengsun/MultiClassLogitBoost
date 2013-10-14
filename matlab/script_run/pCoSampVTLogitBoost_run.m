%% 
name = 'c300f1n50';
dir_data = 'D:\Users\sp\data\dataset2_mat';
% dir_data = 'D:\Data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pCoSampVTLogitBoost',name);
%%
num_Tpre = 600;
T = 600;
cv  = {0.1};
cJ = {8};
cns = {1};
%%% feature
crf = {1.1};
%%% budget
crb = {.01};
cwrb = {1.1};
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