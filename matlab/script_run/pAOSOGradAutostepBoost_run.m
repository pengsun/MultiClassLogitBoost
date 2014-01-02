%%
% name = 'zipcode';
% name = 'pendigits';
name = 'optdigits05';
% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\Data\dataset2_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pAOSOGradAutostepBoost',name);
%%
num_Tpre = 5000;
T = 5000;
cv  = {nan};
cJ = {8};
cns = {1};
%%% sample
crs = {1.1};
cwrs = {0.95};
%%% feature
crf = {0.2};
%%
h = batch_pAOSOGradAutostepBoost();
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