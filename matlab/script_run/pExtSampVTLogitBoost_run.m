%% 
name = 'mnist';
dir_data = 'D:\data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pExtSampVTLogitBoost',name);
%%
num_Tpre = 3000;
T = 5000;
cv  = {0.1};
cJ = {50};
cns = {1};
crs = {1};
crf = {0.003};
crc = {1};
%%
h = batch_pExtSampVTLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
h.crs = crs;
h.crf = crf;
h.crc = crc;
run_all_param(h, fn_data, dir_rst);
clear h;