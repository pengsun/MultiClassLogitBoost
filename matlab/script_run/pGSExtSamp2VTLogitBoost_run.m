%% 
name = 'pendigits';
dir_data = 'D:\data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pGSExtSamp2VTLogitBoost',name);
%%
num_Tpre = 3000;
T = 1000;
cv  = {0.1};
cJ = {8};
cns = {1};
crs = {0.9};
crf = {0.2};
crc = {1};
%%
h = batch_pGSExtSamp2VTLogitBoost();
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