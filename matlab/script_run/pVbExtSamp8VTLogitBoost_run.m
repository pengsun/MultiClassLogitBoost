%% 
name = 'mnist';
dir_data = 'E:\Users\sp\data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pVbExtSamp8VTLogitBoost',name);
%%
num_Tpre = 10000;
T = 5000;
cv  = {0.1};
cJ = {70};
cns = {1};
crs = {0.2351};
crf = {0.031};
crc = {0.6};
%%
h = batch_pVbExtSamp8VTLogitBoost();
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