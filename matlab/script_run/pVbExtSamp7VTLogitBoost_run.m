%% 
name = 'letter4k';
dir_data = 'E:\Users\sp\data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pVbExtSamp7VTLogitBoost',name);
%%

num_Tpre = 10000;
T = 10000;
cv  = {0.1};
cJ = {20};
cns = {1};
crs = {0.2};
crf = {0.2};
crc = {1.1};
%%
h = batch_pVbExtSamp7VTLogitBoost();
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