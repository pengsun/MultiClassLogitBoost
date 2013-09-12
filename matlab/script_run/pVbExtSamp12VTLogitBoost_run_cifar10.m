%% 
name = 'cifar-10';
dir_data = 'E:\Users\sp\data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pVbExtSamp12VTLogitBoost',name);
%%
num_Tpre = 3000;
T = 3000;
cv  = {0.1};
cJ = {70};
cns = {1};
%%% sample
cwrs = {0.9};
crs = {1.1};
%%% feature
crf = {0.018};
%%% class
cwrc = {1.1};
crc = {0.51};
%%
h = batch_pVbExtSamp12VTLogitBoost();
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
% class
h.cwrc = cwrc;
h.crc = crc;
run_all_param(h, fn_data, dir_rst);
clear h;