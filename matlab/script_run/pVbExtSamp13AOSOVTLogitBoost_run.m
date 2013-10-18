%% 
name = 'cifar-10';
dir_data = 'E:\Users\sp\data\dataset_mat';
% dir_data = 'D:\Data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pVbExtSamp13AOSOVTLogitBoost_temp_wrs0.95',name);
%%
num_Tpre = 10000;
T = 10000;
cv  = {0.1};
cJ = {120};
cns = {1};
%%% sample
crs = {1.1};
cwrs = {0.95};
%%% feature
crf = {0.0118};
%%% class
crc = {0.21};
cwrc = {1.1};
%%
h = batch_pVbExtSamp13AOSOVTLogitBoost();
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