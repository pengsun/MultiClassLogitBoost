%% 
name = 'timit.mfcc.winSz11';
dir_data = 'E:\Users\sp\data\dataset3_mat';
% dir_data = 'D:\Data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pVbExtSamp12VTLogitBoost',name);
%%
num_Tpre = 1000;
T = 1000;
cv  = {0.1};
cJ = {70};
cns = {1};
%%% sample
cwrs = {1.1};
crs = {0.01};
%%% feature
crf = {0.02};
%%% class
cwrc = {1.1};
crc = {1.1};
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