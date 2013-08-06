%% 
% name = 'optdigits';
% fn_data = fullfile('E:\Users\sp\dataset_mat',[name,'.mat']);
% dir_rst = fullfile('E:\Users\sp\code_work\gitlab\MulticlassLogitBoost\matlab\rst\pSampVTLogitBoost',name);

name = 'optdigits';
fn_data = fullfile('.\dataset',[name,'.mat']);
dir_rst = fullfile('.\rst\pSampVTLogitBoost',name);

% name = 'M-Image';
% fn_data = fullfile('.\dataset',[name,'.mat']);
% dir_rst = fullfile('.\rst\pSampVTLogitBoost',name);
%%
num_Tpre = 2000;
T = 7;
cv  = {0.1};
cJ = {20};
cns = {1};
crs = {0.8};
crf = {0.8};
crc = {0.6};
%%
h = batch_pSampVTLogitBoost();
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