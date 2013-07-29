%% 
name = 'mnist';
fn_data = fullfile('D:\Users\sp\data\dataset_mat',[name,'.mat']);
dir_rst = fullfile('D:\Users\sp\code_work\gitlab\MulticlassLogitBoost\matlab\rst\pSampVTLogitBoost',name);

% name = 'pendigits';
% fn_data = fullfile('.\dataset',[name,'.mat']);
% dir_rst = fullfile('.\rst\pSampVTLogitBoost',name);

% name = 'optdigits';
% fn_data = fullfile('.\dataset',[name,'.mat']);
% dir_rst = fullfile('.\rst\pSampVTLogitBoost',name);
%%
num_Tpre = 2000;
T = 10000;
cv  = {0.2};
cJ = {50};
cns = {1};
crs = {1};
crf = {1};
crc = {1};
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