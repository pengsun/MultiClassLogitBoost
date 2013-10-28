%% 
name = 'optdigits';
% dir_data = 'E:\Users\sp\data\dataset2_mat';
dir_data = 'D:\Data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pAvgSampVTLogitBoost_rssmall',name);
%%
num_Tpre = 5000;
T = 5000;
cv  = {0.1};
cJ = {2};
cns = {1};
%%% sample
crs = {0.01};
cwrs = {1.1};
%%% feature
crf = {0.2};
%%% class
crc = {1.1};
cwrc = {1.1};
%%%
cTdot = {1,5,10};
%%
h = batch_pAvgSampVTLogitBoost();
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
% Tdot
h.cTdot = cTdot;
run_all_param(h, fn_data, dir_rst);
clear h;