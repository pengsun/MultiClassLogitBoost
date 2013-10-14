%% 
name = 'c300f1n50';
dir_data = 'D:\Users\sp\data\dataset2_mat';
% dir_data = 'D:\Data\dataset_mat';
fn_data = fullfile(dir_data, [name,'.mat']);
dir_rst = fullfile('.\',...
  'rst\pVbExtSamp13VTLogitBoost_Comp_CoSamp',name);
%%
num_Tpre = 600;
T = 600;
cv  = {0.01};
cJ = {8};
cns = {1};
%%% sample
crs = {0.1};
cwrs = {1.1};
%%% feature
crf = {1.2};
%%% class
crc = {0.1};
cwrc = {1.1};
%%
h = batch_pVbExtSamp13VTLogitBoost();
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