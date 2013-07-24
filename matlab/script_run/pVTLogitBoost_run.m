%% 
name = 'M-Image';
fn_data = fullfile('E:\Users\sp\dataset_mat',[name,'.mat']);
dir_rst = fullfile('E:\Users\sp\code_work\gitlab\MulticlassLogitBoost\matlab\rst\pVTLogitBoost',name);

% name = 'pendigits';
% fn_data = fullfile('.\dataset',[name,'.mat']);
% dir_rst = fullfile('.\rst\pVTLogitBoost',name);
%%
num_Tpre = 2000;
T = 10000;
cv  = {0.4};
cJ = {50};
cns = {1};
%%
h = batch_pVTLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
run_all_param(h, fn_data, dir_rst);
clear h;