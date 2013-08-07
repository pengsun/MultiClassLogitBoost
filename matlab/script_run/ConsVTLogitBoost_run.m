%% 
% name = 'mnist';
% fn_data = fullfile('E:\Users\sp\dataset_mat\',[name,'.mat']);
% dir_rst = fullfile('.\rst\VTLogitBoost',name);

name = 'optdigits';
fn_data = fullfile('.\dataset\',[name,'.mat']);
dir_rst = fullfile('.\rst\ConsVTLogitBoost',name);
%%
num_Tpre = 2000;
T = 10000;
cv  = {0.1};
cJ = {20};
cns = {1};
%%
h = batch_ConsVTLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
run_all_param(h, fn_data, dir_rst);
clear h;