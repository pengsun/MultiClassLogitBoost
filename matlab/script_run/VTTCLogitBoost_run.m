%% 
name = 'pendigits';
fn_data = fullfile('.\dataset',[name,'.mat']);
dir_rst = fullfile('.\rst\VTTCLogitBoost',name);
%%
num_Tpre = 2000;
T = 700;
clambda  = {1e-8};
cJ = {20};
cns = {1};
%%
h = batch_VTTCLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.clambda = clambda;
h.cJ = cJ;
h.cns = cns;
run_all_param(h, fn_data, dir_rst);
clear h;