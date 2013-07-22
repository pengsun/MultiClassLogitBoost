%% 
name = 'mnist';
fn_data = fullfile('D:\Users\sp\data\dataset_mat',[name,'.mat']);
dir_rst = fullfile(...
  'D:\Users\sp\code_work\gitlab\MulticlassLogitBoost\matlab\rst\VTDropoutLogitBoost',...
  name);
%%
num_Tpre = 2000;
T = 10000;
cv  = {1};
cJ = {50};
cns = {1};
%%
h = batch_VTDropoutLogitBoost();
h.num_Tpre = num_Tpre;
h.T = T;
h.cv = cv;
h.cJ = cJ;
h.cns = cns;
run_all_param(h, fn_data, dir_rst);
clear h;