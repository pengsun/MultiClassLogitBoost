%% config
name = 'optdigits';
algoname1 = 'pVbExtSamp5VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T10000_v1.0e-01_J20_ns1_rs9.00e-01_rf2.00e-01_rc1.10e+00.mat';

dir_data = 'E:\Users\sp\data\dataset_mat';
% dir_data = 'D:\data\dataset_mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
% F1 = tmp.F;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;

nr_wts = tmp.nr_wts;
nr_wtc = tmp.nr_wtc;
clear tmp;
%% examples
figure('name',name); 
title('#examples');
hold on;
plot(nr_wts,'marker','x','linewidth',2);
hold off;
grid on;
%% print examples
navg = mean(nr_wts);
fprintf(name);fprintf('\n');
fprintf('avg examples = %d\n',navg);

tmp_fn = fullfile(dir_data, [name,'.mat']);
tmp = load(tmp_fn);
ntr = size(tmp.Xtr,2);
clear tmp;

fprintf('ntr = %d\n', ntr);
fprintf('rs = %d\n\n', navg/ntr);
%%
figure('name',name); 
title('#classes');
hold on;
plot(nr_wtc,'marker','o','linewidth',4,'color','r');
hold off;
grid on;
%% print last result
fprintf('last result:\n');
fprintf('%s: %d @ %d\n', algoname1, err_it1(end), it1(end));
fprintf('\n');
%% plot error
% figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
% grid on;
%% plot grad
figure('name',name);  
title('||grad||_1'); 
hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
hold off;
grid on;