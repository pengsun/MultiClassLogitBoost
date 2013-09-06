%% config
% name = 'zipcode';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J20_ns1_rs9.00e-001_rf5.00e-002_rc1e+000.mat';

% name = 'optdigits';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T1000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';

% name = 'pendigits';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T1000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';

% name = 'isolet';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J20_ns1_rs9.00e-001_rf5.14e-002_rc1e+000.mat';

% name = 'letter4k';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';

% name = 'M-Basic';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J50_ns1_rs9.00e-001_rf3.10e-002_rc1e+000.mat';

% name = 'M-Image';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J50_ns1_rs9.00e-001_rf3.10e-002_rc1e+000.mat';

% name = 'M-Rand';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J50_ns1_rs9.00e-001_rf3.10e-002_rc1e+000.mat';

% name = 'M-Noise3';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J50_ns1_rs9.00e-001_rf3.10e-002_rc1e+000.mat';

% name = 'letter';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T10000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';

% name = 'mnist';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J70_ns1_rs9.00e-001_rf3.10e-002_rc1e+000.mat';

% name = 'cifar-10';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J120_ns1_rs9.50e-001_rf1.18e-002_rc1e+000.mat';
%% config
name = 'optdigits';
algoname1 = 'pVbExtSamp5VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T10000_v1.0e-01_J20_ns1_rs6.00e-01_rf2.00e-01_rc1e+00.mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
% abs_grad1 = tmp.loss_tr;
% F1 = tmp.F;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;

nr_wts = tmp.nr_wts;
nr_wtc = tmp.nr_wtc;
clear tmp;
%%
figure('name',name); 
title('#examples');
hold on;
plot(nr_wts,'marker','x','linewidth',2);
hold off;
grid on;
%%
% figure;
% title('nrc');
% hold on;
% plot(nr_wtc,'marker','o','linewidth',4,'color','r');
% hold off;
% grid on;
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