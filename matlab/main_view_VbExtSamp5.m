%% data name
% name = 'poker100k';
% name = 'pendigits';
% name = 'optdigits';
% name = 'M-Basic';
% name = 'M-Image';
% name = 'isolet';
% name = 'mnist';
% name = 'timit.mfcc.winSz11';
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

name = 'M-Noise3';
algoname1 = 'pVbExtSamp5VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T3000_v1.0e-001_J50_ns1_rs9.00e-001_rf3.10e-002_rc1e+000.mat';

% name = 'letter';
% algoname1 = 'pVbExtSamp5VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T3000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';
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
figure;
title nr-s;
plot(nr_wts,'marker','x','linewidth',4);
grid on;
%%
figure;
title nr-c;
plot(nr_wtc,'marker','o','linewidth',4,'color','r');
grid on;
%% plot error
figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','marker','.');
% plot(it2,err_it2, 'color','b','marker','.');
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
grid on;
%% plot grad
figure('name',name);  title grad; hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
grid on;