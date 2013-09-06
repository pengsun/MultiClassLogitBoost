%% config
% name = 'optdigits';
% algoname = 'pVbExtSamp5VTLogitBoost';
% dir_root = fullfile('.\rst',algoname);
% fn1 = 'T10000_v1.0e-01_J20_ns1_rs6.00e-01_rf2.00e-01_rc1e+00.mat';
% fn2 = 'T1000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';

% name = 'pendigits';
% algoname = 'pVbExtSamp5VTLogitBoost';
% dir_root = fullfile('.\rst',algoname);
% fn1 = 'T10000_v1.0e-01_J20_ns1_rs6.00e-01_rf2.00e-01_rc1e+00.mat';
% fn2 = 'T1000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';

% name = 'zipcode';
% algoname = 'pVbExtSamp5VTLogitBoost';
% dir_root = fullfile('.\rst',algoname);
% fn1 = 'T10000_v1.0e-01_J20_ns1_rs6.00e-01_rf5.00e-02_rc1e+00.mat';
% fn2 = 'T3000_v1.0e-001_J20_ns1_rs9.00e-001_rf5.00e-002_rc1e+000.mat';

name = 'letter4k';
algoname = 'pVbExtSamp5VTLogitBoost';
dir_root = fullfile('.\rst',algoname);
fn1 = 'T10000_v1.0e-01_J20_ns1_rs6.00e-01_rf2.00e-01_rc1e+00.mat';
fn2 = 'T3000_v1.0e-001_J20_ns1_rs9.00e-001_rf2.00e-001_rc1e+000.mat';
%% load
ffn1 = fullfile(dir_root,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
% abs_grad1 = tmp.loss_tr;
% F1 = tmp.F;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;
nr_wts1 = tmp.nr_wts;
nr_wtc1 = tmp.nr_wtc;
clear tmp;

ffn2 = fullfile(dir_root,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
% abs_grad1 = tmp.loss_tr;
% F1 = tmp.F;
num_it2 = tmp.num_it;
time_tr2 = tmp.time_tr;
nr_wts2 = tmp.nr_wts;
nr_wtc2 = tmp.nr_wtc;
clear tmp;
%%
figure('name',name); 
title('#examples');
hold on;
plot(nr_wts1,'marker','.','linewidth',2,'color','b');
plot(nr_wts2,'marker','o','linewidth',2,'color','r');
hold off;
h = legend(fn1,fn2);
set(h,'Interpreter','none');
grid on;
%%
% figure;
% title('nrc');
% hold on;
% plot(nr_wtc,'marker','o','linewidth',4,'color','r');
% hold off;
% grid on;
% plot error
figure('name',name); title error; 
hold on;
plot(it1,err_it1, 'color','b','lineWidth', 2, 'marker','.');
plot(it2,err_it2, 'color','r','lineWidth', 2, 'marker','o');
hold off;
h = legend(fn1,fn2);
set(h,'Interpreter','none');
grid on;
%% plot grad
figure('name',name);  
title('||grad||_1'); 
hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','b','marker','.');
plot(it2, log10( eps + abs_grad2(it2) ), 'color','r','marker','o');
h = legend(fn1,fn2);
set(h,'Interpreter','none');
hold off;
grid on;