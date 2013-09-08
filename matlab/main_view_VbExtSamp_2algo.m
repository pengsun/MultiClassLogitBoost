%% config
name = 'zipcode';
%% config algo
algoname1 = 'pVbExtSamp6VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T10000_v1.0e-01_J20_ns1_rs2.00e-01_rf5.00e-02_rc1.10e+00.mat';

algoname2 = 'pVbExtSamp7VTLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T10000_v1.0e-01_J20_ns1_rs2.00e-01_rf5.00e-02_rc1.10e+00.mat';
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
nr_wts1 = tmp.nr_wts;
nr_wtc1 = tmp.nr_wtc;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
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
%% examples ratio
figure('name',name); 
title('#examples');
hold on;
plot(nr_wts1,'marker','.','linewidth',2,'color','b');
plot(nr_wts2,'marker','o','linewidth',2,'color','r');
hold off;
h = legend([algoname1,' ',fn1],...
    [algoname2,' ',fn2]);
set(h,'Interpreter','none');
grid on;
%%
% figure;
% title('nrc');
% hold on;
% plot(nr_wtc,'marker','o','linewidth',4,'color','r');
% hold off;
% grid on;
%% plot error
figure('name',name); title error; 
hold on;
plot(it1,err_it1, 'color','b','lineWidth', 2, 'marker','.');
plot(it2,err_it2, 'color','r','lineWidth', 2, 'marker','o');
hold off;
h = legend([algoname1,' ',fn1],...
    [algoname2,' ',fn2]);
set(h,'Interpreter','none');
grid on;
%% plot grad
figure('name',name);  
title('||grad||_1'); 
hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','b','marker','.');
plot(it2, log10( eps + abs_grad2(it2) ), 'color','r','marker','o');
h = legend([algoname1,' ',fn1],...
    [algoname2,' ',fn2]);
set(h,'Interpreter','none');
hold off;
grid on;