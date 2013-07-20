%% data name
% name = 'poker100k';
% name = 'pendigits';
name = 'optdigits';
% name = 'M-Basic';
% name = 'isolet';
%% algo name
% algoname1 = 'VTLogitBoost';
% dir_root1 = fullfile('.\rst',algoname1);
% fn1 = 'T300_v1.0e-001_J20_ns1.mat';
algoname1 = 'VTTCLogitBoost_TRound5';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T300_lambda0_J20_ns1.mat';
% fn1 = 'T300_lambda1.0e-004_J20_ns1.mat';

algoname2 = 'VTTCLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T300_lambda0_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-007_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-006_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-005_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-004_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-003_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-002_J20_ns1.mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
F1 = tmp.F;
num_it1 = tmp.num_it;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
F2 = tmp.F;
num_it2 = tmp.num_it;

%% error
figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','marker','.');
% plot(it2,err_it2, 'color','b','marker','.');
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','o');
plot(it2,err_it2, 'color','b','lineWidth', 2, 'marker','.');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2]);
set(h,'Interpreter','none'); 
grid on; hold off; 

% tune the appearence
ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/2);
set(gca, 'ylim', [0,500]);

%% grad
figure('name',name);  title grad; hold on;
% plot(it1, log10( abs_grad1(it1) ), 'color','r','marker','.');
% plot(it2, log10( abs_grad2(it2) ), 'color','b','marker','.');
plot(it1, abs_grad1(it1), 'color','r','marker','o');
plot(it2, abs_grad2(it2), 'color','b','marker','.');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2]);
set(h, 'Interpreter','none'); 
grid on;  hold off;

% tune the appearence
% ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/3);