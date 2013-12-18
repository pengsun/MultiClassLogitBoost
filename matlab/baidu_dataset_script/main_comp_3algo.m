%% data name
% name = 'poker100k';
% name = 'pendigits';
% name = 'optdigits';
% name = 'M-Basic';
% name = 'isolet';
name = 'mnist';
%% algo name
algoname1 = 'pVbExtSamp9VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v1.0e-01_J70_ns1_rs2.35e-01_rf3.10e-02_rc6.00e-01.mat';

algoname2 = 'pVbExtSamp8VTLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T5000_v1.0e-01_J70_ns1_rs2.35e-01_rf3.10e-02_rc6.00e-01.mat';

algoname3 = 'pVbExtSamp6VTLogitBoost';
dir_root3 = fullfile('.\rst',algoname3);
fn3 = 'T3000_v1.0e-01_J70_ns1_rs2.35e-01_rf3.10e-02_rc1.10e+00.mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
% F1 = tmp.F;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
% F2 = tmp.F;
num_it2 = tmp.num_it;
time_tr2 = tmp.time_tr;

ffn3 = fullfile(dir_root3,name,fn3);
tmp = load(ffn3);
it3 = tmp.it;
err_it3 = tmp.err_it;
abs_grad3 = tmp.abs_grad;
% F3 = tmp.F;
num_it3 = tmp.num_it;
time_tr3 = tmp.time_tr;
%% plot error
figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','marker','.');
% plot(it2,err_it2, 'color','b','marker','.');
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
plot(it2,err_it2, 'color','b','lineWidth', 2, 'marker','o');
plot(it3,err_it3, 'color','c','lineWidth', 2, 'marker','^');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2],...
    [algoname3,'-',fn3]);
set(h,'Interpreter','none'); 
grid on; hold off; 

% tune the appearence
ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/2);
set(gca, 'ylim', [0,500]);
%% print best result
fprintf('-------------\n');
fprintf('dataset: %s\n\n', name);
fprintf('best result:\n');
[err1best,it1best] = min(err_it1);
[err2best,it2best] = min(err_it2);
[err3best,it3best] = min(err_it3);
fprintf('%s: %d @ %d\n', algoname1, err1best, it1(it1best) );
fprintf('%s: %d @ %d\n', algoname2, err2best, it2(it2best));
fprintf('%s: %d @ %d\n\n', algoname3, err3best, it3(it3best) );
%% print last result
fprintf('last result:\n');
fprintf('%s: %d @ %d\n', algoname1, err_it1(end), it1(end));
fprintf('%s: %d @ %d\n', algoname2, err_it2(end), it2(end));
fprintf('%s: %d @ %d\n\n', algoname3, err_it3(end), it3(end));
%% print training time
fprintf('training time:\n');
fprintf('%s: %d \n', algoname1, time_tr1);
fprintf('%s: %d \n', algoname2, time_tr2);
fprintf('%s: %d \n\n', algoname3, time_tr3);
%% plot grad
figure('name',name);  title grad; hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
plot(it2, log10( eps + abs_grad2(it2) ), 'color','b','marker','o');
plot(it3, log10( eps + abs_grad3(it3) ), 'color','c','marker','^');
% plot(it1, abs_grad1(it1), 'color','r','marker','o');
% plot(it2, abs_grad2(it2), 'color','b','marker','.');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2],...
    [algoname3,'-',fn3]);
set(h, 'Interpreter','none'); 
grid on;  hold off;

% tune the appearence
% ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/3);