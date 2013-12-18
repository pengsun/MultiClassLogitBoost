%% data name
name = 'S';
%% algo name
algoname1 = 'pAOSOLogitBoostV2';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v5.0e-002_J2_ns1_wrs1.10e+000_rs1.10e+000_rf1.10e+000.mat';

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

%% plot error
figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','marker','.');
% plot(it2,err_it2, 'color','b','marker','.');
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
h = legend(...
    [algoname1,'-',fn1]);
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
fprintf('%s: %d @ %d\n', algoname1, err1best, it1(it1best) );
%% print last result
fprintf('last result:\n');
fprintf('%s: %d @ %d\n', algoname1, err_it1(end), it1(end));
%% print training time
fprintf('training time:\n');
fprintf('%s: %d \n', algoname1, time_tr1);
%% plot grad
figure('name',name);  title grad; hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
% plot(it1, abs_grad1(it1), 'color','r','marker','o');
% plot(it2, abs_grad2(it2), 'color','b','marker','.');
h = legend(...
    [algoname1,'-',fn1]);
set(h, 'Interpreter','none'); 
grid on;  hold off;

% tune the appearence
% ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/3);