%% data name
% name = 'poker100k';
% name = 'pendigits';
% name = 'optdigits';
% name = 'M-Basic';
name = 'M-Image';
% name = 'isolet';
%% algo name
algoname1 = 'VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T100000_v1.0e-001_J50_ns1.mat';
% fn1 = 'T700_v1_J20_ns1.mat';

algoname2 = 'VTDropoutLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T10000_v1_J50_ns1.mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
% abs_grad1 = tmp.abs_grad;
abs_grad1 = tmp.loss_tr;
% F1 = tmp.F;
num_it1 = tmp.num_it;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
% F2 = tmp.F;
num_it2 = tmp.num_it;

%% plot error
figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','marker','.');
% plot(it2,err_it2, 'color','b','marker','.');
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
plot(it2,err_it2, 'color','b','lineWidth', 2, 'marker','o');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2]);
set(h,'Interpreter','none'); 
grid on; hold off; 

% tune the appearence
ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/2);
% set(gca, 'ylim', [0,500]);
%% print best result
fprintf('-------------\n');
fprintf('dataset: %s\n\n', name);
fprintf('best result:\n');
[err1best,it1best] = min(err_it1);
[err2best,it2best] = min(err_it2);
fprintf('%s: %d @ %d\n', algoname1, err1best, it1(it1best) );
fprintf('%s: %d @ %d\n', algoname2, err2best, it2(it2best));

%% print last result
fprintf('last result:\n');
fprintf('%s: %d @ %d\n', algoname1, err_it1(end), it1(end));
fprintf('%s: %d @ %d\n', algoname2, err_it2(end), it2(end));
%% plot grad
figure('name',name);  title grad; hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
plot(it2, log10( eps + abs_grad2(it2) ), 'color','b','marker','o');
% plot(it1, abs_grad1(it1), 'color','r','marker','o');
% plot(it2, abs_grad2(it2), 'color','b','marker','.');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2]);
set(h, 'Interpreter','none'); 
grid on;  hold off;

% tune the appearence
% ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/3);