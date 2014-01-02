%% config
name = 'cifar-10';
algoname1 = 'pVbExtSamp11VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v1.0e-01_J70_ns1_rs9.00e-01_rf1.80e-02_rc9.00e-01.mat';
fn2 = 'T5000_v1.0e-01_J70_ns1_rs9.00e-01_rf1.80e-02_rc6.00e-01.mat';
fn3 = 'T5000_v1.0e-01_J70_ns1_rs9.00e-01_rf1.80e-02_rc1.10e+00.mat';

% dir_data = 'D:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset_mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;
nr_wts1 = tmp.nr_wts;
nr_wtc1 = tmp.nr_wtc;
tree_node_cc1 = tmp.tree_node_cc;
tree_node_sc1 = tmp.tree_node_sc;
clear tmp;

ffn2 = fullfile(dir_root1,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
num_it2 = tmp.num_it;
time_tr2 = tmp.time_tr;
nr_wts2 = tmp.nr_wts;
nr_wtc2 = tmp.nr_wtc;
tree_node_cc2 = tmp.tree_node_cc;
tree_node_sc2 = tmp.tree_node_sc;
clear tmp;

ffn3 = fullfile(dir_root1,name,fn3);
tmp = load(ffn3);
it3 = tmp.it;
err_it3 = tmp.err_it;
abs_grad3 = tmp.abs_grad;
num_it3 = tmp.num_it;
time_tr3 = tmp.time_tr;
nr_wts3 = tmp.nr_wts;
nr_wtc3 = tmp.nr_wtc;
tree_node_cc3 = tmp.tree_node_cc;
tree_node_sc3 = tmp.tree_node_sc;
clear tmp;
%% info
% tmp_fn = fullfile(dir_data, [name,'.mat']);
% tmp = load(tmp_fn);
% ntr = size(tmp.Xtr,2);
% nclass = max(tmp.Ytr)+1;
% clear tmp;

ntr = 50000;
nclass = 10;
%% print number of operations(split searching)
nop1 = 0;
for i = 1 : numel(tree_node_sc1)
  nop1 = nop1 + sum(tree_node_cc1{i}.*tree_node_sc1{i});
end
nop2 = 0;
for i = 1 : numel(tree_node_sc2)
  nop2 = nop2 + sum(tree_node_cc2{i}.*tree_node_sc2{i});
end
nop3 = 0;
for i = 1 : numel(tree_node_sc3)
  nop3 = nop3 + sum(tree_node_cc3{i}.*tree_node_sc3{i});
end

fprintf('%s:\n',name);
fprintf('number of operations (split searching):\n');
fprintf('nop1 = %d\n',nop1);
fprintf('nop2 = %d\n',nop2);
fprintf('nop3 = %d\n',nop3);
fprintf('\n');
%% print last error
fprintf('last error:\n');
fprintf('%s: %d @ %d\n', fn1, err_it1(end), it1(end));
fprintf('%s: %d @ %d\n', fn2, err_it2(end), it2(end));
fprintf('%s: %d @ %d\n', fn3, err_it3(end), it3(end));
fprintf('\n');
%% plot nop v.s. error
for i = 1 : numel(tree_node_sc1)
  vnop1(i) = sum(tree_node_cc1{i}.*tree_node_sc1{i});
end
for i = 1 : numel(tree_node_sc2)
  vnop2(i) = sum(tree_node_cc2{i}.*tree_node_sc2{i});
end
for i = 1 : numel(tree_node_sc3)
  vnop3(i) = sum(tree_node_cc3{i}.*tree_node_sc3{i});
end

figure('name',name); 
title( sprintf('nop v.s. error') );
hold on;
plot(cumsum(vnop1(it1)),err_it1(it1), 'marker','x','linewidth',1,'color','m');
plot(cumsum(vnop2(it2)),err_it2(it2), 'marker','.','linewidth',1,'color','r');
plot(cumsum(vnop3(it3)),err_it3(it3), 'marker','*','linewidth',1,'color','b');
set(gca,'xscale','log');
hold off;
h = legend(fn1,fn2,fn3);
set(h,'Interpreter','none');
grid on;
%% plot nop v.s. gradient
figure('name',name); 
title( sprintf('nop v.s. grad') );
hold on;
plot(cumsum(vnop1(it1)),abs_grad1(it1), 'marker','x','linewidth',1,'color','m');
plot(cumsum(vnop2(it2)),abs_grad2(it2), 'marker','.','linewidth',1,'color','r');
plot(cumsum(vnop3(it3)),abs_grad3(it3), 'marker','*','linewidth',1,'color','b');
% set(gca,'xscale','log','yscale','log');
set(gca,'xscale','log');
hold off;
h = legend(fn1,fn2,fn3);
set(h,'Interpreter','none');
grid on;