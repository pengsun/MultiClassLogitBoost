%% config
name = 'mnist';
algoname1 = 'pVbExtSamp11VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T10000_v1.0e-001_J70_ns1_rs4.00e-001_rf3.10e-002_rc9.00e-001.mat';
fn2 = 'T10000_v1.0e-001_J70_ns1_rs4.00e-001_rf3.10e-002_rc1.10e+000.mat';

dir_data = 'D:\Users\sp\data\dataset_mat';
% dir_data = 'D:\data\dataset_mat';
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
%% info
tmp_fn = fullfile(dir_data, [name,'.mat']);
tmp = load(tmp_fn);
ntr = size(tmp.Xtr,2);
nclass = max(tmp.Ytr)+1;
clear tmp;
%% number of operations(split searching)
nop1 = 0;
for i = 1 : numel(tree_node_sc1)
  nop1 = nop1 + sum(tree_node_cc1{i}.*tree_node_sc1{i});
end
nop2 = 0;
for i = 1 : numel(tree_node_sc2)
  nop2 = nop2 + sum(tree_node_cc2{i}.*tree_node_sc2{i});
end

fprintf('%s:\n',name);
fprintf('number of operations (split searching):\n');
fprintf('nop1 = %d\n',nop1);
fprintf('nop2 = %d\n',nop2);
fprintf('ratio = %d\n\n', nop1/nop2);
%% plot nop v.s. error
for i = 1 : numel(tree_node_sc1)
  vnop1(i) = sum(tree_node_cc1{i}.*tree_node_sc1{i});
end
for i = 1 : numel(tree_node_sc2)
  vnop2(i) = sum(tree_node_cc2{i}.*tree_node_sc2{i});
end

figure('name',name); 
title( sprintf('nop v.s. error') );
hold on;
plot(cumsum(vnop1),err_it1, 'marker','x','linewidth',2,'color','r');
plot(cumsum(vnop2),err_it2, 'marker','.','linewidth',2,'color','b');
hold off;
legend(fn1,fn2);
grid on;
