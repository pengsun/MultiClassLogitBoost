%% config
name = 'timit.mfcc.winSz11';

algoname1 = 'pCoSampVTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T1000_v1.0e-001_J70_ns1_rf2.00e-002_rb3.86e-001_wrb1.10e+000.mat';

algoname2 = 'pVbExtSamp12VTLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T1000_v1.0e-001_J70_ns1_wrs9.00e-001_rs1.10e+000_rf2.00e-002_wrc1.10e+000_rc5.00e-001.mat';

algoname3 = 'pVbExtSamp12VTLogitBoost';
dir_root3 = fullfile('.\rst',algoname3);
fn3 = 'T1000_v1.0e-001_J70_ns1_wrs9.00e-001_rs1.10e+000_rf2.00e-002_wrc1.10e+000_rc1.00e-001.mat';

dir_data = 'D:\Users\sp\data\dataset3_mat';
% dir_data = 'D:\data\dataset_mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;
tree_node_cc1 = tmp.tree_node_cc;
tree_node_sc1 = tmp.tree_node_sc;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
num_it2 = tmp.num_it;
time_tr2 = tmp.time_tr;
tree_node_cc2 = tmp.tree_node_cc;
tree_node_sc2 = tmp.tree_node_sc;
clear tmp;

ffn3 = fullfile(dir_root3,name,fn3);
tmp = load(ffn3);
it3 = tmp.it;
err_it3 = tmp.err_it;
abs_grad3 = tmp.abs_grad;
num_it3 = tmp.num_it;
time_tr3 = tmp.time_tr;
tree_node_cc3 = tmp.tree_node_cc;
tree_node_sc3 = tmp.tree_node_sc;
clear tmp;
%% info
tmp_fn = fullfile(dir_data, [name,'.mat']);
tmp = load(tmp_fn);
ntr = size(tmp.Xtr,2);
nclass = max(tmp.Ytr)+1;
clear tmp;

% ntr = 50000;
% nclass = 10;
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
%% print best error
fprintf('best error:\n');
[tmp1,tmp1_ind] = min(err_it1);
fprintf('%s: %d @ %d\n', ffn1, tmp1, it1(tmp1_ind));
[tmp2,tmp2_ind] = min(err_it2);
fprintf('%s: %d @ %d\n', ffn2, tmp2, it2(tmp2_ind));
[tmp3,tmp3_ind] = min(err_it3);
fprintf('%s: %d @ %d\n', ffn3, tmp3, it3(tmp3_ind));
fprintf('\n');
%% print last error
fprintf('last error:\n');
fprintf('%s: %d @ %d\n', ffn1, err_it1(end), it1(end));
fprintf('%s: %d @ %d\n', ffn2, err_it2(end), it2(end));
fprintf('%s: %d @ %d\n', ffn3, err_it3(end), it3(end));
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
h = legend(ffn1,ffn2,ffn3);
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
h = legend(ffn1,ffn2,ffn3);
set(h,'Interpreter','none');
grid on;