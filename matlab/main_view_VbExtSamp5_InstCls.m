%% config
name = 'ocr.eng.94.fea';
algoname1 = 'pVbExtSamp12SkimVTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T1000_v1.0e-001_J50_ns1_wrs9.00e-001_rs1.10e+000_rf2.10e-002_wrc1.10e+000_rc1.10e+000.mat';
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

tree_node_cc = tmp.tree_node_cc;
tree_node_sc = tmp.tree_node_sc;
clear tmp;
%% class Max Min
for i = 1 : numel(tree_node_cc)
  Mc(i) = max( tree_node_cc{i} );
  mc(i) = min( tree_node_cc{i} );
end
figure;
plot(it1, mc, it1, Mc, 'marker','x');
legend('minCls','MAXCls');
grid on;
%% class root node
for i = 1 : numel(tree_node_cc)
  cc(i) = tree_node_cc{i}(1);
end
figure;
plot(it1, cc, 'marker','x');
title('Cls Count, root node');
grid on;
%% examples & class
for i = 1 : numel(tree_node_sc)
  sc(i) = tree_node_sc{i}(1);
end
for i = 1 : numel(tree_node_cc)
  cc(i) = tree_node_cc{i}(1);
end
% print examples & class
navg = mean(sc);
cavg = mean(cc);
fprintf(name);fprintf('\n');
fprintf('avg examples = %d\n',navg);
%% plot error
% figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
% grid on;
%% plot grad
% figure('name',name);  
% title('||grad||_1'); 
% hold on;
% plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
% hold off;
% grid on;