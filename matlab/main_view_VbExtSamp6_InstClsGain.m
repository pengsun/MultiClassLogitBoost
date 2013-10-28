%% config
name = 'optdigits';
algoname1 = 'pAvgSampVTLogitBoost_rssmall';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v1.0e-001_J20_ns1_Tdot3_wrs1.10e+000_rs1.00e-002_rf2.00e-001_wrc1.10e+000_rc1.10e+000.mat';
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

tree_node_cc= tmp.tree_node_cc;
tree_node_sc = tmp.tree_node_sc;
tree_node_all_sc = tmp.tree_node_all_sc;
tree_leaf_gain = tmp.tree_leaf_gain;
tree_leaf_allsample_gain = tmp.tree_leaf_allsample_gain;
clear tmp;
%% examples & class & gain
for i = 1 : numel(tree_node_sc)
  sc(i) = tree_node_sc{i}(1);
  all_sc(i) = tree_node_all_sc{i}(1);
end
for i = 1 : numel(tree_node_cc)
  cc(i) = tree_node_cc{i}(1);
end
for i = 1 : numel(tree_leaf_gain)
  gain(i) = sum(tree_leaf_gain{i}) ./ sc(i);
  allsample_gain(i) = sum(tree_leaf_allsample_gain{i}) ./ all_sc(i);
end
%% class Max Min
% for i = 1 : numel(tree_node_cc)
%   Mc(i) = max( tree_node_cc{i} );
%   mc(i) = min( tree_node_cc{i} );
% end
% figure;
% plot(it1, mc, it1, Mc, 'marker','x');
% legend('minCls','MAXCls');
% grid on;
%% class root node
% for i = 1 : numel(tree_node_cc)
%   cc(i) = tree_node_cc{i}(1);
% end
% figure;
% plot(it1, cc, 'marker','x');
% title('Cls Count, root node');
% grid on;

%% print examples & class
navg = mean(sc);
cavg = mean(cc);
fprintf(name);fprintf('\n');
fprintf('avg examples = %d\n',navg);
%% plot avarage gain
figure('name',name); title('average gain'); hold on;
plot(it1,gain(it1), 'color','r','lineWidth', 2, 'marker','+');
plot(it1,allsample_gain(it1), 'color','b','lineWidth', 2, 'marker','O');
grid on;
set(gca,'yscale','log');
legend('gain','allsample gain');
%% plot error
figure('name',name); title error; hold on;
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
grid on;
%% plot grad
figure('name',name);  
title('||grad||_1'); 
hold on;
plot(it1, abs_grad1(it1) , 'color','r','marker','.');
set(gca,'yscale','log');
hold off;
grid on;