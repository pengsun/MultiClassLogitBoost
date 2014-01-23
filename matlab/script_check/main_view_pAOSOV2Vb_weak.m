%% config
% name = 'optdigits05';
% name = 'zipcode38';
% name = 'pendigits49';
name = 'mnist10k05';
algoname1 = 'pAOSOLogitBoostV2Vb';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T1000_v1.0e-001_J2_ns1_wrs1.10e+000_rs1.10e+000_rf3.10e-002.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset2_mat';

% it_ind = [];
% it_ind = [1000, 2000,3000,4000,4700];
% it_ind = round( linspace(1200,2000,12) );
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
% F1 = tmp.F;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;

tree_node_cc = tmp.tree_node_cc;
tree_node_sc = tmp.tree_node_sc;
grad_cls = tmp.GradCls;
loss_cls = tmp.LossCls;
pp = tmp.pp;
tree_si_to_leaf = tmp.tree_si_to_leaf;
clear tmp;
%% load dataset
tmp = load( fullfile(dir_data, [name,'.mat']) );
r = double( (tmp.Ytr==0) );
clear tmp;
%%
T = length(pp);
for i = 2 : T
  tmp = pp{i-1}; % last iteration!
  p = tmp(1,:);
  
  % each node
  uid = unique( tree_si_to_leaf{i} );
  for j = 1 : length(uid)
    ix = (tree_si_to_leaf{i}==uid(j));
    p_node = p(ix);
    r_node = r(ix);
    g_node = 2*(p_node - r_node);
    wr(i,j) = abs(sum(g_node))/sum(abs(g_node));
  end
end

% the firs row is omitted
wr(1,:) = [];

%% plot weak
figure;
plot(wr);
% set(gca,'ylim',[0.7,1.3]);
title('Node Weak');
grid on;
%% print weak
fprintf('min weak = %d\n', min(min(wr)));
fprintf('max weak = %d\n', max(max(wr)));
fprintf('median weak = %d\n', median(median(wr)));