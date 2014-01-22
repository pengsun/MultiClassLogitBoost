%% config
name = 'optdigits05';
% name = 'zipcode38';
% name = 'pendigits49';
% name = 'mnist10k05';
algoname1 = 'pAOSOLogitBoostV2Vb';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T1000_v1.0e-001_J2_ns1_wrs1.10e+000_rs1.10e+000_rf2.00e-001.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
% dir_data = 'D:\data\dataset_mat';

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
    pprev_node = pprev(ix);
    h_node = sum( 4.*p_node.*(1-p_node) );
    hprev_node = sum( 4.*pprev_node.*(1-pprev_node) );
    rr(i, j) = h_node/(hprev_node + eps);
  end
end

figure;
plot(rr);
% set(gca,'ylim',[0.7,1.3]);
title('Node Hessian ratio');
grid on;
%% weight (Hessian)
% for i = 1 : length(pp)
%   tmp = pp{i};
%   p = tmp(1,:);
%   h(i) = sum( 4.*p.*(1-p) );
% end
% ratio = [];
% for j = 2 : length(h)
%   ratio(end+1) = h(j)/(h(j-1)+eps);
% end
% figure;
% plot(h);
% title('Hessian');
% set(gca,'yscale','log');
% grid on;
% 
% figure;
% plot(ratio);
% title('Hessian ratio');
% grid on;
%% node weight
% for i = 2 : length(pp)
%   tmp = pp{i};
%   p = tmp(1,:);
%   tmp = pp{i-1};
%   pprev = tmp(1,:);
%   
%   % each node
%   uid = unique( tree_si_to_leaf{i} );
%   for j = 1 : length(uid)
%     ix = (tree_si_to_leaf{i}==uid(j));
%     p_node = p(ix);
%     pprev_node = pprev(ix);
%     h_node = sum( 4.*p_node.*(1-p_node) );
%     hprev_node = sum( 4.*pprev_node.*(1-pprev_node) );
%     rr(i, j) = h_node/(hprev_node + eps);
%   end
% end
% 
% figure;
% plot(rr);
% % set(gca,'ylim',[0.7,1.3]);
% title('Node Hessian ratio');
% grid on;
%% plot loss
% L = sum(loss_cls);
% figure;
% plot(it1,L(it1)); 
% set(gca,'yscale','log'); grid on;
% title('Loss');
%% plot error
% figure;
% plot(it1, err_it1(it1));
% title('testing error');
% grid on;
%% print error
% fprintf('%s:\n',fn1);
% fprintf('err = %d\n\n',err_it1(end));
