%% config
% name = 'optdigits05';
name = 'pendigits49';
% name = 'zipcode38';
% name = 'mnist10k05';
algoname1 = 'pAOSOGradBoostVb';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T1000_v1.0e-01_J8_ns1_wrs1.10e+00_rs1.10e+00_rf4.00e-01.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset2_mat';

% it_ind = [];
% it_ind = [1000, 2000,3000,4000,4700];
% it_ind = round( linspace(1200,2000,12) );
%% load result
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
%% Gradient

for i = 1 : length(pp)
  tmp = pp{i};
  p = tmp(1,:);
  gmax(i) = max( abs(2*(p-r)) );
  gmin(i) = min( abs(2*(p-r)) );
  
  N = length(p);
  L(i) = sum( -r.*log(p) - (1-r).*log(1-p) )./N;
  gabs(i) = sum( abs(2*(p-r)) )./N;
end

figure;
plot(1:length(gmax), [gmin(:),gmax(:)]);
title('Gradient Min Max');
legend('min','max');
grid on;


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
%% Loss
L1 = sum(loss_cls)./N;
figure;
plot(it1,L(it1)); 
set(gca,'yscale','log'); grid on;
title('Loss');
grid on;
%% gradient & loss
rgL = gabs(:)./(L(:));
% tmp = abs_grad1(:)/N;
% rgL = tmp(:)./L(:);
figure;
plot(1:length(rgL), rgL);
grid on;
legend('g/L ratio');

fprintf('min rgL: %d\n\n', min(rgL));
%% plot error
% figure;
% plot(it1, err_it1(it1));
% title('testing error');
% grid on;
%% print error
% fprintf('%s:\n',fn1);
% fprintf('err = %d\n\n',err_it1(end));