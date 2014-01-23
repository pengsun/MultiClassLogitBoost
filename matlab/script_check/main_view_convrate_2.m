%% config
% name = 'optdigits05';
% name = 'zipcode38';
name = 'pendigits49';
% name = 'mnist10k05';

algoname1 = 'pAOSOMARTVb';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T1000_v1.0e-001_J2_ns1_wrs1.10e+000_rs1.10e+000_rf4.00e-001.mat';

algoname2 = 'pAOSOLogitBoostV2Vb';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T1000_v1.0e-001_J2_ns1_wrs1.10e+000_rs1.10e+000_rf4.00e-001.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset2_mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
loss_cls1 = tmp.LossCls;
pp1 = tmp.pp;
tree_si_to_leaf1 = tmp.tree_si_to_leaf;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
loss_cls2 = tmp.LossCls;
pp2 = tmp.pp;
tree_si_to_leaf2 = tmp.tree_si_to_leaf;
clear tmp;
%% load dataset
tmp = load( fullfile(dir_data, [name,'.mat']) );
r = double( (tmp.Ytr==0) );
clear tmp;
%% loss 
L1 = sum(loss_cls1);
L2 = sum(loss_cls2);
figure;
hold on;
plot(it1,L1(it1));
plot(it2,L2(it2),'color','r');
hold off;
legend('MART','Logit');
set(gca,'yscale','log'); grid on;
title('Loss');

%% plot error
figure;
hold on;
plot(it1, err_it1(it1));
plot(it2, err_it2(it2), 'color','r');
hold off;
legend('MART','Logit');
title('testing error');
grid on;
%% print error
% fprintf('%s:\n',fn1);
% fprintf('err = %d\n\n',err_it1(end));
%% weak
%%% first
T = length(pp1);
for i = 2 : T
  tmp = pp1{i-1}; % last iteration!
  p1 = tmp(1,:);
  
  % each node
  uid1 = unique( tree_si_to_leaf1{i} );
  for j = 1 : length(uid1)
    ix = (tree_si_to_leaf1{i}==uid1(j));
    p_node1 = p1(ix);
    r_node1 = r(ix);
    g_node1 = 2*(p_node1 - r_node1);
    wr1(i,j) = abs(sum(g_node1))/sum(abs(g_node1));
  end
end

% the firs row is omitted
wr1(1,:) = [];

%%% second
T = length(pp2);
for i = 2 : T
  tmp = pp2{i-1}; % last iteration!
  p2 = tmp(1,:);
  
  % each node
  uid2 = unique( tree_si_to_leaf2{i} );
  for j = 1 : length(uid2)
    ix = (tree_si_to_leaf2{i}==uid2(j));
    p_node2 = p2(ix);
    r_node2 = r(ix);
    g_node2 = 2*(p_node2 - r_node2);
    wr2(i,j) = abs(sum(g_node2))/sum(abs(g_node2));
  end
end

% the firs row is omitted
wr2(1,:) = [];
%% print weak
fprintf('MART: \n');
fprintf('min weak1 = %d\n', min(min(wr1)));
fprintf('max weak1 = %d\n', max(max(wr1)));
fprintf('median weak1 = %d\n', median(median(wr1)));
fprintf('\n');
fprintf('Logit: \n');
fprintf('min weak2 = %d\n', min(min(wr2)));
fprintf('max weak2 = %d\n', max(max(wr2)));
fprintf('median weak2 = %d\n', median(median(wr2)));