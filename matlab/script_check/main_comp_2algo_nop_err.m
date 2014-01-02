%% config
name = 'isolet';

algoname1 = 'pCoSamp2VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v1.0e-01_J20_ns1_rf5.14e-02_rb5.00e-02_wrb9.50e-01.mat';

algoname2 = 'pVbExtSamp14VTLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T5000_v1.0e-01_J20_ns1_wrs9.50e-01_rs5.00e-02_rf5.14e-02_wrc1.10e+00_rc1.10e+00.mat';

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
losscls1 = tmp.LossCls;
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
losscls2 = tmp.LossCls;
clear tmp;

%% print info
fprintf('%s:\n',name);
fprintf('#1: %s\n',ffn1);
fprintf('#2: %s\n',ffn2);
fprintf('\n');
%% print number of operations(split searching)
nop1 = 0;
for i = 1 : numel(tree_node_sc1)
  nop1 = nop1 + sum(tree_node_cc1{i}.*tree_node_sc1{i});
end
nop2 = 0;
for i = 1 : numel(tree_node_sc2)
  nop2 = nop2 + sum(tree_node_cc2{i}.*tree_node_sc2{i});
end

fprintf('number of operations (split searching):\n');
fprintf('nop1 = %d\n',nop1);
fprintf('nop2 = %d\n',nop2);
fprintf('\n');
%% print best error
fprintf('best error:\n');
[tmp1,tmp1_ind] = min(err_it1);
fprintf('#1: %d @ %d\n',  tmp1, it1(tmp1_ind));
[tmp2,tmp2_ind] = min(err_it2);
fprintf('#2: %d @ %d\n', tmp2, it2(tmp2_ind));
fprintf('\n');
%% print last error
fprintf('last error:\n');
fprintf('#1: %d @ %d\n', err_it1(end), it1(end));
fprintf('#2: %d @ %d\n', err_it2(end), it2(end));
fprintf('\n');
%% print raw training time
fprintf('raw training time:\n');
fprintf('#1: %d\n', time_tr1);
fprintf('#2: %d\n', time_tr2);
fprintf('\n');
%% plot nop v.s. error
for i = 1 : numel(tree_node_sc1)
  vnop1(i) = sum(tree_node_cc1{i}.*tree_node_sc1{i});
end
for i = 1 : numel(tree_node_sc2)
  vnop2(i) = sum(tree_node_cc2{i}.*tree_node_sc2{i});
end

figure('name','nop v.s. error'); 
title( sprintf('nop v.s. error') );
hold on;
plot(cumsum(vnop1(it1)),err_it1(it1), 'marker','x','linewidth',1,'color','m');
plot(cumsum(vnop2(it2)),err_it2(it2), 'marker','.','linewidth',1,'color','b');
set(gca,'xscale','log');
hold off;
h = legend(ffn1,ffn2);
set(h,'Interpreter','none');
grid on;
%% plot nop v.s. gradient
figure('name', 'nop v.s. grad'); 
title( sprintf('nop v.s. grad') );
hold on;
plot(cumsum(vnop1(it1)),abs_grad1(it1), 'marker','x','linewidth',1,'color','m');
plot(cumsum(vnop2(it2)),abs_grad2(it2), 'marker','.','linewidth',1,'color','b');
% set(gca,'xscale','log','yscale','log');
% set(gca,'xscale','log');
set(gca,'yscale','log');
hold off;
h = legend(ffn1,ffn2);
set(h,'Interpreter','none');
grid on;
%% plot nop v.s. loss
figure('name','nop v.s.loss'); 
title( sprintf('nop v.s.loss') );
hold on;
plot(cumsum(vnop1(it1)), sum( losscls1(:,it1) ),...
  'marker','x','linewidth',1,'color','m');
plot(cumsum(vnop2(it2)), sum( losscls2(:,it2) ),...
  'marker','.','linewidth',1,'color','b');
% set(gca,'xscale','log','yscale','log');
% set(gca,'xscale','log');
set(gca,'yscale','log');
hold off;
h = legend(ffn1,ffn2);
set(h,'Interpreter','none');
grid on;
%% plot iter v.s. nop
% figure('name',name); 
% title( sprintf('iter v.s. nop') );
% hold on;
% plot(it1,vnop1(it1), 'marker','x','linewidth',1,'color','m');
% plot(it2,vnop2(it2), 'marker','.','linewidth',1,'color','r');
% plot(it3,vnop3(it3), 'marker','*','linewidth',1,'color','b');
% % set(gca,'xscale','log','yscale','log');
% % set(gca,'xscale','log');
% % set(gca,'yscale','log');
% hold off;
% h = legend(ffn1,ffn2,ffn3);
% set(h,'Interpreter','none');
% grid on;

%% plot iter v.s. error
figure('name','iter v.s. error'); 
title( sprintf('iter v.s. error') );
hold on;
plot(it1,err_it1(it1), 'marker','x','linewidth',1,'color','m');
plot(it2,err_it2(it2), 'marker','.','linewidth',1,'color','b');
set(gca,'xscale','log');
% set(gca,'yscale','log');
hold off;
h = legend(ffn1,ffn2);
set(h,'Interpreter','none');
grid on;
%% plot iter v.s. gradient
figure('name','iter v.s. grad'); 
title( sprintf('iter v.s. grad') );
hold on;
plot(it1,abs_grad1(it1), 'marker','x','linewidth',1,'color','m');
plot(it2,abs_grad2(it2), 'marker','.','linewidth',1,'color','b');
% set(gca,'xscale','log','yscale','log');
% set(gca,'xscale','log');
set(gca,'yscale','log');
hold off;
h = legend(ffn1,ffn2);
set(h,'Interpreter','none');
grid on;