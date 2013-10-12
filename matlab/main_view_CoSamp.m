%% config
name = 'letter4k';
algoname1 = 'pCoSampVTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v1.0e-001_J20_ns1_rf2.00e-001_rb1.00e-002_wrb1.10e+000.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset_mat';

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
clear tmp;
%% info
tmp_fn = fullfile(dir_data, [name,'.mat']);
tmp = load(tmp_fn);
ntr = size(tmp.Xtr,2);
nclass = max(tmp.Ytr)+1;
clear tmp;
% ntr = 50000;
% nclass = 10;
%% sample & class count
for i = 1 : numel(tree_node_cc)
  cc(i) = tree_node_cc{i}(1);
end
for i = 1 : numel(tree_node_sc)
  sc(i) = tree_node_sc{i}(1);
end
% figure('name',name);
% title('class count');
% hold on;
% hold off;
% grid on;
%% plot class loss
figure('name',name);  
title('class loss'); 
hold on;
plot(it1, loss_cls(:,it1));
set(gca,'yscale','log');
hold off;
grid on;
%% plot class loss ratio Max Min
temp = loss_cls(:,it1);
tt = sort(temp,'ascend');
gm = tt(1,:);
gM = tt(end-1,:);
% clear tt;
% gm = min(temp);
% gM = max(temp);
clear temp;

figure('name',name);  
title('class loss min max ratio'); 
hold on;
plot(it1, gm./(gM+eps), 'color','r', 'linewidth',4);
hold off;
grid on;
%% print class loss top-bottom
% % it_ind = round( linspace(1,it1(end),30) );
% it_ind = [1:10, round( linspace(11,it1(end),10) )];
% temp = loss_cls(:,it_ind);
% fprintf('Top-Bottom Loss classes:\n');
% for i = 1 : size(temp,2)
%   fprintf('iter %d: ',it_ind(i));
%   gi = temp(:,i);
%   [gs,ind] = sort(gi,'descend');
%   fprintf('(%d, %d, %d)  ',...
%     ind(1),ind(2),ind(3));
%   fprintf('(%d, %d, %d)\n',...
%     ind(end-2),ind(end-1),ind(end));
% end
% fprintf('\n');
% clear temp;
%% plot grad
% figure('name',name);  
% title('||grad||_1'); 
% hold on;
% plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
% hold off;
% grid on;
%% plot class grad
figure('name',name);  
title('class ||grad||_1'); 
hold on;
plot(it1, grad_cls(:,it1));
set(gca,'yscale','log');
hold off;
grid on;
%% plot class grad ratio Max Min
temp = grad_cls(:,it1);
tt = sort(temp,'ascend');
gm = tt(1,:);
gM = tt(end,:);
% clear tt;
% gm = min(temp);
% gM = max(temp);
clear temp;

figure('name',name);  
title('class ||grad||_1 min max ratio'); 
hold on;
plot(it1, gm./(gM+eps),'linewidth',4);
hold off;
grid on;
%% print class grad top-bottom
% % it_ind = round( linspace(1,it1(end),30) );
% it_ind = [1:10, round( linspace(11,it1(end),10) )];
% temp = grad_cls(:,it_ind);
% fprintf('Top-Bottom Grad classes:\n');
% for i = 1 : size(temp,2)
%   fprintf('iter %d: ',it_ind(i));
%   gi = temp(:,i);
%   [gs,ind] = sort(gi,'descend');
%   fprintf('(%d, %d, %d)  ',...
%     ind(1),ind(2),ind(3));
%   fprintf('(%d, %d, %d)\n',...
%     ind(end-2),ind(end-1),ind(end));
% end
% fprintf('\n');
% clear temp;
