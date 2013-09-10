%% config
name = 'letter';
algoname1 = 'pVbExtSamp10VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T10000_v1.0e-001_J20_ns1_rs6.00e-001_rf2.00e-001_rc9.00e-001.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset_mat';
%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
% F1 = tmp.F;
num_it1 = tmp.num_it;
time_tr1 = tmp.time_tr;

nr_wts = tmp.nr_wts;
nr_wtc = tmp.nr_wtc;
tree_node_cc = tmp.tree_node_cc;
clear tmp;
%% info
tmp_fn = fullfile(dir_data, [name,'.mat']);
tmp = load(tmp_fn);
ntr = size(tmp.Xtr,2);
nclass = max(tmp.Ytr)+1;
clear tmp;
%% print examples
navg = mean(nr_wts);
fprintf(name);fprintf('\n\n');
fprintf('avg examples = %d\n',navg);
fprintf('ntr = %d\n', ntr);
fprintf('rs = %d\n\n', navg/ntr);
%% print last result
fprintf('last result:\n');
fprintf('%s: %d @ %d\n', algoname1, err_it1(end), it1(end));
fprintf('\n');
%% plot examples
% figure('name',name); 
% title('#examples');
% hold on;
% plot(nr_wts,'marker','x','linewidth',2);
% hold off;
% grid on;
%% classes - MinMax
for i = 1 : numel(tree_node_cc)
  Mcc(i) = max( tree_node_cc{i} ); %#ok<SAGROW>
  mcc(i) = min( tree_node_cc{i} ); %#ok<SAGROW>
end
figure('name',name); 
title( sprintf('#classes (K = %d)',nclass) );
hold on;
plot(Mcc,'marker','s','linewidth',2,'color','r','markersize',8);
plot(mcc,'marker','.','linewidth',2,'color','b');
hold off;
legend('Max cc', 'Min cc');
grid on;
%% classes: average of average
for i = 1 : numel(tree_node_cc)
  avg_cc(i) = mean( tree_node_cc{i} ); %#ok<SAGROW>
end
figure('name',name); 
title( sprintf('average #classes (K = %d)',nclass) );
hold on;
plot(avg_cc, 'marker','.','linewidth',2,'color','b');
hold off;
grid on;
%% plot error
figure('name',name); 
title('error'); 
hold on;
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','.');
grid on;
%% plot grad
figure('name',name);  
title('||grad||_1'); 
hold on;
plot(it1, log10( eps + abs_grad1(it1) ), 'color','r','marker','.');
hold off;
grid on;