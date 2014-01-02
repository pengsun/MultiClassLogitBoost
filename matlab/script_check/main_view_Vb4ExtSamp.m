%% config
name = 'optdigits';
algoname1 = 'pVbExtSamp12VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T5000_v1.0e-01_J20_ns1_wrs9.50e-01_rs1.10e+00_rf2.00e-01_wrc1.10e+00_rc2.10e-01.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset_mat';

it_ind = [];
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
clear tmp;
%% info
tmp_fn = fullfile(dir_data, [name,'.mat']);
tmp = load(tmp_fn);
ntr = size(tmp.Xtr,2);
nclass = max(tmp.Ytr)+1;
clear tmp;
% ntr = 50000;
% nclass = 10;
%% print examples
for i = 1 : numel(tree_node_sc)
  sc = tree_node_sc{i};
  nr(i) = sc(1); %#ok<SAGROW>
end

navg = mean(nr);
fprintf(name);fprintf('\n\n');
fprintf('avg examples = %d\n',navg);
fprintf('ntr = %d\n', ntr);
fprintf('rs = %d\n\n', navg/ntr);
%% print last result
fprintf('last result:\n');
fprintf('%s: %d @ %d\n', algoname1, err_it1(end), it1(end));
fprintf('\n');
%% print best result
[tmp,tmp_ind] = min(err_it1);
fprintf('best result:\n');
fprintf('%s: %d @ %d\n', algoname1, tmp, it1(tmp_ind));
fprintf('\n');
%% print number of operations(split searching)
nop = 0;
nop_orig = 0;
for i = 1 : numel(tree_node_sc)
  nop = nop + sum(tree_node_cc{i}.*tree_node_sc{i});
  nop_orig = nop_orig + nclass*sum(tree_node_sc{i});
end
fprintf('number of operations (split searching):\n');
fprintf('nop = %d\n',nop);
fprintf('nop without class sampling = %d\n', nop_orig);
fprintf('ratio = %d\n\n', nop/nop_orig);
%% plot examples
% figure('name',name); 
% title('#examples');
% hold on;
% plot(nr_wts,'marker','x','linewidth',2);
% hold off;
% grid on;
%% plot classes - MinMax
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
%% plot classes: average of average
for i = 1 : numel(tree_node_cc)
  avg_cc(i) = mean( tree_node_cc{i} ); %#ok<SAGROW>
end
figure('name',name); 
title( sprintf('average #classes (K = %d)',nclass) );
hold on;
plot(avg_cc, 'marker','.','linewidth',2,'color','b');
hold off;
grid on;
%% plot class distribution
for i = 1 : numel(it_ind)
  ii = it_ind(i);
  cc = tree_node_cc{ii};
  
  figure;
  hold on;
  title( sprintf('iter %d of %d',ii,numel(it1)) );
  hist(cc, (1:nclass) );
  set(gca, 'xlim', [0.5, nclass+0.5]);
  set(gca, 'ylim', [0,numel(cc)+1]);
  xlabel('#classes'); ylabel('#nodes');
  hold off;
end
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