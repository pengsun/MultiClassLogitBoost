%% config
name = 'letter4k';
algoname1 = 'pVbExtSamp11VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T10000_v1.0e-001_J20_ns1_rs5.10e-001_rf2.00e-001_rc9.00e-001.mat';

% dir_data = 'E:\Users\sp\data\dataset_mat';
dir_data = 'D:\data\dataset_mat';

it = [1,2, ];
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
%% class distrib
% for
