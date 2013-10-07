%%
name = 'optdigits';
dir_data = 'E:\Users\sp\data\dataset_mat';
fn = fullfile(dir_data, [name,'.mat']);
load(fn);
%%
cls = 6;
ind = find(Ytr==cls);
ii = ind(1);
%%
xx = Xtr(:,ii);
img = reshape(xx,8,8);
img = img';
img = double(img);
img = img./16;
%%
imtool(img,'initialMagnification',4800);
% img = 