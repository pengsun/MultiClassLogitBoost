%%
name = 'zipcode_eqcls';
dir_data = 'E:\Users\sp\data\dataset_mat';
fn = fullfile(dir_data, [name,'.mat']);
load(fn);
%%
cls = 5;
ind = find(Ytr==cls);
ii = ind(333);
%% 
fprintf('#class %d: %d \n\n',cls, numel(ind));
%%
xx = Xtr(:,ii);
img = reshape(xx,16,16);
img = img';
img = double(img);
% img = img./16;
%%
imtool(img,'initialMagnification',1000);
% img = 