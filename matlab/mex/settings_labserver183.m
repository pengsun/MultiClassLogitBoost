%% opencv
% root dir
dir_root = 'E:\Users\sp\WorkStudy\opencv243';
% dir
cvinc1 = fullfile(dir_root,'modules\ml\include');
cvinc2 = fullfile(dir_root,'\modules\core\include');
linkdird = fullfile(dir_root,'\lib\Debug');
linkdir = fullfile(dir_root, '\lib\Release');
%
lib1d = 'opencv_ml243d';
lib2d = 'opencv_core243d';
lib1 = 'opencv_ml243';
lib2 = 'opencv_core243';
%% source codes
dir_src = '../../src/';
%% options
tmpld = '-I%s -I%s -L%s -l%s';
opt_cmdd = sprintf(tmpld,...
  dir_src,...
  cvinc2,...
  linkdird,lib2d);
%% string template
tmpl = '-I%s -I%s -L%s -l%s';
opt_cmd = sprintf(tmpld,...
  dir_src,...
  cvinc2,...
  linkdir,lib2);
% %% option
% tmpld = '-I%s -I%s -L%s -l%s -L%s -l%s';
% opt_cmdd = sprintf(tmpld,...
%   cvinc1,cvinc2,...
%   linkdird, lib1d, linkdird,lib2d);
% %%
% tmpl = '-I%s -I%s -L%s -l%s -L%s -l%s';
% opt_cmd = sprintf(tmpld,...
%   cvinc1,cvinc2,...
%   linkdir, lib1, linkdir,lib2);