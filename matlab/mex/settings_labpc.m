%% OpenCV 
% root dir
dir_root = 'D:\WorkStudy\OpenCV-2.2.0';
% dir
cvinc2 = fullfile(dir_root,'\modules\core\include');
linkdird = fullfile(dir_root,'\lib\Debug');
linkdir = fullfile(dir_root, '\lib\Release');
% 
lib2d = 'opencv_core220d';
lib2 = 'opencv_core220';
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