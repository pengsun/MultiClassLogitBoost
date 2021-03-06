%%
settings
% settings_server
%%
name = 'AOSOLogitBoost_mex';
fn = {...
  [name,'.cpp'],...
  'utilCPP.cpp',...
  fullfile(dir_src,'MLData.cpp'),...
  fullfile(dir_src,'AOSOLogitBoost.cpp')...
  };
%% Debug
% outnamed = sprintf('-output %s', [name,'d']);
% cmdd = sprintf('mex -g -v %s %s %s %s',...
%   fn{:});
% cmdd = sprintf('%s %s',...
%   cmdd, opt_cmdd);
% eval(cmdd);
% copyfile([name,'.',mexext], './../private/');
%% Release
outname = sprintf('-output %s', name);
cmd = sprintf('mex -O %s %s %s %s',...
  fn{:});
cmd = sprintf('%s %s',...
  cmd, opt_cmd);
eval(cmd);
copyfile([name,'.',mexext], './../private/');