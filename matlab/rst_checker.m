classdef rst_checker
  %RST_CHECKER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function print(obj, dir_rst,name_algo, name_data)    
      % read
      dir_mat = fullfile(dir_rst,name_algo,name_data,'*.mat');
      files = dir(dir_mat);
      for i = 1 : numel(files)
        fn = files(i).name;
        
        % parameter v
        idx = strfind(fn,'_');
        str = fn( idx(1)+2 : idx(2)-1 );
        v(i) = str2double(str);
        
        % err
        fn_full = fullfile(dir_rst,name_algo,name_data, fn);
        st = load( fn_full ,'err');
        err(i) = st.err; %#ok<*SAGROW>
      end
      
      % sorting
      [sv,ii] = sort(v,'ascend');
      serr = err(ii);
      
      % print
      fprintf('%s\n',name_algo);
      fprintf('%s\n',name_data);
      for i = 1 : numel(sv)
        fprintf( 'v = %0.2f, err = %d\n', sv(i), serr(i) );
      end
      fprintf('\n');
    end % print
    
    function print2(obj, dir_rst,name_algo, name_data)    
      % read
      dir_mat = fullfile(dir_rst,name_algo,name_data,'*.mat');
      files = dir(dir_mat);
      [Jv,err,max_it] = deal([]);
      for i = 1 : numel(files)
        fn = files(i).name;
        
        % parameter J and v
        idx = strfind(fn,'_');
        str = fn( idx(2)+2 : idx(3)-1 );
        Jv(i,1) = str2double(str);
        str = fn( idx(1)+2 : idx(2)-1 );
        Jv(i,2) = str2double(str);
        
        % err
        fn_full = fullfile(dir_rst,name_algo,name_data, fn);
        st = load( fn_full ,'err','it');
        err(i) = st.err; %#ok<*SAGROW>
        
        % max iter
        max_it(i) = st.it(end);
      end
      
      % print
      fprintf('%s\n',name_algo);
      fprintf('%s\n',name_data);
      if (isempty(Jv)), fprintf('null\n\n'); return; end
      
      % sorting
      [sJv,ii] = sortrows(Jv,[-1,-2]);
      serr = err(ii);
      smax_it = max_it(ii);
      

      for i = 1 : size(sJv,1)
        fprintf( 'J = %d, v = %0.2f,  max_it = %d, err = %d\n',...
          sJv(i,1),sJv(i,2), smax_it(i), serr(i) );
      end
      fprintf('\n');
    end % print2
    
    function print3(obj, dir_rst,name_algo, name_data)    
      % read
      dir_mat = fullfile(dir_rst,name_algo,name_data,'*.mat');
      files = dir(dir_mat);
      [Jv, last_err,last_it, best_err,best_it, tr_time] = deal([]);
      for i = 1 : numel(files)
        fn = files(i).name;
        
        % parameter J and v
        idx = strfind(fn,'_');
        str = fn( idx(2)+2 : idx(3)-1 );
        Jv(i,1) = str2double(str);
        str = fn( idx(1)+2 : idx(2)-1 );
        Jv(i,2) = str2double(str);
        
        % err
        fn_full = fullfile(dir_rst,name_algo,name_data, fn);
        st = load( fn_full);
        % last
        last_err(i) = st.err; %#ok<*SAGROW>
        last_it(i) = st.it(end);
        % best
        [best_err(i),best_it(i)] = min(st.err_it);
        % training time
        tr_time(i) = st.time_tr;
        
        clear st;
      end
      
      % print
      fprintf('%s\n',name_algo);
      fprintf('%s\n',name_data);
      if (isempty(Jv)), fprintf('null\n\n'); return; end
      
      % sorting
      [sJv,ii] = sortrows(Jv,[-1,-2]);
      slast_err = last_err(ii);
      slast_it = last_it(ii);
      sbest_err = best_err(ii);
      sbest_it = best_it(ii);
      str_time = tr_time(ii);
      
      % formatted output
      stmpl = ...
        ['J = %d, v = %0.2f, it = %d, err = %d, it*= %d, err* = %d, ',...
        'trtime = %3.2fdays\n'];
      for i = 1 : size(sJv,1)
        fprintf(stmpl,...
          sJv(i,1),sJv(i,2),...
          slast_it(i), slast_err(i), sbest_it(i), sbest_err(i),...
          str_time(i)/3600/24);
      end
      fprintf('\n');
    end % print3
  end
  
end

