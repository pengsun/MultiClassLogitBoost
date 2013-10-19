classdef pVbExtSamp12SkimVTLogitBoost
  % Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    ptr;
  end
  
  methods
    function obj = train(obj,X,Y, varargin)
      [var_cat_mask,T,J,v, node_size,...
          rs,rf,rc, wrs,wrc] = parse_input(varargin{:});
      if (isempty(var_cat_mask))
        nvar = size(X,1);
        var_cat_mask = uint8( zeros(nvar,1) );
      end
      obj.ptr = pVbExtSamp12SkimVTLogitBoost_mex('train',...
        X,Y,var_cat_mask,...
        T, J, v,...
        node_size,...
        rs,rf,rc, wrs,wrc);
    end
    
    function [NumIter, TrLoss] = get (obj)
      [NumIter, TrLoss] =...
        pVbExtSamp12SkimVTLogitBoost_mex('get',obj.ptr);
    end
    
    function tree_node_cc = get_cc(obj)
      NumIter = pVbExtSamp12SkimVTLogitBoost_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_node_cc{i} = ...
          pVbExtSamp12SkimVTLogitBoost_mex('get_cc',obj.ptr, i); %#ok<AGROW>
      end
    end
    
    function tree_node_sc = get_sc(obj)
      NumIter = pVbExtSamp12SkimVTLogitBoost_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_node_sc{i} = ...
          pVbExtSamp12SkimVTLogitBoost_mex('get_sc',obj.ptr, i); %#ok<AGROW>
      end
    end    
    
    function tree_is_leaf = get_is_leaf(obj)
      NumIter = pVbExtSamp12SkimVTLogitBoost_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_is_leaf{i} = ...
          pVbExtSamp12SkimVTLogitBoost_mex('get_is_leaf',obj.ptr, i); %#ok<AGROW>
      end % for i
    end % get_is_leaf
    
    function Y = predict(obj, X, T)
      if (nargin==2) 
        T = pVbExtSamp12SkimVTLogitBoost_mex('get',obj.ptr);
      end
      Y = pVbExtSamp12SkimVTLogitBoost_mex('predict',obj.ptr, X, T);
    end
    
    function delete(obj)
      pVbExtSamp12SkimVTLogitBoost_mex('delete',obj.ptr);
    end
  end % method
  
end % 

function [var_cat_mask, T, J, v, node_size,...
    rs,rf,rc, wrs,wrc] = parse_input(varargin)
  var_cat_mask = [];
  T = 5;
  J = 8;
  v = 1;
  node_size = 5;
  for i = 1 : 2 : nargin
    name = varargin{i};
    switch name
      case 'var_cat_mask'
        var_cat_mask = varargin{i+1};
      case 'T'
        T = varargin{i+1};
      case 'v'
        v = varargin{i+1};
      case 'J'
        J = varargin{i+1};
      case 'node_size'
        node_size  = varargin{i+1};
      case 'rs'
        rs = varargin{i+1};
      case 'rf'
        rf = varargin{i+1};
      case 'rc'
        rc = varargin{i+1};
      case 'wrs'
        wrs = varargin{i+1};        
      case 'wrc'
        wrc = varargin{i+1};
      otherwise
        error('Unknow properties');
    end % switch
  end % for
end

