classdef pExtSampVTLogitBoost
  % Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    ptr;
  end
  
  methods
    function obj = train(obj,X,Y, varargin)
      [var_cat_mask,T,J,v, node_size, rs,rf,rc] = parse_input(varargin{:});
      if (isempty(var_cat_mask))
        nvar = size(X,1);
        var_cat_mask = uint8( zeros(nvar,1) );
      end
      obj.ptr = pExtSampVTLogitBoost_mex('train',...
        X,Y,var_cat_mask,...
        T, J, v,...
        node_size,...
        rs,rf,rc);
    end
    
    function [NumIter, TrLoss,F,P,tree] = get (obj)
      [NumIter, TrLoss, F,P] = pExtSampVTLogitBoost_mex('get',obj.ptr);
      tree = 0;
    end
    
    function Y = predict(obj, X, T)
      if (nargin==2) 
        T = pExtSampVTLogitBoost_mex('get',obj.ptr);
      end
      Y = pExtSampVTLogitBoost_mex('predict',obj.ptr, X, T);
    end
    
    function delete(obj)
      pExtSampVTLogitBoost_mex('delete',obj.ptr);
    end
  end % method
  
end % 

function [var_cat_mask, T, J, v, node_size, rs,rf,rc] = parse_input(varargin)
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
      otherwise
        error('Unknow properties');
    end % switch
  end % for
end

