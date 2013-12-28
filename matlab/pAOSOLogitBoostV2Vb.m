classdef pAOSOLogitBoostV2Vb
  % Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    ptr;
  end
  
  methods
    function obj = train(obj,X,Y, varargin)
      [var_cat_mask,T,J,v, node_size,...
          rs,rf,wrs] = parse_input(varargin{:});
      if (isempty(var_cat_mask))
        nvar = size(X,1);
        var_cat_mask = uint8( zeros(nvar,1) );
      end
      obj.ptr = pAOSOLogitBoostV2Vb_mex('train',...
        X,Y,var_cat_mask,...
        T, J, v,...
        node_size,...
        rs,rf,wrs);
    end
    
    function [NumIter, TrLoss,F,P,tree,...
        GradCls,LossCls] = get (obj)
      [NumIter, TrLoss, F,P, GradCls,LossCls] =...
        pAOSOLogitBoostV2Vb_mex('get',obj.ptr);
      tree = 0;
    end    
    
    function tree_node_cc = get_cc(obj)
      NumIter = pAOSOLogitBoostV2Vb_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_node_cc{i} = ...
          pAOSOLogitBoostV2Vb_mex('get_cc',obj.ptr, i); %#ok<AGROW>
      end
    end
    
    function tree_node_sc = get_sc(obj)
      NumIter = pAOSOLogitBoostV2Vb_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_node_sc{i} = ...
          pAOSOLogitBoostV2Vb_mex('get_sc',obj.ptr, i); %#ok<AGROW>
      end
    end    
    
    function tree_is_leaf = get_is_leaf(obj)
      NumIter = pAOSOLogitBoostV2Vb_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_is_leaf{i} = ...
          pAOSOLogitBoostV2Vb_mex('get_is_leaf',obj.ptr, i); %#ok<AGROW>
      end % for i
    end % get_is_leaf    

    function tree_si_to_leaf = get_si_to_leaf(obj)
      NumIter = pAOSOLogitBoostV2Vb_mex('get',obj.ptr);
      for i = 1 : NumIter
        tree_si_to_leaf{i} = ...
          pAOSOLogitBoostV2Vb_mex('get_si_to_leaf',obj.ptr, i); %#ok<AGROW>
      end % for i
    end % get_is_leaf    
    
    function Y = predict(obj, X, T)
      if (nargin==2) 
        T = pAOSOLogitBoostV2Vb_mex('get',obj.ptr);
      end
      Y = pAOSOLogitBoostV2Vb_mex('predict',obj.ptr, X, T);
    end
    
    function delete(obj)
      pAOSOLogitBoostV2Vb_mex('delete',obj.ptr);
    end
  end % method
  
end % 

function [var_cat_mask, T, J, v, node_size,...
    rs,rf,wrs] = parse_input(varargin)
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
      case 'wrs'
        wrs = varargin{i+1};  
      otherwise
        error('Unknow properties');
    end % switch
  end % for
end

