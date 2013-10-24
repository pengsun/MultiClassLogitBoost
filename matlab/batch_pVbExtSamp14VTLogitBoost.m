classdef batch_pVbExtSamp14VTLogitBoost < batch_vb6_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp14VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp14VTLogitBoost';
    end
    
  end % methods
  
end %

