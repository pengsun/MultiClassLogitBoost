classdef batch_pCoSampVTLogitBoost < batch_CoSampboost_basic
  %batch_CoSampboost_basic Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pCoSampVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pCoSampVTLogitBoost';
    end
    
  end % methods
  
end %

