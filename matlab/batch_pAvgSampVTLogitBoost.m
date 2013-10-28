classdef batch_pAvgSampVTLogitBoost < batch_avgsampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAvgSampVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pAvgSampVTLogitBoost';
    end
    
  end % methods
  
end %

