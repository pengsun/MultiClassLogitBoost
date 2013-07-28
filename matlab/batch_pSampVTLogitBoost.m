classdef batch_pSampVTLogitBoost < batch_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pSampVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pSampVTLogitBoost';
    end
    
  end % methods
  
end %

