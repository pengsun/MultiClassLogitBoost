classdef batch_pExtSampVTLogitBoost < batch_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pExtSampVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pExtSampVTLogitBoost';
    end
    
  end % methods
  
end %

