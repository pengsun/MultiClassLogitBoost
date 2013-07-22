classdef batch_pVTLogitBoost < batch_boost_basic
  %batch_pVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVTLogitBoost';
    end
    
  end
  
end

