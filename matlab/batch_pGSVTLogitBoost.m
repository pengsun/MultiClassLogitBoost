classdef batch_pGSVTLogitBoost < batch_boost_basic
  %batch_pVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pGSVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pGSVTLogitBoost';
    end
    
  end
  
end

