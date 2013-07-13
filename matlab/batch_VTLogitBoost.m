classdef batch_VTLogitBoost < batch_boost_basic
  %batch_AOSOBoostlog Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'VTLogitBoost';
    end
    
  end
  
end

