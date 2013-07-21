classdef batch_VTDropoutLogitBoost < batch_boost_basic
  %batch_AOSOBoostlog Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = VTDropoutLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'VTDropoutLogitBoost';
    end
    
  end
  
end

