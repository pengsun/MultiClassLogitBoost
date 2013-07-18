classdef batch_VTTCLogitBoost < batch_tcboost_basic
  %batch_VTTCLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = VTTCLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'VTTCLogitBoost';
    end
    
  end
  
end

