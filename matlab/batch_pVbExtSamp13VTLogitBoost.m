classdef batch_pVbExtSamp13VTLogitBoost < batch_vb5_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp13VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp13VTLogitBoost';
    end
    
  end % methods
  
end %

