classdef batch_pVbExtSamp13AOSOVTLogitBoost < batch_vb5_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp13AOSOVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp13AOSOVTLogitBoost';
    end
    
  end % methods
  
end %

