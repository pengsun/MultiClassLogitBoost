classdef batch_pVbExtSamp12VTLogitBoost < batch_vb4_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp12VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp12VTLogitBoost';
    end
    
  end % methods
  
end %

