classdef batch_pVbExtSamp5VTLogitBoost < batch_vb_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp5VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp5VTLogitBoost';
    end
    
  end % methods
  
end %

