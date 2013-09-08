classdef batch_pVbExtSamp6VTLogitBoost < batch_vb_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp6VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp6VTLogitBoost';
    end
    
  end % methods
  
end %

