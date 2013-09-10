classdef batch_pVbExtSamp10VTLogitBoost < batch_vb2_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp10VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp10VTLogitBoost';
    end
    
  end % methods
  
end %

