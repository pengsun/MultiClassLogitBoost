classdef batch_pVbExtSamp11VTLogitBoost < batch_vb3_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp11VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp11VTLogitBoost';
    end
    
  end % methods
  
end %

