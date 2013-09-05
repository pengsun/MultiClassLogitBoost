classdef batch_pExtSamp4VTLogitBoost < batch_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pExtSamp4VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pExtSamp4VTLogitBoost';
    end
    
  end % methods
  
end %

