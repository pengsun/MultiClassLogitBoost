classdef batch_pExtSamp2VTLogitBoost < batch_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pExtSamp2VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pExtSamp2VTLogitBoost';
    end
    
  end % methods
  
end %

