classdef batch_pCoSamp2VTLogitBoost < batch_CoSampboost_basic
  %batch_CoSampboost_basic Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pCoSamp2VTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pCoSamp2VTLogitBoost';
    end
    
  end % methods
  
end %

