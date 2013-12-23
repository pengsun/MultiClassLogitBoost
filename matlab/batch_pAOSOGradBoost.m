classdef batch_pAOSOGradBoost < batch_aososampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOGradBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOGradBoost';
    end
    
  end % methods
  
end %

