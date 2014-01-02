classdef batch_pAOSOGradAutostepBoost < batch_aososampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOGradAutostepBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOGradAutostepBoost';
    end
    
  end % methods
  
end %

