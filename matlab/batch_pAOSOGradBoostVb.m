classdef batch_pAOSOGradBoostVb < batch_aososampboostvb_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOGradBoostVb();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOGradBoostVb';
    end
    
  end % methods
  
end %

