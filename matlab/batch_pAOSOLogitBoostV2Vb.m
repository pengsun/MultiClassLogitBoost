classdef batch_pAOSOLogitBoostV2Vb < batch_aososampboostvb_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOLogitBoostV2Vb();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOLogitBoostV2Vb';
    end
    
  end % methods
  
end %

