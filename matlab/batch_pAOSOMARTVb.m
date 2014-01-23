classdef batch_pAOSOMARTVb < batch_aososampboostvb_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pAOSOMARTVb();
    end
    
    function na = get_algo_name(obj)
      na = 'pAOSOMARTVb';
    end
    
  end % methods
  
end %

