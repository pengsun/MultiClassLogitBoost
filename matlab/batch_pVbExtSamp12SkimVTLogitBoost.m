classdef batch_pVbExtSamp12SkimVTLogitBoost < batch_vb4Skim_sampboost_basic
  %batch_pSampVTLogitBoost Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function h = get_handle(obj) %#ok<MANU>
      h = pVbExtSamp12SkimVTLogitBoost();
    end
    
    function na = get_algo_name(obj)
      na = 'pVbExtSamp12SkimVTLogitBoost';
    end
    
  end % methods
  
end %

