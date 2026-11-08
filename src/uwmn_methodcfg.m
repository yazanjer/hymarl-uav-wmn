function a = uwmn_methodcfg(name, overrides)
%UWMN_METHODCFG  Agent configuration for each method, with optional overrides
%   (used for ablation/sensitivity). name in {B1,B2,Proposed}. overrides is a
%   struct of fields to merge (e.g. struct('useLyap',false) to ablate Lyapunov).
a = struct('gamma',0.99,'tau',1.0,'tau_soft',0.01,'lrA',1e-3,'lrC',1e-3,...
  'noise',0.2,'psmooth',0.15,'dPmax',1.0,'dS',0.2,'kL',0.5,'Ttarget',0.3,...
  'K',1,'powerMode','cont','useLyap',false,'aug',false,'assign','N','hyst',0,'chanMode','learned');
switch name
  case 'B1',       a.K=1; a.powerMode='cont'; a.useLyap=true;  a.aug=false; a.assign='N';
  case 'B2',       a.K=1; a.powerMode='disc'; a.useLyap=false; a.aug=true;  a.assign='N';
  case 'Proposed', a.K=3; a.powerMode='cont'; a.useLyap=true;  a.aug=true;  a.assign='L';
end
if nargin>1 && ~isempty(overrides)
  f=fieldnames(overrides); for i=1:numel(f), a.(f{i})=overrides.(f{i}); end
end
end
