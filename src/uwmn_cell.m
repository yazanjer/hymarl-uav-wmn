function R = uwmn_cell(scenario, N, S, episodes, methods, T)
%UWMN_CELL  Run one experimental cell: train+eval every method over S seeds on
%   (scenario, N), returning the stats struct. B1/B2/Classic are single-channel;
%   Proposed uses its configured K. Identical seeds across methods (paired stats).
if nargin<5||isempty(methods), methods={'Classic','B1','B2','Proposed'}; end
if nargin<6, T=60; end
mn = {'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
      'loadVar','handover','energyPerBit','sinr_mean'};
nM=numel(methods); nMet=numel(mn);
Data=nan(S,nM,nMet); curves=cell(nM,1);
for s=1:S
  for mi=1:nM
    nm=methods{mi};
    if strcmp(nm,'Classic')
      cfg=uwmn_config(scenario,N,1);
      ev=uwmn_evalclassic(cfg,s,30,T);
    else
      aCfg=uwmn_methodcfg(nm);
      cfg=uwmn_config(scenario,N,aCfg.K);
      [ag,cv]=uwmn_train(aCfg,cfg,s,episodes,T);
      if s==1, curves{mi}=cv; end
      ev=uwmn_eval(ag,aCfg,cfg,s,30,T);
    end
    for k=1:nMet, Data(s,mi,k)=ev.(mn{k}); end
  end
  fprintf('   [%s N=%d] seed %d/%d done\n', scenario, N, s, S);
end
R=uwmn_stats(Data,methods,mn,curves);
end
