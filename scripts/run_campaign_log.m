function run_campaign_log(S, episodes, Sab, Ssens, tag)
%RUN_CAMPAIGN_LOG  Campaign (N-sweep, Scenarios B/C, ablation, sensitivity) on the
%   canonical protocol of run_relog: S seeds x `episodes` training episodes, 30
%   evaluation episodes x 60 steps, rng(1000*seed+idx,'twister') before every
%   train/eval, all-seed training curves and per-step evaluation traces logged.
%   Results: results/<tag>.mat (struct camp, same fields as campaign.mat plus logs).
%
%   run_campaign_log(8,50,8,4,'campaign8')
if nargin<1, S=8; end
if nargin<2, episodes=50; end
if nargin<3, Sab=8; end
if nargin<4, Ssens=4; end
if nargin<5, tag='campaign8'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); T=60; nEval=30;
methods={'Classic','B1','B2','Proposed'};
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
    'loadVar','handover','energyPerBit','sinr_mean'};
camp=struct(); t0=tic;
fprintf('=== CAMPAIGN-LOG S=%d ep=%d Sab=%d Ssens=%d ===\n',S,episodes,Sab,Ssens);
cells={'A',5,'A5'; 'A',20,'A20'; 'B',10,'B10'; 'C',10,'C10'};
for c=1:size(cells,1)
  fprintf('-- %s/N=%d --\n',cells{c,1},cells{c,2});
  camp.(cells{c,3})=cellLog(cells{c,1},cells{c,2},S,episodes,methods,mn,T,nEval);
  save(fullfile(resdir,[tag '_partial.mat']),'camp','-v7');
end

fprintf('-- Ablation --\n');
ablNames={'Full','noLyap','noAug','noLoadBal','noMultiChan'};
ablOv={struct(),struct('useLyap',false),struct('aug',false),...
       struct('assign','N'),struct('K',1)};
AbData=nan(Sab,numel(ablNames),numel(mn));
for s=1:Sab
  for v=1:numel(ablNames)
    aCfg=uwmn_methodcfg('Proposed',ablOv{v});
    cfg=uwmn_config('A',10,aCfg.K);
    rng(1000*s+10+v,'twister');
    ag=uwmn_train(aCfg,cfg,s,episodes,T);
    ev=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:numel(mn), AbData(s,v,k)=ev.(mn{k}); end
  end
  fprintf('   ablation seed %d/%d (%.0f min)\n',s,Sab,toc(t0)/60);
end
camp.ablation=struct('names',{ablNames},'Data',AbData,'metrics',{mn});
camp.ablation.Stats=statsVsFirst(AbData,ablNames,mn);
save(fullfile(resdir,[tag '_partial.mat']),'camp','-v7');

fprintf('-- Sensitivity K --\n');
Ks=[1 2 3 4]; SK=nan(Ssens,numel(Ks),numel(mn));
for s=1:Ssens
  for i=1:numel(Ks)
    aCfg=uwmn_methodcfg('Proposed',struct('K',Ks(i)));
    cfg=uwmn_config('A',10,Ks(i));
    rng(1000*s+20+i,'twister');
    ag=uwmn_train(aCfg,cfg,s,episodes,T); ev=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:numel(mn), SK(s,i,k)=ev.(mn{k}); end
  end
end
fprintf('-- Sensitivity lambda_var --\n');
lams=[0.1 0.4 0.8]; SL=nan(Ssens,numel(lams),numel(mn));
for s=1:Ssens
  for i=1:numel(lams)
    aCfg=uwmn_methodcfg('Proposed');
    cfg=uwmn_config('A',10,3); cfg.lam3_var=lams(i);
    rng(1000*s+30+i,'twister');
    ag=uwmn_train(aCfg,cfg,s,episodes,T); ev=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:numel(mn), SL(s,i,k)=ev.(mn{k}); end
  end
end
% .Data keeps the seed-mean (same shape as campaign.mat) ; .DataSeeds keeps all seeds
camp.sensK=struct('K',Ks,'Data',squeeze(mean(SK,1)),'DataSeeds',SK,'metrics',{mn});
camp.sensL=struct('lam',lams,'Data',squeeze(mean(SL,1)),'DataSeeds',SL,'metrics',{mn});
camp.protocol=struct('S',S,'episodes',episodes,'Sab',Sab,'Ssens',Ssens,'nEval',nEval,'T',T,...
  'rng','rng(1000*seed+idx,''twister'') before each train/eval');
save(fullfile(resdir,[tag '.mat']),'camp','-v7');
fprintf('=== CAMPAIGN-LOG DONE in %.1f min ===\n',toc(t0)/60);
end

function R=cellLog(scenario,N,S,episodes,methods,mn,T,nEval)
nM=numel(methods); nMet=numel(mn);
Data=nan(S,nM,nMet); curves=cell(nM,1); curvesAll=nan(S,nM,episodes); Trace=nan(S,nM,T,nMet);
for s=1:S
  for mi=1:nM
    nm=methods{mi};
    rng(1000*s+mi,'twister');
    if strcmp(nm,'Classic')
      cfg=uwmn_config(scenario,N,1);
      ev=uwmn_evalclassic(cfg,s,nEval,T); tr=nan(T,nMet);
    else
      aCfg=uwmn_methodcfg(nm);
      cfg=uwmn_config(scenario,N,aCfg.K);
      [ag,cv]=uwmn_train(aCfg,cfg,s,episodes,T);
      if s==1, curves{mi}=cv; end
      curvesAll(s,mi,:)=cv;
      [ev,tr]=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    end
    for k=1:nMet, Data(s,mi,k)=ev.(mn{k}); end
    Trace(s,mi,:,:)=reshape(tr,[1 1 T nMet]);
  end
  fprintf('   [%s N=%d] seed %d/%d done\n',scenario,N,s,S);
end
R=uwmn_stats(Data,methods,mn,curves);
R.curvesAll=curvesAll; R.trace=Trace;
end

function St=statsVsFirst(D,names,mn)
% paired Wilcoxon of every variant against the first (Full model)
St=struct();
for k=1:numel(mn)
  pv=nan(1,numel(names));
  for v=2:numel(names)
    x=D(:,1,k); y=D(:,v,k);
    if size(D,1)>=2 && any(x~=y), pv(v)=signrank(x,y); end
  end
  St.(mn{k})=struct('mean',squeeze(mean(D(:,:,k),1)),'sd',squeeze(std(D(:,:,k),0,1)),'wilcoxon_p',pv);
end
end
