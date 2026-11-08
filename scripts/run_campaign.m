function run_campaign(S, episodes, Sab, epAbl)
%RUN_CAMPAIGN  Breadth runs that complete the reduced protocol beyond the
%   Scenario-A/N=10 headline: N-sweep {5,20}, Scenarios B & C, a component
%   ablation of the proposed framework, and K / reward-weight sensitivity.
%   All results are real and saved to results/campaign.mat.
if nargin<1, S=6;  end
if nargin<2, episodes=40; end
if nargin<3, Sab=4; end
if nargin<4, epAbl=30; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); if ~exist(resdir,'dir'), mkdir(resdir); end
methods={'Classic','B1','B2','Proposed'};
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
    'loadVar','handover','energyPerBit','sinr_mean'};
camp=struct(); t0=tic;
fprintf('=== CAMPAIGN S=%d ep=%d Sab=%d ===\n',S,episodes,Sab);

% ---- N-sweep (Scenario A); N=10 already in headline_A8 ----
fprintf('-- A/N=5 --\n');  camp.A5 =uwmn_cell('A',5, S,episodes,methods);
fprintf('-- A/N=20 --\n'); camp.A20=uwmn_cell('A',20,S,episodes,methods);
% ---- Scenarios B, C at N=10 ----
fprintf('-- B/N=10 --\n'); camp.B10=uwmn_cell('B',10,S,episodes,methods);
fprintf('-- C/N=10 --\n'); camp.C10=uwmn_cell('C',10,S,episodes,methods);
save(fullfile(resdir,'campaign_cells.mat'),'camp','-v7');

% ---- Ablation of the proposed framework (A, N=10) ----
fprintf('-- Ablation --\n');
ablNames={'Full','noLyap','noAug','noLoadBal','noMultiChan'};
ablOv={struct(),struct('useLyap',false),struct('aug',false),...
       struct('assign','N'),struct('K',1)};
AbData=nan(Sab,numel(ablNames),numel(mn));
for s=1:Sab
  for v=1:numel(ablNames)
    aCfg=uwmn_methodcfg('Proposed',ablOv{v});
    cfg=uwmn_config('A',10,aCfg.K);
    ag=uwmn_train(aCfg,cfg,s,epAbl);
    ev=uwmn_eval(ag,aCfg,cfg,s,25);
    for k=1:numel(mn), AbData(s,v,k)=ev.(mn{k}); end
  end
  fprintf('   ablation seed %d/%d\n',s,Sab);
end
camp.ablation=struct('names',{ablNames},'Data',AbData,'metrics',{mn});

% ---- Sensitivity: number of channels K ----
fprintf('-- Sensitivity K --\n');
Ks=[1 2 3 4]; sensK=nan(numel(Ks),numel(mn));
for i=1:numel(Ks)
  aCfg=uwmn_methodcfg('Proposed',struct('K',Ks(i)));
  cfg=uwmn_config('A',10,Ks(i));
  ag=uwmn_train(aCfg,cfg,1,epAbl); ev=uwmn_eval(ag,aCfg,cfg,1,25);
  for k=1:numel(mn), sensK(i,k)=ev.(mn{k}); end
end
% ---- Sensitivity: load-variance reward weight ----
fprintf('-- Sensitivity lambda_var --\n');
lams=[0.1 0.4 0.8]; sensL=nan(numel(lams),numel(mn));
for i=1:numel(lams)
  aCfg=uwmn_methodcfg('Proposed');
  cfg=uwmn_config('A',10,3); cfg.lam3_var=lams(i);
  ag=uwmn_train(aCfg,cfg,1,epAbl); ev=uwmn_eval(ag,aCfg,cfg,1,25);
  for k=1:numel(mn), sensL(i,k)=ev.(mn{k}); end
end
camp.sensK=struct('K',Ks,'Data',sensK,'metrics',{mn});
camp.sensL=struct('lam',lams,'Data',sensL,'metrics',{mn});

save(fullfile(resdir,'campaign.mat'),'camp','-v7');
fprintf('=== CAMPAIGN DONE in %.1f min ===\nDONE_CAMPAIGN\n',toc(t0)/60);
end
