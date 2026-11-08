function run_improve(S, episodes, hs, tag)
%RUN_IMPROVE  Article revision experiments on the canonical protocol.
%  (1) Non-learning multi-channel references on every campaign cell:
%      Heur-MC-LB (K=3, greedy channels, load-aware association),
%      Heur-MC-NN (K=3, greedy channels, nearest-router association),
%      Classic-LB (K=1, load-aware association).
%  (2) LA-MCMARL with hysteresis association (assign 'H', theta = h * mean demand),
%      trained S seeds x episodes on Scenario A, N=10, K=3.
%  run_improve(8,50,[1 3],'improve')
if nargin<1, S=8; end
if nargin<2, episodes=50; end
if nargin<3, hs=[1 3]; end
if nargin<4, tag='improve'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); T=60; nEval=30;
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
    'loadVar','handover','energyPerBit','sinr_mean'};
imp=struct(); imp.metrics=mn; t0=tic;
fprintf('=== IMPROVE S=%d ep=%d ===\n',S,episodes);
cells={'A',10,'A10';'A',5,'A5';'A',20,'A20';'B',10,'B10';'C',10,'C10'};
hm={'Heur-MC-LB',3,'L';'Heur-MC-NN',3,'N';'Classic-LB',1,'L'};
for c=1:size(cells,1)
  D=nan(S,size(hm,1),numel(mn)); Tr=nan(S,size(hm,1),T,numel(mn));
  for s=1:S
    for v=1:size(hm,1)
      cfg=uwmn_config(cells{c,1},cells{c,2},hm{v,2});
      [ev,tr]=uwmn_evalheur(cfg,s,nEval,T,hm{v,3});
      for k=1:numel(mn), D(s,v,k)=ev.(mn{k}); end
      Tr(s,v,:,:)=reshape(tr,[1 1 T numel(mn)]);
    end
  end
  imp.heur.(cells{c,3})=struct('methods',{hm(:,1)'},'Data',D,'trace',Tr);
  fprintf('   heuristics %s done (%.1f min)\n',cells{c,3},toc(t0)/60);
  save(fullfile(resdir,[tag '_partial.mat']),'imp','-v7');
end
H=nan(S,numel(hs),numel(mn)); HT=nan(S,numel(hs),T,numel(mn)); HC=nan(S,numel(hs),episodes);
for s=1:S
  for v=1:numel(hs)
    aCfg=uwmn_methodcfg('Proposed',struct('assign','H','hyst',hs(v)));
    cfg=uwmn_config('A',10,aCfg.K);
    rng(1000*s+40+v,'twister');
    [ag,cv]=uwmn_train(aCfg,cfg,s,episodes,T);
    [ev,tr]=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:numel(mn), H(s,v,k)=ev.(mn{k}); end
    HT(s,v,:,:)=reshape(tr,[1 1 T numel(mn)]); HC(s,v,:)=cv;
  end
  fprintf('   hysteresis seed %d/%d (%.1f min)\n',s,S,toc(t0)/60);
  imp.hyst=struct('h',hs,'Data',H,'trace',HT,'curves',HC);
  save(fullfile(resdir,[tag '_partial.mat']),'imp','-v7');
end
imp.protocol=struct('S',S,'episodes',episodes,'nEval',nEval,'T',T,...
  'rng','rng(1000*seed+40+v,''twister'') before each hysteresis train/eval; heuristics deterministic given env seeds');
save(fullfile(resdir,[tag '.mat']),'imp','-v7');
fprintf('=== IMPROVE DONE in %.1f min ===\n',toc(t0)/60);
end
