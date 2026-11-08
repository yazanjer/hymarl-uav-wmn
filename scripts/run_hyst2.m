function run_hyst2(S, episodes, tag)
%RUN_HYST2  Article revision, second hysteresis experiment on Scenario A, N=10:
%  (a) LA-MCMARL without hysteresis (h=0) under a fresh stream rng(1000*s+40),
%      to separate the hysteresis effect from the random-stream effect;
%  (b) HyMARL (greedy channels) with hysteresis h=3, rng(1000*s+51).
%  run_hyst2(8,50,'hyst2')
if nargin<1, S=8; end
if nargin<2, episodes=50; end
if nargin<3, tag='hyst2'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); T=60; nEval=30;
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
    'loadVar','handover','energyPerBit','sinr_mean'};
specs={'LA-h0',struct('assign','L'),40; 'Hy-h3',struct('chanMode','greedy','assign','H','hyst',3),51};
out=struct(); out.metrics=mn; out.names={specs{:,1}}; t0=tic;
D=nan(S,size(specs,1),numel(mn)); Tr=nan(S,size(specs,1),T,numel(mn));
for s=1:S
  for v=1:size(specs,1)
    aCfg=uwmn_methodcfg('Proposed',specs{v,2}); cfg=uwmn_config('A',10,aCfg.K);
    rng(1000*s+specs{v,3},'twister');
    ag=uwmn_train(aCfg,cfg,s,episodes,T); [ev,tr]=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:numel(mn), D(s,v,k)=ev.(mn{k}); end
    Tr(s,v,:,:)=reshape(tr,[1 1 T numel(mn)]);
  end
  fprintf('   hyst2 seed %d/%d (%.1f min)\n',s,S,toc(t0)/60);
end
out.Data=D; out.trace=Tr;
save(fullfile(resdir,[tag '.mat']),'out','-v7');
fprintf('=== HYST2 DONE in %.1f min ===\n',toc(t0)/60);
end
