function run_hybrid(S, episodes, cellsIn, tag)
%RUN_HYBRID  Article revision: learned mobility and power (shared TD3 actor with the
%  delay-driven correction and neighbour aggregation) + classical greedy channel
%  assignment (NetworkSim.greedyChannels, K=3) + load-aware association.
%  run_hybrid(8,50,{'A',10,'A10'},'hybrid')
if nargin<1, S=8; end
if nargin<2, episodes=50; end
if nargin<3, cellsIn={'A',10,'A10'}; end
if nargin<4, tag='hybrid'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); T=60; nEval=30;
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
    'loadVar','handover','energyPerBit','sinr_mean'};
hyb=struct(); hyb.metrics=mn; t0=tic;
fprintf('=== HYBRID S=%d ep=%d ===\n',S,episodes);
for c=1:size(cellsIn,1)
  D=nan(S,numel(mn)); Tr=nan(S,T,numel(mn)); Cv=nan(S,episodes);
  for s=1:S
    aCfg=uwmn_methodcfg('Proposed',struct('chanMode','greedy'));
    cfg=uwmn_config(cellsIn{c,1},cellsIn{c,2},aCfg.K);
    rng(1000*s+50,'twister');
    [ag,cv]=uwmn_train(aCfg,cfg,s,episodes,T);
    [ev,tr]=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:numel(mn), D(s,k)=ev.(mn{k}); end
    Tr(s,:,:)=reshape(tr,[1 T numel(mn)]); Cv(s,:)=cv;
    fprintf('   [%s] seed %d/%d (%.1f min)\n',cellsIn{c,3},s,S,toc(t0)/60);
  end
  hyb.(cellsIn{c,3})=struct('Data',D,'trace',Tr,'curves',Cv);
  save(fullfile(resdir,[tag '_partial.mat']),'hyb','-v7');
end
hyb.protocol=struct('S',S,'episodes',episodes,'nEval',nEval,'T',T,'rng','rng(1000*seed+50,''twister'')');
save(fullfile(resdir,[tag '.mat']),'hyb','-v7');
fprintf('=== HYBRID DONE in %.1f min ===\n',toc(t0)/60);
end
