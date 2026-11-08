function R = run_headline(S, episodes, tag)
%RUN_HEADLINE  Scenario-A headline comparison: Classic, B1, B2, Proposed.
%   Trains each learning method over S seeds, evaluates frozen policies (30 eps),
%   and runs the Wilcoxon / Friedman / Bonferroni protocol (thesis Sec.3.5.2.5).
%   R = RUN_HEADLINE(S, episodes, tag)
if nargin<1, S=5;  end
if nargin<2, episodes=60; end
if nargin<3, tag='A'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); if ~exist(resdir,'dir'), mkdir(resdir); end
scenario='A'; N=10; K=3; T=60; nEval=30;
methods={'Classic','B1','B2','Proposed'}; nM=numel(methods);
metricNames={'delay','pdr','throughput','chanUtil','cci','specEff', ...
             'fairness','loadVar','handover','energyPerBit','sinr_mean'};
nMet=numel(metricNames);
Data=nan(S,nM,nMet);         % seed x method x metric
curves=cell(nM,1);
curvesAll=nan(S,nM,episodes);   % training return, every seed
Trace=nan(S,nM,T,nMet);         % per-step eval metrics (mean over eval episodes)
fprintf('=== Headline Scenario A: S=%d seeds, %d episodes ===\n',S,episodes);
for s=1:S
  for mi=1:nM
    nm=methods{mi};
    % B1/B2/Classic are single-channel (their design); Proposed is multi-channel
    % (K=3) — the multi-channel interference-mitigation contribution (Obj.2).
    Km=1; if strcmp(nm,'Proposed'), Km=K; end
    cfg=uwmn_config(scenario,N,Km);
    rng(1000*s+mi,'twister');      % reproducible agent init + minibatch sampling
    t0=tic;
    if strcmp(nm,'Classic')
      [ev,tr]=eval_classic(cfg,s,nEval);
    else
      [ag,cv]=train_agent(nm,cfg,s,episodes,T);
      if s==1, curves{mi}=cv; end
      curvesAll(s,mi,:)=cv;
      if ~contains(tag,'smoke'), md=fullfile(resdir,'models',tag); if ~exist(md,'dir'), mkdir(md); end; save(fullfile(md,sprintf('seed%d_%s.mat',s,nm)),'ag','-v7'); end
      [ev,tr]=eval_agent(ag,nm,cfg,s,nEval,T);
    end
    for k=1:nMet, Data(s,mi,k)=ev.(metricNames{k}); end
    Trace(s,mi,:,:)=reshape(tr,[1 1 T nMet]);
    fprintf('  seed %d %-9s delay=%.1f thr=%.1f pdr=%.2f fair=%.2f (%.0fs)\n',...
        s,nm,ev.delay,ev.throughput,ev.pdr,ev.fairness,toc(t0));
  end
end

% ---- summary + statistics ----
pIdx=find(strcmp(methods,'Proposed'));
fprintf('\n=== MEAN +/- SD (Scenario A, N=%d) ===\n',N);
Stats=struct();
for k=1:nMet
  mu=squeeze(mean(Data(:,:,k),1)); sd=squeeze(std(Data(:,:,k),0,1));
  fprintf('%-13s',metricNames{k});
  for mi=1:nM, fprintf(' | %-8s %.2f±%.2f',methods{mi},mu(mi),sd(mi)); end
  % Wilcoxon proposed vs B1 and B2 (paired signed-rank)
  pv=nan(1,nM); W=nan(1,nM); rb=nan(1,nM);
  for mi=1:nM
    if mi==pIdx, continue; end
    x=Data(:,pIdx,k); y=Data(:,mi,k);
    if S>=2 && any(x~=y)
      [p,~,st]=signrank(x,y);
      pv(mi)=p; if isfield(st,'signedrank'), W(mi)=st.signedrank; end
      d=x-y; rb(mi)=(sum(d>0)-sum(d<0))/max(sum(d~=0),1);   % rank-biserial
    end
  end
  fprintf('  | Wilx p(vsB1)=%.3f p(vsB2)=%.3f\n',pv(strcmp(methods,'B1')),pv(strcmp(methods,'B2')));
  Stats.(metricNames{k})=struct('mean',mu,'sd',sd,'wilcoxon_p',pv,'W',W,'rankbiserial',rb);
end

% Friedman across methods on the headline metric (throughput) if enough seeds
if S>=3
  try
    fp_delay=friedman(squeeze(Data(:,:,1)),1,'off');
    fp_thr =friedman(squeeze(Data(:,:,3)),1,'off');
    fprintf('\nFriedman p: delay=%.4f throughput=%.4f (Bonferroni alpha=%.4f)\n',...
        fp_delay,fp_thr,0.05/3);
    Stats.friedman_delay=fp_delay; Stats.friedman_throughput=fp_thr;
  catch e, fprintf('Friedman skipped: %s\n',e.message); end
end

R.Data=Data; R.methods=methods; R.metricNames=metricNames; R.Stats=Stats; R.curves=curves;
R.curvesAll=curvesAll; R.trace=Trace; R.rngPolicy='rng(1000*seed+methodIdx,''twister'') before each train/eval';
save(fullfile(resdir,sprintf('headline_%s.mat',tag)),'R','-v7');
writeCSV(fullfile(resdir,sprintf('headline_%s.csv',tag)),Data,methods,metricNames);
fprintf('\nSaved results to results/headline_%s.mat / .csv\n',tag);
end

% =================== training ===================
function [agent,curve]=train_agent(name,cfg,seed,episodes,T)
aCfg=methodCfg(name); env=NetworkSim(cfg,seed); env.reset(seed);
useAug=aCfg.aug;
dObs=useAug*env.obsDimAug()+(~useAug)*env.obsDim();
agent=Agent(dObs,aCfg);
cap=30000; dA=agent.actDim;
B.o=zeros(cap,dObs);B.a=zeros(cap,dA);B.r=zeros(cap,1);
B.o2=zeros(cap,dObs);B.ctx=zeros(cap,dObs);B.ctx2=zeros(cap,dObs);
bn=0; batch=128; curve=nan(episodes,1);
for ep=1:episodes
  env.reset(seed*1000+ep); o=getobs(env,useAug); epR=0;
  for t=1:T
    ctx=repmat(mean(o,1),size(o,1),1);
    [ea,av]=agent.act(o,env,true);
    [~,r,~]=env.step(ea); o2=getobs(env,useAug);
    ctx2=repmat(mean(o2,1),size(o2,1),1);
    for i=1:size(o,1)
      bn=bn+1; ix=mod(bn-1,cap)+1;
      B.o(ix,:)=o(i,:);B.a(ix,:)=av(i,:);B.r(ix)=r(i);
      B.o2(ix,:)=o2(i,:);B.ctx(ix,:)=ctx(i,:);B.ctx2(ix,:)=ctx2(i,:);
    end
    o=o2; epR=epR+mean(r);
    if bn>=batch
      ns=min(bn,cap); ix=randi(ns,[batch,1]);
      mb.o=B.o(ix,:);mb.a=B.a(ix,:);mb.r=B.r(ix);mb.o2=B.o2(ix,:);
      mb.ctx=B.ctx(ix,:);mb.ctx2=B.ctx2(ix,:);
      agent.update(mb);
    end
  end
  curve(ep)=epR;
end
end

function o=getobs(env,useAug)
if useAug, o=env.getObsAug(); else, o=env.getObs(); end
end

% =================== evaluation ===================
function [ev,tr]=eval_agent(agent,name,cfg,seed,neps,T)
aCfg=methodCfg(name); env=NetworkSim(cfg,seed); useAug=aCfg.aug;
[ev,tr]=accEval(@()agent, env, useAug, seed, neps, T, false);
end
function [ev,tr]=eval_classic(cfg,seed,neps)
env=NetworkSim(cfg,seed);
[ev,tr]=accEval([], env, false, seed, neps, 60, true);
end
function [ev,tr]=accEval(agFn, env, useAug, seed, neps, T, classic)
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness','loadVar','handover','energyPerBit','sinr_mean'};
acc=zeros(neps,numel(mn)); tr=zeros(T,numel(mn));
for e=1:neps
  env.reset(seed*7+e); o=getobs(env,useAug); mm=zeros(T,numel(mn));
  for t=1:T
    if classic
      ea.dv=zeros(env.cfg.N,2); ea.dP=zeros(env.cfg.N,1);
      ea.chan=leastOcc(env); ea.assign=[];
    else
      ag=agFn(); [ea,~]=ag.act(o,env,false);
    end
    [~,~,m]=env.step(ea); o=getobs(env,useAug);
    for k=1:numel(mn), mm(t,k)=m.(mn{k}); end
  end
  acc(e,:)=mean(mm,1); tr=tr+mm/neps;
end
mu=mean(acc,1);
for k=1:numel(mn), ev.(mn{k})=mu(k); end
end
function ch=leastOcc(env)
K=env.cfg.K;occ=zeros(1,K);for k=1:K,occ(k)=sum(env.uavChan==k);end
[~,b]=min(occ); ch=b*ones(env.cfg.N,1);
end

% =================== configs ===================
function a=methodCfg(name)
a=struct('gamma',0.99,'tau',1.0,'tau_soft',0.01,'lrA',1e-3,'lrC',1e-3,...
  'noise',0.2,'psmooth',0.15,'dPmax',1.0,'dS',0.2,'kL',0.5,'Ttarget',0.3,...
  'K',1,'powerMode','cont','useLyap',false,'aug',false,'assign','N','hyst',0,'chanMode','learned');
switch name
  case 'B1',       a.K=1; a.powerMode='cont'; a.useLyap=true;  a.aug=false; a.assign='N';
  case 'B2',       a.K=1; a.powerMode='disc'; a.useLyap=false; a.aug=true;  a.assign='N';
  case 'Proposed', a.K=3; a.powerMode='cont'; a.useLyap=true;  a.aug=true;  a.assign='L';
end
end

function writeCSV(fn,Data,methods,mn)
fid=fopen(fn,'w'); fprintf(fid,'metric,method,mean,sd\n');
for k=1:numel(mn)
  for mi=1:numel(methods)
    fprintf(fid,'%s,%s,%.5f,%.5f\n',mn{k},methods{mi},mean(Data(:,mi,k)),std(Data(:,mi,k)));
  end
end
fclose(fid);
end
