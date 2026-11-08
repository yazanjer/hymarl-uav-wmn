function R = run_fair(S, episodes, tag)
%RUN_FAIR  Equal-spectrum fair comparison to isolate the ARCHITECTURE contribution
%   from the channel-count advantage (addresses the K=3-vs-K=1 confound).
%   All learners operate at K=3 except the explicit single-channel variant:
%     Proposed    : full framework (GNN aug + Lyapunov + load-balancing), K=3
%     MC-MATD3    : vanilla multi-channel learner (none of the above), K=3
%     B1-MC       : B1 (MATD3 + Lyapunov) given the multi-channel head, K=3
%     B2-MC       : B2 (self-coord + discrete power) given multi-channel, K=3
%     Proposed-K1 : the full architecture restricted to a single channel, K=1
%   If Proposed beats MC-MATD3 at equal K=3, the architecture earns its place.
%
%   R = run_fair(12, 150, 'fairA')
if nargin<1, S=12; end
if nargin<2, episodes=150; end
if nargin<3, tag='fairA'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results'); if ~exist(resdir,'dir'), mkdir(resdir); end
scenario='A'; N=10; T=60; nEval=30;

% label, base method, config overrides
specs = {
  'Proposed',    'Proposed', struct();
  'MC-MATD3',    'Proposed', struct('aug',false,'useLyap',false,'assign','N');
  'B1-MC',       'B1',       struct('K',3);
  'B2-MC',       'B2',       struct('K',3);
  'Proposed-K1', 'Proposed', struct('K',1);
};
methods = specs(:,1)';
mn={'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
    'loadVar','handover','energyPerBit','sinr_mean'};
nM=numel(methods); nMet=numel(mn);
Data=nan(S,nM,nMet); curves=cell(nM,1);
curvesAll=nan(S,nM,episodes); Trace=nan(S,nM,T,nMet);
fprintf('=== FAIR (equal-K) comparison: S=%d ep=%d ===\n',S,episodes);
for s=1:S
  for mi=1:nM
    aCfg=uwmn_methodcfg(specs{mi,2}, specs{mi,3});
    cfg=uwmn_config(scenario,N,aCfg.K);
    rng(1000*s+mi,'twister');      % reproducible agent init + minibatch sampling
    t0=tic; [ag,cv]=uwmn_train(aCfg,cfg,s,episodes,T);
    if s==1, curves{mi}=cv; end
    curvesAll(s,mi,:)=cv;
    if ~contains(tag,'smoke'), md=fullfile(resdir,'models',tag); if ~exist(md,'dir'), mkdir(md); end; save(fullfile(md,sprintf('seed%d_%s.mat',s,methods{mi})),'ag','-v7'); end
    [ev,tr]=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
    for k=1:nMet, Data(s,mi,k)=ev.(mn{k}); end
    Trace(s,mi,:,:)=reshape(tr,[1 1 T nMet]);
    fprintf('  seed %d %-11s delay=%.1f thr=%.1f fair=%.2f loadVar=%.1f (%.0fs)\n',...
        s,methods{mi},ev.delay,ev.throughput,ev.fairness,ev.loadVar,toc(t0));
  end
end
% Proposed vs MC-MATD3 is the key paired test (architecture effect at equal K)
R=uwmn_stats_named(Data,methods,mn,curves,'Proposed','MC-MATD3');
R.curvesAll=curvesAll; R.trace=Trace; R.rngPolicy='rng(1000*seed+methodIdx,''twister'') before each train/eval';
save(fullfile(resdir,sprintf('%s.mat',tag)),'R','-v7');
fprintf('\nSaved results/%s.mat\n',tag);
% quick console verdict
kd=find(strcmp(mn,'delay')); kt=find(strcmp(mn,'throughput'));
pD=R.Stats.delay.wilcoxon_p(strcmp(methods,'MC-MATD3'));
pT=R.Stats.throughput.wilcoxon_p(strcmp(methods,'MC-MATD3'));
fprintf('KEY: Proposed vs MC-MATD3 (equal K=3) Wilcoxon p: delay=%.3f throughput=%.3f\n',pD,pT);
end

function R=uwmn_stats_named(Data,methods,mn,curves,refName,~)
% like uwmn_stats but reference = refName (Proposed) vs all others
S=size(Data,1); nMet=numel(mn); pIdx=find(strcmp(methods,refName));
Stats=struct();
for k=1:nMet
  mu=squeeze(mean(Data(:,:,k),1)); sd=squeeze(std(Data(:,:,k),0,1));
  pv=nan(1,numel(methods)); rb=nan(1,numel(methods));
  for mi=1:numel(methods)
    if mi==pIdx, continue; end
    x=Data(:,pIdx,k); y=Data(:,mi,k);
    if S>=2 && any(x~=y)
      pv(mi)=signrank(x,y);
      d=x-y; rb(mi)=(sum(d>0)-sum(d<0))/max(sum(d~=0),1);
    end
  end
  Stats.(mn{k})=struct('mean',mu,'sd',sd,'wilcoxon_p',pv,'rankbiserial',rb);
end
R.Data=Data; R.methods=methods; R.metricNames=mn; R.Stats=Stats; R.curves=curves;
end
