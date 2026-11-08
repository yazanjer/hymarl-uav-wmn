function [agent, curve] = uwmn_train(aCfg, cfg, seed, episodes, T)
%UWMN_TRAIN  Train one agent (config aCfg) on env cfg for `episodes` episodes.
%   Parameter-shared TD3 with CTDE mean-field twin critics (thesis Alg.3.1 core).
if nargin<5, T=60; end
env = NetworkSim(cfg, seed); env.reset(seed);
useAug = aCfg.aug;
dObs = useAug*env.obsDimAug() + (~useAug)*env.obsDim();
agent = Agent(dObs, aCfg);
cap=30000; dA=agent.actDim;
B.o=zeros(cap,dObs); B.a=zeros(cap,dA); B.r=zeros(cap,1);
B.o2=zeros(cap,dObs); B.ctx=zeros(cap,dObs); B.ctx2=zeros(cap,dObs);
bn=0; batch=128; curve=nan(episodes,1);
for ep=1:episodes
  env.reset(seed*1000+ep); o=uwmn_getobs(env,useAug); epR=0;
  for t=1:T
    ctx=repmat(mean(o,1),size(o,1),1);
    [ea,av]=agent.act(o,env,true);
    [~,r,~]=env.step(ea); o2=uwmn_getobs(env,useAug);
    ctx2=repmat(mean(o2,1),size(o2,1),1);
    for i=1:size(o,1)
      bn=bn+1; ix=mod(bn-1,cap)+1;
      B.o(ix,:)=o(i,:); B.a(ix,:)=av(i,:); B.r(ix)=r(i);
      B.o2(ix,:)=o2(i,:); B.ctx(ix,:)=ctx(i,:); B.ctx2(ix,:)=ctx2(i,:);
    end
    o=o2; epR=epR+mean(r);
    if bn>=batch
      ns=min(bn,cap); ix=randi(ns,[batch,1]);
      mb.o=B.o(ix,:); mb.a=B.a(ix,:); mb.r=B.r(ix); mb.o2=B.o2(ix,:);
      mb.ctx=B.ctx(ix,:); mb.ctx2=B.ctx2(ix,:);
      agent.update(mb);
    end
  end
  curve(ep)=epR;
end
end
