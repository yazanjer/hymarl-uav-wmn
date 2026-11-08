function [ev, tr] = uwmn_eval(agent, aCfg, cfg, seed, neps, T)
%UWMN_EVAL  Evaluate a frozen agent; returns struct of the 11 measured metrics.
if nargin<6, T=60; end
env = NetworkSim(cfg, seed); useAug = aCfg.aug;
mn = {'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
      'loadVar','handover','energyPerBit','sinr_mean'};
acc = zeros(neps, numel(mn)); tr = zeros(T, numel(mn));
for e=1:neps
  env.reset(seed*7+e); o=uwmn_getobs(env,useAug); mm=zeros(T,numel(mn));
  for t=1:T
    [ea,~]=agent.act(o,env,false);
    [~,~,m]=env.step(ea); o=uwmn_getobs(env,useAug);
    for k=1:numel(mn), mm(t,k)=m.(mn{k}); end
  end
  acc(e,:)=mean(mm,1); tr=tr+mm/neps;
end
mu=mean(acc,1); for k=1:numel(mn), ev.(mn{k})=mu(k); end
end
