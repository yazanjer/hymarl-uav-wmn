function [ev, tr] = uwmn_evalheur(cfg, seed, neps, T, assignMode)
%UWMN_EVALHEUR  Non-learning multi-channel reference (article revision): UAVs hold
%   their initial positions and powers; channels are re-assigned at every step by
%   the classical greedy rule NetworkSim.greedyChannels (K channels); users are
%   associated by the load-aware rule (assignMode 'L') or the nearest router ('N').
%   Uses the same environment seeds as uwmn_eval, so results are paired.
if nargin<4, T=60; end
if nargin<5, assignMode='L'; end
env = NetworkSim(cfg, seed);
mn = {'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
      'loadVar','handover','energyPerBit','sinr_mean'};
acc = zeros(neps, numel(mn)); tr = zeros(T, numel(mn));
for e=1:neps
  env.reset(seed*7+e); mm=zeros(T,numel(mn));
  for t=1:T
    ea.dv=zeros(cfg.N,2); ea.dP=zeros(cfg.N,1);
    ea.chan=env.greedyChannels();
    if assignMode=='L', ea.assign=env.loadBalanceAssign(); else, ea.assign=[]; end
    [~,~,m]=env.step(ea);
    for k=1:numel(mn), mm(t,k)=m.(mn{k}); end
  end
  acc(e,:)=mean(mm,1); tr=tr+mm/neps;
end
mu=mean(acc,1); for k=1:numel(mn), ev.(mn{k})=mu(k); end
end
