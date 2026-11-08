function ev = uwmn_evalclassic(cfg, seed, neps, T)
%UWMN_EVALCLASSIC  Non-learning classical reference: hold position, fixed power,
%   least-occupied channel, nearest-router assignment.
if nargin<4, T=60; end
env = NetworkSim(cfg, seed);
mn = {'delay','pdr','throughput','chanUtil','cci','specEff','fairness',...
      'loadVar','handover','energyPerBit','sinr_mean'};
acc = zeros(neps, numel(mn));
for e=1:neps
  env.reset(seed*7+e); mm=zeros(T,numel(mn));
  for t=1:T
    ea.dv=zeros(cfg.N,2); ea.dP=zeros(cfg.N,1);
    K=cfg.K; occ=zeros(1,K); for k=1:K, occ(k)=sum(env.uavChan==k); end
    [~,b]=min(occ); ea.chan=b*ones(cfg.N,1); ea.assign=[];
    [~,~,m]=env.step(ea);
    for k=1:numel(mn), mm(t,k)=m.(mn{k}); end
  end
  acc(e,:)=mean(mm,1);
end
mu=mean(acc,1); for k=1:numel(mn), ev.(mn{k})=mu(k); end
end
