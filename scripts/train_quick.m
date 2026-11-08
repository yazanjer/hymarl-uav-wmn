function out = train_quick(scenario, episodes, seed)
%TRAIN_QUICK  Minimal end-to-end TD3 learner on the shared NetworkSim, used to
%   de-risk the deep-RL training pipeline in MATLAB (thesis Alg. 3.1 core).
%   Parameter-shared decentralised actors + CTDE mean-field twin critics.
%   Compares the learned policy against a fixed classical baseline.
%
%   out = TRAIN_QUICK('A', 40, 1)
if nargin<1, scenario='A'; end
if nargin<2, episodes=40;  end
if nargin<3, seed=1;       end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','src'));
resdir = fullfile(here,'..','results'); if ~exist(resdir,'dir'), mkdir(resdir); end

rng(seed);
N=10; K=3; T=60;
cfg = uwmn_config(scenario, N, K);
env = NetworkSim(cfg, seed);
dObs = env.obsDim(); dAct = 3;         % [dvx, dvy, dP]

% ---- networks ----
actor  = mlp(dObs, [64 64], dAct, 'tanh');
critIn = dObs + dAct + dObs;           % obs + act + mean-obs context
c1 = mlp(critIn,[64 64],1,'lin');  c2 = mlp(critIn,[64 64],1,'lin');
tActor=actor; tc1=c1; tc2=c2;
optA=struct('m',[],'v',[]); optC1=optA; optC2=optA;  % adam trailing states
gamma=0.99; tau=0.01; lrA=1e-3; lrC=1e-3; batch=128; polDelay=2; noise=0.2; psmooth=0.15;

buf = struct('o',[],'a',[],'r',[],'o2',[],'ctx',[],'ctx2',[]); cap=20000; bn=0;
curve = nan(episodes,1); updc=0;

for ep=1:episodes
    env.reset(seed*1000+ep);
    o = env.getObs();
    epR = 0;
    for t=1:T
        ctx = repmat(mean(o,1), N, 1);
        a = actAll(actor, o);
        a = a + noise*randn(N,dAct); a=max(min(a,1),-1);
        act.dv = a(:,1:2)*2; act.dP = a(:,3)*1;
        act.chan = greedyChan(env);  act.assign = [];
        [o2,r,m] = env.step(act);
        ctx2 = repmat(mean(o2,1), N, 1);
        % store per-agent transitions
        for i=1:N
            bn=bn+1; idx=mod(bn-1,cap)+1;
            buf.o(idx,:)=o(i,:); buf.a(idx,:)=a(i,:); buf.r(idx,1)=r(i);
            buf.o2(idx,:)=o2(i,:); buf.ctx(idx,:)=ctx(i,:); buf.ctx2(idx,:)=ctx2(i,:);
        end
        o=o2; epR=epR+mean(r);
        % ---- TD3 update ----
        if bn>=batch
            ns=min(bn,cap); ix=randi(ns,[batch,1]);
            bo=buf.o(ix,:); ba=buf.a(ix,:); br=buf.r(ix,:);
            bo2=buf.o2(ix,:); bctx=buf.ctx(ix,:); bctx2=buf.ctx2(ix,:);
            % target action w/ smoothing
            a2 = actForward(tActor, dlarray(bo2','CB'));
            a2 = a2 + psmooth*randn(size(a2),'like',a2); a2=max(min(a2,1),-1);
            q1t = criticForward(tc1,[bo2';extractdata(a2);bctx2']);
            q2t = criticForward(tc2,[bo2';extractdata(a2);bctx2']);
            y = br' + gamma*min(extractdata(q1t),extractdata(q2t));
            % critic grads
            [g1,l1]=dlfeval(@criticLoss,c1,[bo';ba';bctx'],y);
            [g2,~ ]=dlfeval(@criticLoss,c2,[bo';ba';bctx'],y);
            [c1,optC1]=adamStep(c1,g1,optC1,lrC,updc+1);
            [c2,optC2]=adamStep(c2,g2,optC2,lrC,updc+1);
            updc=updc+1;
            if mod(updc,polDelay)==0
                gA=dlfeval(@actorLoss,actor,c1,dlarray(bo','CB'),bctx');
                [actor,optA]=adamStep(actor,gA,optA,lrA,updc);
                tActor=soft(tActor,actor,tau); tc1=soft(tc1,c1,tau); tc2=soft(tc2,c2,tau);
            end
        end
    end
    curve(ep)=epR;
    if mod(ep,5)==0, fprintf('  ep %2d/%d  R=%.2f  delay=%.0f PDR=%.2f thr=%.1f\n',...
            ep,episodes,epR,m.delay,m.pdr,m.throughput); end
end

% ---- evaluation: learned vs classical baseline (frozen, 30 eval eps) ----
evL = evalPolicy(@(o,e) actAll(actor,o), cfg, seed, 30, true);
evC = evalPolicy(@(o,e) zeros(size(o,1),3),   cfg, seed, 30, true); % baseline: no move/power change
fprintf('\n[%s] LEARNED : delay=%.1f PDR=%.2f thr=%.1f fair=%.2f\n',scenario,evL.delay,evL.pdr,evL.throughput,evL.fairness);
fprintf('[%s] CLASSIC : delay=%.1f PDR=%.2f thr=%.1f fair=%.2f\n',scenario,evC.delay,evC.pdr,evC.throughput,evC.fairness);

figure('Visible','off'); plot(movmean(curve,3),'LineWidth',1.5); grid on;
xlabel('episode'); ylabel('mean per-agent return'); title(sprintf('TD3 learning curve (Scen %s)',scenario));
saveas(gcf, fullfile(resdir,sprintf('learncurve_%s.png',scenario)));
out.curve=curve; out.learned=evL; out.classic=evC;
save(fullfile(resdir,sprintf('train_quick_%s.mat',scenario)),'out','cfg');
end

% ================= helpers =================
function net = mlp(inD, hid, outD, outT)
L = featureInputLayer(inD,'Name','in');
for h=hid, L=[L; fullyConnectedLayer(h); reluLayer]; end %#ok<AGROW>
L=[L; fullyConnectedLayer(outD)];
if strcmp(outT,'tanh'), L=[L; tanhLayer]; end
net = dlnetwork(L);
end
function a = actAll(net,o)     % deterministic action, plain arrays
a = extractdata(actForward(net, dlarray(o','CB')))';
end
function y = actForward(net,X), y = forward(net,X); end
function q = criticForward(net,X), q = forward(net,dlarray(X,'CB')); end
function [g,loss]=criticLoss(net,X,y)
q = forward(net,dlarray(X,'CB'));
loss = mean((q - dlarray(y','CB')).^2,'all');
g = dlgradient(loss,net.Learnables);
end
function g=actorLoss(actor,critic,O,ctx)
a = forward(actor,O);
X = [stripdims(O); stripdims(a); dlarray(ctx)];
q = forward(critic, dlarray(X,'CB'));
loss = -mean(q,'all');
g = dlgradient(loss, actor.Learnables);
end
function [net,st]=adamStep(net,g,st,lr,t)
[net,st.m,st.v]=adamupdate(net,g,st.m,st.v,t,lr);
end
function tnet=soft(tnet,net,tau)
tnet.Learnables.Value=cellfun(@(a,b) (1-tau)*a+tau*b, tnet.Learnables.Value, net.Learnables.Value,'uni',0);
end
function ch=greedyChan(env)   % least-occupied channel per UAV (interference-aware baseline)
K=env.cfg.K; N=env.cfg.N; occ=zeros(1,K);
for k=1:K, occ(k)=sum(env.uavChan==k); end
[~,best]=min(occ); ch=best*ones(N,1);
end
function ev=evalPolicy(polfn, cfg, seed, neps, greedyC)
env=NetworkSim(cfg,seed); N=cfg.N; acc=zeros(neps,6);
for e=1:neps
    env.reset(seed*7+e); o=env.getObs();
    mm=zeros(60,6);
    for t=1:60
        a=polfn(o,env); a=max(min(a,1),-1);
        act.dv=a(:,1:2)*2; act.dP=a(:,3);
        if greedyC, act.chan=greedyChan(env); else, act.chan=[]; end
        act.assign=[];
        [o,~,m]=env.step(act);
        mm(t,:)=[m.delay,m.pdr,m.throughput,m.fairness,m.cci,m.loadVar];
    end
    acc(e,:)=mean(mm,1);
end
mu=mean(acc,1);
ev.delay=mu(1); ev.pdr=mu(2); ev.throughput=mu(3); ev.fairness=mu(4); ev.cci=mu(5); ev.loadVar=mu(6);
end
