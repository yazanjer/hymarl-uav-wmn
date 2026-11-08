function export_ns3(outdir)
%EXPORT_NS3  Export the trained actors and the evaluation-episode traces of the headline
%  cell (Scenario A, N=10) for the packet-level ns-3 validation (evaluation only).
%   - policies/<Method>_s<seed>.json : actor weights + method configuration
%     (B1, B2, Proposed from results/models/A8log; Hybrid retrained with the
%     run_hybrid protocol rng(1000*s+50) and checked against results/hybrid.mat)
%   - traces/s<seed>_e<ep>.bin : user demands and user positions at every step of the
%     evaluation episodes env.reset(seed*7+ep), ep=1..30 (identical for every method,
%     because the simulator's random stream does not depend on the actions)
%  export_ns3('../ns3_export')
if nargin<1, outdir='ns3_export'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results');
if ~exist(fullfile(outdir,'policies'),'dir'), mkdir(fullfile(outdir,'policies')); end
if ~exist(fullfile(outdir,'traces'),'dir'), mkdir(fullfile(outdir,'traces')); end
S=8; T=60; nEval=30; t0=tic;

% ---- saved headline agents ----
meth={'B1','B2','Proposed'};
for s=1:S
  for m=1:numel(meth)
    L=load(fullfile(resdir,'models','A8log',sprintf('seed%d_%s.mat',s,meth{m})),'ag');
    writePolicy(L.ag, uwmn_methodcfg(meth{m}), meth{m}, ...
        fullfile(outdir,'policies',sprintf('%s_s%d.json',meth{m},s)));
  end
end
fprintf('headline policies exported (%.1f min)\n',toc(t0)/60);

% ---- hybrid: retrain with the run_hybrid protocol and verify against hybrid.mat ----
H=load(fullfile(resdir,'hybrid.mat'),'hyb'); ref=H.hyb.A10.Data;
md=fullfile(resdir,'models','hybridA10'); if ~exist(md,'dir'), mkdir(md); end
chk=zeros(S,3);
for s=1:S
  aCfg=uwmn_methodcfg('Proposed',struct('chanMode','greedy'));
  cfg=uwmn_config('A',10,aCfg.K);
  rng(1000*s+50,'twister');
  [ag,~]=uwmn_train(aCfg,cfg,s,50,T);
  ev=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
  chk(s,:)=[s, ev.delay, ref(s,1)];
  save(fullfile(md,sprintf('seed%d_Hybrid.mat',s)),'ag','-v7');
  writePolicy(ag, aCfg, 'Hybrid', fullfile(outdir,'policies',sprintf('Hybrid_s%d.json',s)));
  fprintf('hybrid seed %d: delay %.4f (hybrid.mat %.4f) (%.1f min)\n',s,ev.delay,ref(s,1),toc(t0)/60);
end
writematrix(chk, fullfile(outdir,'hybrid_check.csv'));

% ---- traces of the evaluation episodes ----
cfg=uwmn_config('A',10,1); env=NetworkSim(cfg,1);
for s=1:S
  for e=1:nEval
    env.reset(s*7+e);
    P=zeros(T+1,cfg.U,2); P(1,:,:)=reshape(env.userPos(:,1:2),[1 cfg.U 2]);
    ea=struct('dv',zeros(cfg.N,2),'dP',zeros(cfg.N,1));
    for t=1:T
      env.step(ea); P(t+1,:,:)=reshape(env.userPos(:,1:2),[1 cfg.U 2]);
    end
    fid=fopen(fullfile(outdir,'traces',sprintf('s%d_e%d.bin',s,e)),'w');
    fwrite(fid,[cfg.U T],'int32');
    fwrite(fid,env.userDemand,'float64');
    for t=1:T+1, fwrite(fid,squeeze(P(t,:,:)),'float64'); end   % U x 2, column-major (x then y)
    fclose(fid);
  end
end
fprintf('traces exported; total %.1f min\n',toc(t0)/60);
end

function writePolicy(ag, aCfg, name, fn)
Lr=ag.actor.Learnables; layers={};
names=unique(Lr.Layer,'stable');
for i=1:numel(names)
  W=extractdata(Lr.Value{strcmp(Lr.Layer,names{i}) & strcmp(Lr.Parameter,'Weights')});
  b=extractdata(Lr.Value{strcmp(Lr.Layer,names{i}) & strcmp(Lr.Parameter,'Bias')});
  layers{end+1}=struct('W',double(W),'b',double(b(:)')); %#ok<AGROW>
end
p=struct('method',name,'aug',logical(aCfg.aug),'K',aCfg.K,'powerMode',aCfg.powerMode,...
  'useLyap',logical(aCfg.useLyap),'kL',aCfg.kL,'Ttarget',aCfg.Ttarget,'dPmax',aCfg.dPmax,...
  'dS',aCfg.dS,'chanMode',aCfg.chanMode,'assign',aCfg.assign,'hyst',aCfg.hyst,...
  'dObs',ag.dObs,'actDim',ag.actDim,'contDim',ag.contDim,'chanN',ag.chanN,'powN',ag.powN);
p.layers=layers;
fid=fopen(fn,'w'); fwrite(fid,jsonencode(p)); fclose(fid);
end
