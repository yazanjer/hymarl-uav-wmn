function export_ns3_cells(outdir)
%EXPORT_NS3_CELLS  Retrain, verify and export the policies of the fleet-size and scenario
%  cells (A5, A20, B10, C10) for the packet-level ns-3 validation, with the protocols of
%  run_campaign_log (B1, B2, Proposed: rng(1000*s+mi), mi = 2,3,4) and run_hybrid
%  (Hybrid: rng(1000*s+50)). Each retrained agent is evaluated and compared with the
%  stored result (campaign8.mat / hybrid.mat) before export. Also exports the user traces
%  of the evaluation episodes of each cell.
%  export_ns3_cells('../ns3_export')
if nargin<1, outdir='ns3_export'; end
here=fileparts(mfilename('fullpath')); addpath(fullfile(here,'..','src'));
resdir=fullfile(here,'..','results');
S=8; T=60; nEval=30; ep=50; t0=tic;
cells={'A',5,'A5'; 'A',20,'A20'; 'B',10,'B10'; 'C',10,'C10'};
Cm=load(fullfile(resdir,'campaign8.mat'),'camp'); H=load(fullfile(resdir,'hybrid.mat'),'hyb');
meth={'B1','B2','Proposed','Hybrid'}; midx=[2 3 4 50];
% job list
J={}; for c=1:size(cells,1), for s=1:S, for m=1:4, J(end+1,:)={c,s,m}; end, end, end %#ok<AGROW>
nJ=size(J,1); chk=nan(nJ,6);
for c=1:size(cells,1)
  pd=fullfile(outdir,'policies',cells{c,3}); if ~exist(pd,'dir'), mkdir(pd); end
  md=fullfile(resdir,'models',['cells_' cells{c,3}]); if ~exist(md,'dir'), mkdir(md); end
end
usePar = ~isempty(ver('parallel'));
if usePar
  try, p=gcp('nocreate'); if isempty(p), parpool('Processes'); end
  catch, usePar=false; end
end
fprintf('%d jobs, parallel=%d\n',nJ,usePar);
if usePar
  parfor j=1:nJ, chk(j,:)=oneJob(J(j,:),cells,meth,midx,S,T,nEval,ep,outdir,resdir,Cm,H); end
else
  for j=1:nJ, chk(j,:)=oneJob(J(j,:),cells,meth,midx,S,T,nEval,ep,outdir,resdir,Cm,H);
    fprintf('job %d/%d (%.1f min)\n',j,nJ,toc(t0)/60); end
end
writematrix(chk, fullfile(outdir,'cells_check.csv'));
fprintf('max |delay - stored| = %.3g ms\n', max(abs(chk(:,3)-chk(:,4))));
% traces
for c=1:size(cells,1)
  td=fullfile(outdir,'traces',cells{c,3}); if ~exist(td,'dir'), mkdir(td); end
  cfg=uwmn_config(cells{c,1},cells{c,2},1); env=NetworkSim(cfg,1);
  for s=1:S
    for e=1:nEval
      env.reset(s*7+e);
      P=zeros(T+1,cfg.U,2); P(1,:,:)=reshape(env.userPos(:,1:2),[1 cfg.U 2]);
      ea=struct('dv',zeros(cfg.N,2),'dP',zeros(cfg.N,1));
      for t=1:T, env.step(ea); P(t+1,:,:)=reshape(env.userPos(:,1:2),[1 cfg.U 2]); end
      fid=fopen(fullfile(td,sprintf('s%d_e%d.bin',s,e)),'w');
      fwrite(fid,[cfg.U T],'int32'); fwrite(fid,env.userDemand,'float64');
      for t=1:T+1, fwrite(fid,squeeze(P(t,:,:)),'float64'); end
      fclose(fid);
    end
  end
end
fprintf('=== EXPORT CELLS DONE in %.1f min ===\n',toc(t0)/60);
end

function r=oneJob(j,cells,meth,midx,S,T,nEval,ep,outdir,resdir,Cm,H) %#ok<INUSL>
c=j{1}; s=j{2}; m=j{3}; nm=meth{m}; cell=cells{c,3};
if strcmp(nm,'Hybrid'), aCfg=uwmn_methodcfg('Proposed',struct('chanMode','greedy'));
else, aCfg=uwmn_methodcfg(nm); end
cfg=uwmn_config(cells{c,1},cells{c,2},aCfg.K);
rng(1000*s+midx(m),'twister');
ag=uwmn_train(aCfg,cfg,s,ep,T);
ev=uwmn_eval(ag,aCfg,cfg,s,nEval,T);
if strcmp(nm,'Hybrid'), ref=H.hyb.(cell).Data(s,[1 3]);
else, ref=squeeze(Cm.camp.(cell).Data(s,midx(m),[1 3]))'; end
saveAgent(fullfile(resdir,'models',['cells_' cell],sprintf('seed%d_%s.mat',s,nm)),ag);
writePolicy(ag,aCfg,nm,fullfile(outdir,'policies',cell,sprintf('%s_s%d.json',nm,s)));
r=[c*100+s, m, ev.delay, ref(1), ev.throughput, ref(2)];
end

function saveAgent(fn,ag), save(fn,'ag','-v7'); end

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
