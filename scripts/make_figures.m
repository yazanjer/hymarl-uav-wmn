function make_figures(htag, ftag, ctag)
%MAKE_FIGURES  Detailed result figures for Chapters 4-6 and the supplementary
%   appendix, generated from logged runs only (no hand-entered values).
%   htag : headline tag  (results/headline_<htag>.mat, needs R.curvesAll, R.trace)
%   ftag : fair tag      (results/<ftag>.mat,          needs R.curvesAll, R.trace)
%   Also reads results/campaign.mat for scenarios, N-sweep, ablation, sensitivity.
%   Figures are written to output/figures/*.pdf (vector).
%
%   make_figures('A8log','fairAlog')
if nargin<1, htag='A8log'; end
if nargin<2, ftag='fairAlog'; end
if nargin<3, ctag='campaign'; end
here=fileparts(mfilename('fullpath'));
resdir=fullfile(here,'..','results');
figdir=fullfile(here,'..','output','figures');
if ~exist(figdir,'dir'), mkdir(figdir); end
RH=getfield(load(fullfile(resdir,sprintf('headline_%s.mat',htag))),'R');
RF=getfield(load(fullfile(resdir,sprintf('%s.mat',ftag))),'R');
haveCamp=exist(fullfile(resdir,[ctag '.mat']),'file')==2;
if haveCamp, camp=getfield(load(fullfile(resdir,[ctag '.mat'])),'camp'); end

mn=RH.metricNames; mi=@(x) find(strcmp(mn,x));
lab=containers.Map( ...
 {'delay','pdr','throughput','chanUtil','cci','specEff','fairness','loadVar','handover','energyPerBit','sinr_mean'}, ...
 {'Delay (ms)','PDR','Throughput (Mbps)','Channel util.','CCI (dBm)','Spec. eff. (bps/Hz)', ...
  'Jain fairness','Load variance','Handover rate','Energy/bit','Mean SINR (dB)'});
% +1: higher is better, -1: lower is better
dirn=containers.Map( ...
 {'delay','pdr','throughput','chanUtil','cci','specEff','fairness','loadVar','handover','energyPerBit','sinr_mean'}, ...
 {-1,+1,+1,+1,-1,+1,+1,-1,-1,-1,+1});

% ================= CHAPTER 4 =================
% F1 convergence, mean +/- 95% CI over seeds (headline learners)
f=newfig(560,380); plotCurvesCI(RH.curvesAll,RH.methods);
xlabel('Training episode'); ylabel('Mean per-agent return');
title(sprintf('Convergence, mean \\pm 95%% CI over %d seeds (Scenario A, N=10)',size(RH.curvesAll,1)));
save_fig(f,figdir,'ch4_convergence_ci');

% F2 routing metrics over an evaluation episode
f=newfig(900,300);
show={'delay','throughput','pdr'};
for j=1:3
  subplot(1,3,j); plotTraceCI(RH.trace(:,:,:,mi(show{j})),RH.methods);
  xlabel('Time step'); ylabel(lab(show{j})); title(lab(show{j}));
  if j==1, legend(RH.methods,'Location','best'); end
end
save_fig(f,figdir,'ch4_timeseries_routing');

% F3 equal-spectrum: per-seed distributions
f=newfig(900,520);
show={'delay','throughput','pdr','fairness','loadVar','handover'};
for j=1:6
  subplot(2,3,j); boxes(squeeze(RF.Data(:,:,mi(show{j}))),RF.methods);
  title(lab(show{j}));
end
save_fig(f,figdir,'ch4_fair_box');

% F4 equal-spectrum: Wilcoxon p-value heatmap (Proposed vs each configuration)
show={'delay','throughput','pdr','fairness','loadVar','handover','cci','sinr_mean'};
others=find(~strcmp(RF.methods,'Proposed'));
P=nan(numel(others),numel(show)); RB=P;
for j=1:numel(show)
  st=RF.Stats.(show{j}); P(:,j)=st.wilcoxon_p(others); RB(:,j)=st.rankbiserial(others);
end
alpha=0.05/numel(others);
f=newfig(760,300);
imagesc(-log10(P)); colormap(gca,flipud(gray)*0.85+0.15); cb=colorbar; ylabel(cb,'-log_{10} p');
set(gca,'XTick',1:numel(show),'XTickLabel',cellfun(@(k) lab(k),show,'UniformOutput',false), ...
        'YTick',1:numel(others),'YTickLabel',RF.methods(others),'TickLength',[0 0]);
xtickangle(30);
for r=1:size(P,1), for c=1:size(P,2)
  s=sprintf('%.3f',P(r,c)); if P(r,c)<alpha, s=[s '*']; end
  if -log10(P(r,c))>1.2, tc='w'; else, tc='k'; end
  text(c,r,s,'HorizontalAlignment','center','FontSize',8,'Color',tc);
end, end
title(sprintf('Wilcoxon p, Proposed vs each configuration (* : p < %.4f)',alpha));
save_fig(f,figdir,'ch4_fair_pheatmap');

% F5 equal-spectrum convergence
f=newfig(560,380); plotCurvesCI(RF.curvesAll,RF.methods);
xlabel('Training episode'); ylabel('Mean per-agent return');
title(sprintf('Equal-spectrum convergence, mean \\pm 95%% CI (%d seeds)',size(RF.curvesAll,1)));
save_fig(f,figdir,'ch4_fair_convergence');

% ================= CHAPTER 5 =================
% F6 interference metrics over an evaluation episode
f=newfig(760,300);
show={'sinr_mean','cci'};
for j=1:2
  subplot(1,2,j); plotTraceCI(RH.trace(:,:,:,mi(show{j})),RH.methods);
  xlabel('Time step'); ylabel(lab(show{j})); title(lab(show{j}));
  if j==1, legend(RH.methods,'Location','best'); end
end
save_fig(f,figdir,'ch5_timeseries_interf');

% F7 K-sensitivity heatmap (column-normalised, raw values annotated)
if haveCamp
  sk=camp.sensK; smn=sk.metrics;
  show={'throughput','delay','pdr','cci','sinr_mean','specEff','fairness'};
  M=nan(numel(sk.K),numel(show));
  for j=1:numel(show), M(:,j)=sk.Data(:,strcmp(smn,show{j})); end
  rowlab=arrayfun(@(k) sprintf('K = %d',k),sk.K,'UniformOutput',false);
  f=newfig(760,300); scoreHeatmap(M,show,rowlab,lab,dirn);
  title('Sensitivity to the number of channels K (score: 1 = best per metric)');
  save_fig(f,figdir,'ch5_sensK_heatmap');
end

% ================= CHAPTER 6 =================
% F8 assignment metrics over an evaluation episode
f=newfig(760,300);
show={'loadVar','handover'};
for j=1:2
  subplot(1,2,j); plotTraceCI(RH.trace(:,:,:,mi(show{j})),RH.methods,1+strcmp(show{j},'handover'));
  xlabel('Time step'); ylabel(lab(show{j})); title(lab(show{j}));
  if j==1, legend(RH.methods,'Location','best'); end
end
save_fig(f,figdir,'ch6_timeseries_load');

if haveCamp
  % F9 scenario x method heatmaps (A from headline, B and C from campaign)
  Rs={RH,camp.B10,camp.C10}; scen={'A (disaster)','B (event)','C (rural)'};
  show={'throughput','delay','loadVar'};
  f=newfig(980,300);
  for j=1:3
    M=nan(3,numel(RH.methods));
    for s=1:3, M(s,:)=squeeze(mean(Rs{s}.Data(:,:,strcmp(Rs{s}.metricNames,show{j})),1)); end
    subplot(1,3,j); scoreHeatmapRows(M,RH.methods,scen,dirn(show{j}));
    title(lab(show{j}));
  end
  save_fig(f,figdir,'ch6_scenario_heatmap');

  % F10 ablation heatmap: % change relative to the full model
  ab=camp.ablation; amn=ab.metrics; An=ab.names;
  show={'delay','throughput','pdr','fairness','loadVar','handover','cci'};
  M=nan(numel(An),numel(show));
  for j=1:numel(show), M(:,j)=squeeze(mean(ab.Data(:,:,strcmp(amn,show{j})),1)); end
  rel=100*(M-M(1,:))./abs(M(1,:));
  good=rel.*cellfun(@(k) dirn(k),show);          % >0 = improvement
  pretty=containers.Map({'Full','noLyap','noAug','noLoadBal','noMultiChan'}, ...
    {'Full model','- Lyapunov','- graph attention','- load balancing','- multi-channel'});
  rl=cellfun(@(x) pretty(x),An,'UniformOutput',false);
  f=newfig(820,320);
  imagesc(max(min(good,100),-100)); colormap(gca,divmap()); caxis([-100 100]);
  cb=colorbar; ylabel(cb,'% change (+ = better)');
  set(gca,'XTick',1:numel(show),'XTickLabel',cellfun(@(k) lab(k),show,'UniformOutput',false), ...
          'YTick',1:numel(rl),'YTickLabel',rl,'TickLength',[0 0]); xtickangle(30);
  for r=1:size(rel,1), for c=1:size(rel,2)
    text(c,r,sprintf('%+.0f%%',rel(r,c)),'HorizontalAlignment','center','FontSize',8);
  end, end
  title('Ablation: change relative to the full model (Scenario A, N=10)');
  save_fig(f,figdir,'ch6_ablation_heatmap');
end

% ================= SUPPLEMENTARY (appendix) =================
% S1 method x metric score heatmap (headline), raw means annotated
show={'delay','throughput','pdr','fairness','loadVar','handover','cci','sinr_mean','specEff','chanUtil'};
M=nan(numel(RH.methods),numel(show));
for j=1:numel(show), M(:,j)=squeeze(mean(RH.Data(:,:,mi(show{j})),1))'; end
f=newfig(900,300); scoreHeatmap(M,show,RH.methods,lab,dirn);
title('Headline comparison, Scenario A (score: 1 = best per metric; raw means shown)');
save_fig(f,figdir,'app_metric_heatmap');

% S2 radar chart of normalised scores (learning methods + classic)
show={'delay','throughput','pdr','fairness','loadVar','sinr_mean'};
M=nan(numel(RH.methods),numel(show));
for j=1:numel(show), M(:,j)=squeeze(mean(RH.Data(:,:,mi(show{j})),1))'; end
Sc=scoreMat(M,show,dirn);
f=newfig(520,480); radar(Sc,cellfun(@(k) lab(k),show,'UniformOutput',false),RH.methods);
title('Normalised performance profile (Scenario A)');
save_fig(f,figdir,'app_radar');

% S3 all eleven metrics over an evaluation episode (headline)
f=newfig(1000,760);
for k=1:numel(mn)
  subplot(4,3,k); plotTraceCI(RH.trace(:,:,:,k),RH.methods,1+strcmp(mn{k},'handover'));
  title(lab(mn{k})); xlabel('Time step');
  if k==1, legend(RH.methods,'Location','best'); end
end
save_fig(f,figdir,'app_timeseries_all');

% S4 equal-spectrum metrics over an evaluation episode
f=newfig(900,520);
show={'delay','throughput','pdr','loadVar','handover','sinr_mean'};
for j=1:6
  subplot(2,3,j); plotTraceCI(RF.trace(:,:,:,mi(show{j})),RF.methods,1+strcmp(show{j},'handover'));
  title(lab(show{j})); xlabel('Time step');
  if j==1, legend(RF.methods,'Location','best'); end
end
save_fig(f,figdir,'app_fair_timeseries');

% S5 per-seed training curves (small multiples)
learners=find(~all(isnan(RH.curvesAll(:,:,1)),1));
f=newfig(260*numel(learners),260);
for j=1:numel(learners)
  subplot(1,numel(learners),j); hold on;
  C=squeeze(RH.curvesAll(:,learners(j),:));
  plot(C','Color',[0.6 0.6 0.6]); plot(mean(C,1),'k','LineWidth',1.6);
  grid on; title(RH.methods{learners(j)}); xlabel('Episode');
  if j==1, ylabel('Return'); end
end
save_fig(f,figdir,'app_seed_curves');

% S6 N-sweep distributions (throughput)
if haveCamp
  cells={camp.A5,RH,camp.A20}; Ns=[5 10 20];
  f=newfig(900,300);
  for j=1:3
    subplot(1,3,j); boxes(squeeze(cells{j}.Data(:,:,strcmp(cells{j}.metricNames,'throughput'))),RH.methods);
    title(sprintf('Throughput (Mbps), N = %d',Ns(j)));
  end
  save_fig(f,figdir,'app_nsweep_box');
end

% S7 equal-spectrum violin plots (MATLAB R2024b+)
if exist('violinplot','file')==2 || exist('violinplot','builtin')==5
  f=newfig(900,300); show={'loadVar','fairness','handover'};
  for j=1:3
    subplot(1,3,j); Y=squeeze(RF.Data(:,:,mi(show{j})));
    violinplot(categorical(repmat(RF.methods,size(Y,1),1)),Y); grid on;
    title(lab(show{j}));
  end
  save_fig(f,figdir,'app_fair_violin');
else
  fprintf('make_figures: violinplot unavailable, app_fair_violin skipped\n');
end

fprintf('make_figures: wrote figures to %s\n',figdir);
end

% ======================================================================
function f=newfig(w,h)
f=figure('Visible','off','Position',[100 100 w h],'Color','w');
end

function save_fig(f,figdir,name)
fn=fullfile(figdir,[name '.pdf']);
if exist('exportgraphics','file')==2 || exist('exportgraphics','builtin')==5
  exportgraphics(f,fn,'ContentType','vector');
else
  set(f,'PaperPositionMode','auto'); pos=get(f,'Position');
  set(f,'PaperUnits','points','PaperSize',pos(3:4)); print(f,fn,'-dpdf');
end
close(f);
end

function t=tcrit(n)
% two-sided 95% Student-t critical value for n samples
if n<2, t=NaN; return; end
if exist('tinv','file')==2, t=tinv(0.975,n-1); return; end
tab=[12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 2.201 2.179 2.160 2.145 2.131];
if n-1<=numel(tab), t=tab(n-1); else, t=1.96; end
end

function plotCurvesCI(C,methods)
% C: S x M x E
hold on; cols=lines(numel(methods)); h=[]; lg={};
for m=1:numel(methods)
  X=squeeze(C(:,m,:)); if isvector(X), X=X(:)'; end
  if all(isnan(X(:))), continue; end
  if size(X,2)>3, for s=1:size(X,1), X(s,:)=movmean(X(s,:),3); end, end
  mu=mean(X,1); ci=tcrit(size(X,1))*std(X,0,1)/sqrt(size(X,1)); e=1:numel(mu);
  if size(X,1)>1
    fill([e fliplr(e)],[mu-ci fliplr(mu+ci)],cols(m,:),'FaceAlpha',0.18,'EdgeColor','none');
  end
  h(end+1)=plot(e,mu,'LineWidth',1.6,'Color',cols(m,:)); lg{end+1}=methods{m}; %#ok<AGROW>
end
grid on; legend(h,lg,'Location','southeast');
end

function plotTraceCI(T,methods,t0)
% T: S x M x Tsteps (x1); t0 = first time step plotted (default 1)
if nargin<3, t0=1; end
hold on; cols=lines(numel(methods)); T=squeeze(T);
if ndims(T)==2, T=reshape(T,[1 size(T)]); end
for m=1:numel(methods)
  X=squeeze(T(:,m,:)); if isvector(X), X=X(:)'; end
  X=X(:,t0:end);
  mu=mean(X,1); ci=tcrit(size(X,1))*std(X,0,1)/sqrt(size(X,1)); t=t0-1+(1:numel(mu));
  if size(X,1)>1
    fill([t fliplr(t)],[mu-ci fliplr(mu+ci)],cols(m,:),'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
  end
  plot(t,mu,'LineWidth',1.4,'Color',cols(m,:));
end
grid on; xlim([t(1) t(end)]);
end

function boxes(Y,labels)
% Y: S x M per-seed values
if exist('boxplot','file')==2
  boxplot(Y,'Labels',labels); grid on;
else
  hold on;
  for m=1:size(Y,2)
    q=prctile(Y(:,m),[0 25 50 75 100]);
    rectangle('Position',[m-0.25 q(2) 0.5 max(q(4)-q(2),eps)]);
    plot([m-0.25 m+0.25],[q(3) q(3)],'r'); plot([m m],[q(1) q(2)],'k--'); plot([m m],[q(4) q(5)],'k--');
  end
  set(gca,'XTick',1:size(Y,2),'XTickLabel',labels); xlim([0.5 size(Y,2)+0.5]); grid on;
end
xtickangle(30);
end

function Sc=scoreMat(M,keys,dirn)
% column-wise min-max score in [0,1], 1 = best
Sc=nan(size(M));
for j=1:size(M,2)
  v=M(:,j); lo=min(v); hi=max(v);
  if hi>lo, s=(v-lo)/(hi-lo); else, s=ones(size(v)); end
  if dirn(keys{j})<0, s=1-s; end
  Sc(:,j)=s;
end
end

function scoreHeatmap(M,keys,rowlab,lab,dirn)
Sc=scoreMat(M,keys,dirn);
imagesc(Sc); colormap(gca,parula); caxis([0 1]); cb=colorbar; ylabel(cb,'score');
set(gca,'XTick',1:numel(keys),'XTickLabel',cellfun(@(k) lab(k),keys,'UniformOutput',false), ...
        'YTick',1:numel(rowlab),'YTickLabel',rowlab,'TickLength',[0 0]); xtickangle(30);
for r=1:size(M,1), for c=1:size(M,2)
  if Sc(r,c)>0.6, tc='k'; else, tc='w'; end
  text(c,r,fmtv(M(r,c)),'HorizontalAlignment','center','FontSize',8,'Color',tc);
end, end
end

function scoreHeatmapRows(M,collab,rowlab,d)
% M: scenarios x methods, scored within each scenario (row)
Sc=nan(size(M));
for r=1:size(M,1)
  v=M(r,:); lo=min(v); hi=max(v);
  if hi>lo, s=(v-lo)/(hi-lo); else, s=ones(size(v)); end
  if d<0, s=1-s; end
  Sc(r,:)=s;
end
imagesc(Sc); colormap(gca,parula); caxis([0 1]);
set(gca,'XTick',1:numel(collab),'XTickLabel',collab,'YTick',1:numel(rowlab),'YTickLabel',rowlab,'TickLength',[0 0]);
xtickangle(30);
for r=1:size(M,1), for c=1:size(M,2)
  if Sc(r,c)>0.6, tc='k'; else, tc='w'; end
  text(c,r,fmtv(M(r,c)),'HorizontalAlignment','center','FontSize',8,'Color',tc);
end, end
end

function s=fmtv(v)
if abs(v)>=100, s=sprintf('%.0f',v); elseif abs(v)>=10, s=sprintf('%.1f',v);
elseif abs(v)>=0.1, s=sprintf('%.2f',v); else, s=sprintf('%.3g',v); end
end

function cm=divmap()
n=64; a=linspace(0,1,n)';
neg=[0.70+0.30*a, 0.25+0.75*a, 0.25+0.75*a];        % red -> white
pos=[1-0.75*a, 1-0.45*a, 1-0.25*a];                  % white -> blue
cm=[neg; pos(2:end,:)];
end

function radar(Sc,axlab,names)
n=size(Sc,2); th=linspace(0,2*pi,n+1); th=th(1:end-1)+pi/2;
hold on; axis equal off; cols=lines(size(Sc,1));
for g=0.25:0.25:1
  plot(g*cos([th th(1)]),g*sin([th th(1)]),'Color',[0.85 0.85 0.85]);
end
for k=1:n
  plot([0 cos(th(k))],[0 sin(th(k))],'Color',[0.85 0.85 0.85]);
  text(1.15*cos(th(k)),1.15*sin(th(k)),axlab{k},'HorizontalAlignment','center','FontSize',9);
end
h=zeros(1,size(Sc,1));
for m=1:size(Sc,1)
  r=Sc(m,:); h(m)=plot(r([1:end 1]).*cos([th th(1)]),r([1:end 1]).*sin([th th(1)]),'-o', ...
     'Color',cols(m,:),'LineWidth',1.6,'MarkerFaceColor',cols(m,:),'MarkerSize',4);
end
legend(h,names,'Location','southoutside','Orientation','horizontal');
xlim([-1.4 1.4]); ylim([-1.35 1.35]);
end
