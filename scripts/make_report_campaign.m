function make_report_campaign(htag, ctag)
if nargin<1, htag='A8'; end
if nargin<2, ctag='campaign'; end
%MAKE_REPORT_CAMPAIGN  Turn campaign.mat (+ headline_A8.mat for N=10) into
%   thesis artefacts: scalability table/figure, cross-scenario table + Friedman,
%   ablation table, and K / reward-weight sensitivity tables/figure. Real data only.
here=fileparts(mfilename('fullpath'));
resdir=fullfile(here,'..','results');
thesis=fullfile(here,'..','output'); if ~exist(thesis,'dir'), mkdir(thesis); end;
figdir=fullfile(thesis,'figures'); if ~exist(figdir,'dir'), mkdir(figdir); end
camp=getfield(load(fullfile(resdir,[ctag '.mat'])),'camp');
RH=getfield(load(fullfile(resdir,sprintf('headline_%s.mat',htag))),'R');
methods=RH.methods; mn=RH.metricNames;
mi=@(x) find(strcmp(mn,x));
MC=find(strcmp(methods,'Classic')); MB1=find(strcmp(methods,'B1'));
MB2=find(strcmp(methods,'B2')); MP=find(strcmp(methods,'Proposed'));

% ================= Scalability (Scenario A, N in {5,10,20}) =================
Ns=[5 10 20]; cells={camp.A5, RH, camp.A20};
thr=zeros(3,4); dly=zeros(3,4);
for i=1:3
  thr(i,:)=squeeze(mean(cells{i}.Data(:,:,mi('throughput')),1))';
  dly(i,:)=squeeze(mean(cells{i}.Data(:,:,mi('delay')),1))';
end
slope=zeros(1,4); for m=1:4, p=polyfit(Ns',thr(:,m),1); slope(m)=p(1); end
f=figure('Visible','off','Position',[100 100 560 380]); hold on; cols=lines(4);
for m=1:4, plot(Ns,thr(:,m),'-o','Color',cols(m,:),'LineWidth',1.6,'MarkerFaceColor',cols(m,:)); end
grid on; xlabel('Number of UAVs $N$','Interpreter','latex'); ylabel('Throughput (Mbps)');
legend(methods,'Location','northwest'); title('Scalability: throughput vs N (Scenario A)');
exportgraphics(f,fullfile(figdir,'ch4_scalability.pdf'),'ContentType','vector');
fid=fopen(fullfile(thesis,'ch4_scalability.tex'),'w');
fprintf(fid,'%% auto-generated from %s.mat + headline_%s.mat\n',ctag,htag);
fprintf(fid,['\\begin{table}[htbp]\\centering\n\\caption{Scalability sweep on Scenario~A: '...
 'mean throughput (Mbps) versus the number of UAVs $N$, with the scalability slope '...
 '$\\partial\\text{throughput}/\\partial N$ (Mbps per UAV, linear fit over $N\\in\\{5,10,20\\}$).}\n'...
 '\\label{tab:ch4_scal}\\small\n\\begin{tabular}{lcccc}\n\\toprule\n'...
 'Method & $N{=}5$ & $N{=}10$ & $N{=}20$ & Slope (Mbps/UAV)\\\\\n\\midrule\n']);
for m=[MC MB1 MB2 MP]
  nm=methods{m}; if m==MP, nm=['\textbf{' nm '}']; end
  fprintf(fid,'%s & %.1f & %.1f & %.1f & %.2f \\\\\n',nm,thr(1,m),thr(2,m),thr(3,m),slope(m));
end
fprintf(fid,'\\bottomrule\n\\end{tabular}\n\\end{table}\n'); fclose(fid);

% ================= Cross-scenario (A,B,C at N=10) =================
scen={'A','B','C'}; Rs={RH, camp.B10, camp.C10};
met3={'delay','throughput','fairness'}; lab3={'Delay (ms)','Throughput (Mbps)','Fairness'};
fid=fopen(fullfile(thesis,'ch6_scenarios.tex'),'w');
fprintf(fid,'%% auto-generated from %s.mat + headline_%s.mat\n',ctag,htag);
fprintf(fid,['\\begin{table}[htbp]\\centering\n\\caption{End-to-end evaluation across the '...
 'three deployment scenarios at $N=10$ (mean over seeds). Bold marks the best mean in each row; '...
 '$^\\dagger$ marks a benchmark whose difference from Proposed is significant (paired exact Wilcoxon, '...
 'Bonferroni $\\alpha=0.0167$). The Friedman tests across the three scenarios are reported in the text.}\n\\label{tab:ch6_scen}\\small\n'...
 '\\begin{tabular}{llccc}\n\\toprule\nMetric & Scenario & B1 & B2 & Proposed\\\\\n\\midrule\n']);
FrData=struct(); lowBetter=[true false false];
for mm=1:3
  col=mi(met3{mm}); M=nan(3,3); % scenarios x [B1 B2 Proposed]
  for si=1:3
    v=squeeze(mean(Rs{si}.Data(:,:,col),1));
    M(si,:)=[v(MB1) v(MB2) v(MP)];
    pv=Rs{si}.Stats.(met3{mm}).wilcoxon_p; pv=[pv(MB1) pv(MB2) NaN];
    if lowBetter(mm), [~,b]=min(M(si,:)); else, [~,b]=max(M(si,:)); end
    cs=cell(1,3);
    for q=1:3
      c=sprintf('%.2f',M(si,q)); if q==b, c=['\textbf{' c '}']; end
      if ~isnan(pv(q)) && pv(q)<0.05/3, c=[c '$^\dagger$']; end
      cs{q}=c;
    end
    if si==1, rowlab=lab3{mm}; else, rowlab=''; end
    fprintf(fid,'%s & %s & %s & %s & %s\\\\\n',rowlab,scen{si},cs{1},cs{2},cs{3});
  end
  FrData.(met3{mm})=M;
  if mm<3, fprintf(fid,'\\midrule\n'); end
end
fprintf(fid,'\\bottomrule\n\\end{tabular}\n\\end{table}\n'); fclose(fid);
% Friedman across methods (blocks = scenarios) for delay & throughput
try
  fp_d=friedman(FrData.delay,1,'off'); fp_t=friedman(FrData.throughput,1,'off');
catch, fp_d=NaN; fp_t=NaN; end
fid=fopen(fullfile(thesis,'ch6_friedman.tex'),'w');
vb={'does not reject','rejects'};
fprintf(fid,'Across the three scenarios, the Friedman test %s equality of the three methods on delay ($p=%.4f$) and %s it on throughput ($p=%.4f$), at $\\alpha=0.05$.',vb{(fp_d<0.05)+1},fp_d,vb{(fp_t<0.05)+1},fp_t);
fclose(fid);

% ================= Ablation (Scenario A, N=10) =================
ab=camp.ablation; An=ab.names; AD=ab.Data; amn=ab.metrics;
ami=@(x) find(strcmp(amn,x));
fid=fopen(fullfile(thesis,'ch6_ablation.tex'),'w');
fprintf(fid,'%% auto-generated ablation of the proposed framework from %s.mat\n',ctag);
fprintf(fid,['\\begin{table}[htbp]\\centering\n\\caption{Ablation of the proposed framework on '...
 'Scenario~A ($N=10$, mean$\\pm$SD over %d seeds). Each row removes one component; '...
 '$\\Delta$Thr.\\ is the change in mean throughput relative to the full model; '...
 '$^\\dagger$ marks a difference from the full model that is significant '...
 '(paired exact Wilcoxon, Bonferroni $\\alpha=0.0125$).}\n\\label{tab:ch6_abl}\\small\n'...
 '\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}\n\\toprule\nVariant & Delay (ms) & Throughput (Mbps) & '...
 'Fairness & Load var. & Handover & $\\Delta$Thr.\\\\\n\\midrule\n'],size(AD,1));
thrFull=mean(AD(:,1,ami('throughput')));
pretty=containers.Map({'Full','noLyap','noAug','noLoadBal','noMultiChan'},...
 {'Full model','$-$ delay-driven power corr.','$-$ neighbour aggregation','$-$ load balancing','$-$ multi-channel ($K{=}1$)'});
amet={'delay','throughput','fairness','loadVar','handover'}; afmt={'%.1f','%.1f','%.2f','%.1f','%.3f'};
for v=1:numel(An)
  nm=pretty(An{v}); if v==1, nm=['\textbf{' nm '}']; end
  fprintf(fid,'%s',nm);
  for q=1:numel(amet)
    x=AD(:,v,ami(amet{q})); c=[sprintf(afmt{q},mean(x)) '$\pm$' sprintf(afmt{q},std(x))];
    if v>1
      pv=ab.Stats.(amet{q}).wilcoxon_p(v); if ~isnan(pv) && pv<0.05/4, c=[c '$^\dagger$']; end
    end
    fprintf(fid,' & %s',c);
  end
  if v==1, fprintf(fid,' & --- \\\\\n');
  else, fprintf(fid,' & %+.0f\\%% \\\\\n',100*(mean(AD(:,v,ami('throughput')))-thrFull)/thrFull); end
end
fprintf(fid,'\\bottomrule\n\\end{tabular}}\n\\end{table}\n'); fclose(fid);

% ================= Sensitivity: K and lambda =================
sk=camp.sensK; sl=camp.sensL; smn=sk.metrics; smi=@(x) find(strcmp(smn,x));
f=figure('Visible','off','Position',[100 100 760 340]);
subplot(1,2,1); yyaxis left; plot(sk.K, sk.Data(:,smi('throughput')),'-o','LineWidth',1.6); ylabel('Throughput (Mbps)');
yyaxis right; plot(sk.K, sk.Data(:,smi('cci')),'-s','LineWidth',1.6); ylabel('CCI (dBm)');
grid on; xlabel('Number of channels K'); title('Sensitivity to K');
subplot(1,2,2); plot(sl.lam, sl.Data(:,smi('loadVar')),'-o','LineWidth',1.6); hold on;
plot(sl.lam, sl.Data(:,smi('handover'))*1000,'-s','LineWidth',1.6);
grid on; xlabel('\lambda_3^{var} (load-variance weight)'); ylabel('Load var. / 10^{3}\times handover');
legend('Load variance','1000\times handover','Location','best'); title('Sensitivity to \lambda_3^{var}');
exportgraphics(f,fullfile(figdir,'ch5_sensitivity.pdf'),'ContentType','vector');
fid=fopen(fullfile(thesis,'ch5_sensitivity.tex'),'w');
fprintf(fid,'%% auto-generated sensitivity tables from %s.mat\n',ctag);
DS=sk.DataSeeds; nS=size(DS,1);
fprintf(fid,['\\begin{table}[htbp]\\centering\n\\caption{Sensitivity of the proposed framework to '...
 'the number of channels $K$ (Scenario~A, $N=10$; mean$\\pm$SD over %d seeds, each trained and evaluated at the stated $K$).}\n\\label{tab:ch5_sensK}\\small\n'...
 '\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccc}\n\\toprule\n$K$ & Throughput (Mbps) & Delay (ms) & CCI (dBm) & Spec.\\ eff.\\ (bps/Hz) & Mean SINR (dB)\\\\\n\\midrule\n'],nS);
km={'throughput','delay','cci','specEff','sinr_mean'}; kf={'%.1f','%.1f','%.1f','%.2f','%.2f'};
for i=1:numel(sk.K)
  fprintf(fid,'%d',sk.K(i));
  for q=1:numel(km), x=DS(:,i,smi(km{q})); fprintf(fid,[' & ' kf{q} '$\\pm$' kf{q}],mean(x),std(x)); end
  fprintf(fid,' \\\\\n');
end
fprintf(fid,'\\bottomrule\n\\end{tabular}}\n\\end{table}\n');
DL=sl.DataSeeds;
fprintf(fid,['\\begin{table}[htbp]\\centering\n\\caption{Sensitivity of the proposed framework to '...
 'the load-variance reward weight $\\lambda_3^{var}$ (Scenario~A, $N=10$, $K=3$; mean$\\pm$SD over %d seeds).}\n\\label{tab:ch5_sensL}\\small\n'...
 '\\begin{tabular}{lcccc}\n\\toprule\n$\\lambda_3^{var}$ & Throughput (Mbps) & Delay (ms) & Load var. & Handover\\\\\n\\midrule\n'],size(DL,1));
lm={'throughput','delay','loadVar','handover'}; lf={'%.1f','%.1f','%.1f','%.3f'};
for i=1:numel(sl.lam)
  fprintf(fid,'%.1f',sl.lam(i));
  for q=1:numel(lm), x=DL(:,i,smi(lm{q})); fprintf(fid,[' & ' lf{q} '$\\pm$' lf{q}],mean(x),std(x)); end
  fprintf(fid,' \\\\\n');
end
fprintf(fid,'\\bottomrule\n\\end{tabular}\n\\end{table}\n'); fclose(fid);
fprintf('make_report_campaign: wrote scalability, scenarios, ablation, sensitivity artefacts.\n');
end
