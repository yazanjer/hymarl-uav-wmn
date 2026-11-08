function R = uwmn_stats(Data, methods, mn, curves)
%UWMN_STATS  Assemble mean/SD + paired Wilcoxon (Proposed vs each) + rank-biserial
%   into the result struct consumed by make_report* (thesis Sec.3.5.2.5).
S=size(Data,1); nMet=numel(mn); pIdx=find(strcmp(methods,'Proposed'));
Stats=struct();
for k=1:nMet
  mu=squeeze(mean(Data(:,:,k),1)); sd=squeeze(std(Data(:,:,k),0,1));
  pv=nan(1,numel(methods)); W=nan(1,numel(methods)); rb=nan(1,numel(methods));
  for mi=1:numel(methods)
    if mi==pIdx, continue; end
    x=Data(:,pIdx,k); y=Data(:,mi,k);
    if S>=2 && any(x~=y)
      [p,~,st]=signrank(x,y); pv(mi)=p;
      if isfield(st,'signedrank'), W(mi)=st.signedrank; end
      d=x-y; rb(mi)=(sum(d>0)-sum(d<0))/max(sum(d~=0),1);
    end
  end
  Stats.(mn{k})=struct('mean',mu,'sd',sd,'wilcoxon_p',pv,'W',W,'rankbiserial',rb);
end
R.Data=Data; R.methods=methods; R.metricNames=mn; R.Stats=Stats; R.curves=curves;
end
