function run_relog()
%RUN_RELOG  Canonical, fully logged re-run of the Scenario-A headline and the
%   equal-spectrum (fair) comparison with reproducible RNG, followed by
%   regeneration of all thesis tables from the new runs.
%   Logs every seed's training curve and per-time-step evaluation metrics.
%   Same protocol as the original runs: headline S=8, 50 episodes; fair S=8, 80 episodes.
t=tic;
diary(fullfile(fileparts(mfilename('fullpath')),'..','results','run_relog.log'));
run_headline(8,50,'A8log');
run_fair(8,80,'fairAlog');
try
  make_report('A8log');
  make_report_campaign('A8log');
  make_report_fair('fairAlog');
catch e
  fprintf(2,'report step failed: %s\n',e.message);
end
fprintf('run_relog done in %.1f min\n',toc(t)/60);
diary off
end
