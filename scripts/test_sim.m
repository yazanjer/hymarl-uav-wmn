function test_sim()
%TEST_SIM  Unit / sanity test for the shared NetworkSim (thesis Sec. 3.5.1).
%   Runs each scenario for a short random-action rollout, prints metric ranges,
%   and saves a sanity figure to results/. Verifies no NaN/Inf and plausible values.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','src'));
resdir = fullfile(here,'..','results'); if ~exist(resdir,'dir'), mkdir(resdir); end

scen = {'A','B','C'}; N=10; K=3; T=60;
fprintf('=== NetworkSim sanity test ===\n');
figure('Visible','off','Position',[100 100 1100 700]);
for si=1:3
    cfg = uwmn_config(scen{si}, N, K);
    env = NetworkSim(cfg, 42);
    env.reset(42);
    M = zeros(T, 6); % delay,pdr,thr,cci,fair,loadvar
    for t=1:T
        act.dv  = 2*(rand(N,2)-0.5);
        act.dP  = (rand(N,1)-0.5);
        act.chan= randi(K,N,1);
        [~,~,m] = env.step(act);
        M(t,:) = [m.delay, m.pdr, m.throughput, m.cci, m.fairness, m.loadVar];
    end
    assert(all(isfinite(M(:))), 'Non-finite metric in scenario %s', scen{si});
    fprintf('Scen %s: delay=%.1f ms  PDR=%.2f  thr=%.1f Mbps  CCI=%.1f dBm  fair=%.2f  Lvar=%.2f\n',...
        scen{si}, mean(M(:,1)),mean(M(:,2)),mean(M(:,3)),mean(M(:,4)),mean(M(:,5)),mean(M(:,6)));
    subplot(3,3,si);   plot(M(:,1)); title(sprintf('Scen %s delay (ms)',scen{si})); grid on;
    subplot(3,3,si+3); plot(M(:,3)); title('throughput (Mbps)'); grid on;
    subplot(3,3,si+6); plot(M(:,5)); title('Jain fairness'); grid on; ylim([0 1]);
end
saveas(gcf, fullfile(resdir,'sanity_metrics.png'));
fprintf('Saved %s\n', fullfile(resdir,'sanity_metrics.png'));
fprintf('=== PASS ===\n');
end
