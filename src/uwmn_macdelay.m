function [Dmac, Tserv, ptx, pcol] = uwmn_macdelay(nContend, cfg)
%UWMN_MACDELAY  CSMA/CA MAC access delay via Bianchi's Markov-chain model,
%   combined with an M/M/1-style service-time expansion (thesis Sec. 3.3.1).
%   nContend : number of stations contending on the shared channel (>=1).
%   Returns:
%     Dmac  : mean MAC access delay (s)
%     Tserv : mean channel service time for one successful frame (s)
%     ptx   : per-slot transmission probability tau
%     pcol  : conditional collision probability p
%
%   Bianchi fixed point:
%     p   = 1 - (1 - tau)^(n-1)
%     tau = 2(1-2p) / [ (1-2p)(W+1) + p W (1-(2p)^m) ]

n = max(1, round(nContend));
W = cfg.CWmin; m = cfg.mBackoff;

% Solve the Bianchi fixed point by simple iteration.
tau = 0.02;
for it = 1:200
    p   = 1 - (1 - tau)^(n-1);
    p2m = (2*p)^m;
    denom = (1-2*p)*(W+1) + p*W*(1 - p2m);
    tau_new = 2*(1-2*p) / max(denom, 1e-9);
    tau_new = min(max(tau_new, 1e-6), 0.5);
    if abs(tau_new - tau) < 1e-9, tau = tau_new; break; end
    tau = 0.5*tau + 0.5*tau_new;    % damped update for stability
end
ptx = tau; pcol = p;

% Frame durations
Tframe = (cfg.headerBits + cfg.Tpacket) / (cfg.BW);   % payload+header airtime (approx)
Ts = cfg.DIFS + Tframe + cfg.SIFS;                    % successful slot duration
Tc = cfg.DIFS + Tframe;                               % collision slot duration
Ptr = 1 - (1-tau)^n;                                  % prob at least one Tx
Ps  = n*tau*(1-tau)^(n-1) / max(Ptr, 1e-9);           % prob a Tx is successful

% Expected slot length (Bianchi Eq.)
Eslot = (1-Ptr)*cfg.slotTime + Ptr*Ps*Ts + Ptr*(1-Ps)*Tc;

% Expected number of backoff slots before a successful transmission
% (average backoff window across stages, expanded by retransmissions)
avgW = 0;
for k = 0:m-1
    avgW = avgW + ((2^k*W - 1)/2) * p^k;
end
avgW = avgW * (1-p) / max(1 - p^m, 1e-9);
Dmac  = avgW * Eslot + Ts;     % access delay + one successful frame time
Tserv = Ts;
end
