function cfg = uwmn_config(scenario, N, K)
%UWMN_CONFIG  Central configuration for the UAV-WMN hierarchical MARL framework.
%   cfg = UWMN_CONFIG(scenario, N, K) returns a struct of physical-layer,
%   MAC-layer, scenario, and training constants shared IDENTICALLY across the
%   proposed methods and the B1/B2 benchmarks, so that all comparisons are fair
%   (thesis Sec. 3.3.1, 3.5). scenario in {'A','B','C'}; N = #UAVs; K = #channels.
%
%   All physical constants follow thesis Eq. (3.2), (3.9) and IEEE 802.11/802.15.4.
%
%   Reference: Bahaa (2026), "Hierarchical MARL for UAV-Assisted WMN", Ch.3.

if nargin < 1 || isempty(scenario), scenario = 'A'; end
if nargin < 2 || isempty(N),        N = 10;         end
if nargin < 3 || isempty(K),        K = 3;          end

cfg = struct();
cfg.scenario = scenario;
cfg.N = N;                 % number of UAV mesh routers
cfg.K = K;                 % number of orthogonal channels

% ---- Physical layer (Friis + log-normal shadowing, Eq. 3.2) ----
cfg.fc        = 2.4e9;     % carrier frequency (Hz)
cfg.c0        = 3e8;       % speed of light
cfg.d0        = 1;         % reference distance (m)
cfg.PLd0      = 40.0;      % reference path loss at d0 (dB) ~ 2.4 GHz, 1 m
cfg.plExp     = 2.8;       % path-loss exponent s (urban-ish default, per scenario override)
cfg.shadowSig = 4.0;       % log-normal shadowing std (dB)
cfg.Pmax      = 20.0;      % max Tx power (dBm)
cfg.Pmin      = 0.0;       % min Tx power (dBm)
cfg.Prx_sens  = -85.0;     % receiver sensitivity Pmin (dBm) -> edge exists if Prx>=this
cfg.N0_dBm    = -95.0;     % noise floor (dBm) over channel bandwidth
cfg.BW        = 20e6;      % per-channel bandwidth (Hz)
cfg.kappa_adj = 0.30;      % channel-overlap coefficient kappa for adjacent channels
cfg.Gamma_min = 3.0;       % minimum SINR target (dB) (QoS constraint, Eq. 3.10)

% ---- MAC layer (Bianchi CSMA/CA, IEEE 802.11) ----
cfg.CWmin     = 16;        % minimum contention window
cfg.mBackoff  = 6;         % number of backoff stages
cfg.slotTime  = 9e-6;      % PHY slot (s)
cfg.SIFS      = 16e-6;
cfg.DIFS      = 34e-6;
cfg.Tpacket   = 512*8;     % payload bits per packet (512 B)
cfg.headerBits= 40*8;      % MAC+PHY header bits

% ---- Queueing (M/M/1 per node) ----
cfg.arrival_scale = 1.0;   % scales offered load (per-scenario override)

% ---- UAV kinematics ----
cfg.dt        = 0.1;       % worker control step (s), ~10 Hz (Layer 2)
cfg.vmax      = 20.0;      % max UAV speed (m/s)
cfg.z_uav     = 100.0;     % UAV altitude (m), within 50-300 m regulatory band
cfg.dP_step   = 1.0;       % max power increment per step (dB)

% ---- Hierarchy timescales (Sec. 3.3.2) ----
cfg.dTM       = 10;        % Manager epoch = 10 worker steps (Layer 2 grouping)
cfg.dTS       = 10;        % Slow (Layer 1) epoch = 10 fast steps
cfg.fastPerSlow = 10;

% ---- Reward weights ----
% Objective 1 (Eq. 3.8):  r = a1 e^{-T} - b1 ln P - w1 I[P>Pmax] - lam1 H({T})
cfg.a1 = 0.6; cfg.b1 = 0.3; cfg.w1 = 0.1; cfg.lam1 = 0.05;
% Objective 2 (Eq. 3.13): + Gaussian shaping (Eq. 3.14)
cfg.a2 = 0.6; cfg.b2 = 0.3; cfg.g2 = 0.3; cfg.d2 = 0.1; cfg.mu2 = 0.2;
cfg.sigP = 6.0; cfg.sigG = 3.0;   % Gaussian shaping widths (Eq. 3.14)
% Objective 3 (Eq. 3.15): throughput - lam_var Var(L) - lam_HO HO
cfg.lam3_var = 0.4; cfg.lam3_HO = 0.2; cfg.cHO = 0.05;

% ---- Discount factors ----
cfg.gamma1 = 0.99; cfg.gamma2 = 0.99; cfg.gamma3_slow = 0.95; cfg.gamma3_fast = 0.99;

% ---- Scenario-specific overrides (Sec. 3.5.2.2) ----
switch upper(scenario)
    case 'A'   % Disaster recovery: 1000x1000, U=200, RPGM 1-3 m/s, bursty, urban
        % UAV air-to-ground/air-to-air links are predominantly LoS at 100 m
        % altitude, so the effective path-loss exponent is lower than a ground
        % NLoS urban value (thesis Sec.1.8 LoS/NLoS modelling). s=2.5 yields
        % ~330 m range so the N-UAV mesh can form a connected multi-hop backbone.
        cfg.area = 1000; cfg.U = 200; cfg.mobility = 'RPGM';
        cfg.userSpeed = [1 3]; cfg.traffic = 'burst'; cfg.plExp = 2.5;
        cfg.shadowSig = 5.0; cfg.arrival_scale = 1.2;
    case 'B'   % Temporary event: 500x500, U=400, Manhattan 0.5-1.5, video, open-air
        cfg.area = 500;  cfg.U = 400; cfg.mobility = 'MANHATTAN';
        cfg.userSpeed = [0.5 1.5]; cfg.traffic = 'video'; cfg.plExp = 2.5;
        cfg.shadowSig = 3.0; cfg.arrival_scale = 1.6;
    case 'C'   % Rural: 2000x2000, U=80, RWP 0-30 m/s, low volume, free-space
        cfg.area = 2000; cfg.U = 80;  cfg.mobility = 'RWP';
        cfg.userSpeed = [0 30]; cfg.traffic = 'low'; cfg.plExp = 2.2;
        cfg.shadowSig = 2.0; cfg.arrival_scale = 0.7;
    otherwise
        error('uwmn_config:scenario','Unknown scenario %s', scenario);
end

% Down-scale user count for the reduced-but-rigorous local protocol (Sec.5),
% preserving the density ORDERING across scenarios (documented in PROGRESS.md).
cfg.U = round(cfg.U * 0.25);

% ---- Derived linear-domain constants ----
cfg.N0_lin  = 10.^((cfg.N0_dBm-30)/10);   % noise power (W)
cfg.Pmax_lin= 10.^((cfg.Pmax-30)/10);
end
