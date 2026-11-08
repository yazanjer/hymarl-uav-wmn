classdef NetworkSim < handle
%NETWORKSIM  Shared UAV-WMN network simulator (thesis Sec. 3.3.1, 3.5.1).
%   Time-varying heterogeneous graph G(t)=(V,E,C): N UAV mesh routers (3D),
%   U ground users (2D), fixed gateways. Physical layer = Friis + log-normal
%   shadowing (Eq.3.2); multi-channel SINR/CCI (Eq.3.9); MAC = Bianchi CSMA/CA;
%   per-node M/M/1 queueing. This ONE simulator is used identically by the
%   proposed methods and the B1/B2 benchmarks so all comparisons are fair.
%
%   Action struct fields (any subset; unspecified -> sensible default):
%     dv     : N x 2 horizontal velocity increments (m/s change)
%     dP     : N x 1 Tx-power increments (dB)
%     chan   : N x 1 channel index per UAV in {1..K}
%     assign : U x 1 user->UAV assignment (0 => auto nearest-router)
%
%   Author: implementation for Bahaa (2026) thesis, MATLAB R2025a.

    properties
        cfg
        rs                 % RandStream (reproducible)
        % --- dynamic state ---
        uavPos             % N x 3
        uavVel             % N x 2
        uavP               % N x 1 Tx power (dBm)
        uavChan            % N x 1 channel
        gwPos              % G x 3
        userPos            % U x 3
        userDemand         % U x 1 (Mbps)
        assign             % U x 1 user->UAV
        prevAssign         % U x 1
        state              % mobility bookkeeping struct
        t                  % step counter
        % --- cached network computation ---
        net                % struct with last computeNetwork() outputs
    end

    methods
        function obj = NetworkSim(cfg, seed)
            obj.cfg = cfg;
            if nargin < 2, seed = 0; end
            obj.rs = RandStream('threefry','Seed',seed);
        end

        function obs = reset(obj, seed)
            c = obj.cfg;
            if nargin >= 2 && ~isempty(seed)
                obj.rs = RandStream('threefry','Seed',seed);
            end
            A = c.area;
            % Gateways at two edge midpoints (backhaul anchors)
            obj.gwPos = [A*0.5, 0, 0; A*0.5, A, 0];
            % UAVs initialised as a spread coverage grid so the mesh has baseline
            % multi-hop connectivity to the edge gateways (documented deviation from
            % pure-centroid start, for tractable training; see PROGRESS.md).
            g = ceil(sqrt(c.N));
            [gx,gy] = meshgrid(linspace(0.15,0.85,g), linspace(0.05,0.95,g));
            pts = [gx(:), gy(:)]*A; pts = pts(1:c.N,:);
            obj.uavPos = [pts, c.z_uav*ones(c.N,1)];
            obj.uavVel = zeros(c.N,2);
            obj.uavP   = (c.Pmax-6)*ones(c.N,1);        % start mid-high power
            obj.uavChan= mod((0:c.N-1)', c.K) + 1;      % round-robin channels
            % Users
            obj.userPos = [A*rand(obj.rs,[c.U,2]), zeros(c.U,1)];
            switch c.traffic   % offered demand per user (Mbps)
                case 'burst',  base = 0.8;  jit = 1.5;
                case 'video',  base = 2.0;  jit = 1.0;
                otherwise,     base = 0.2;  jit = 0.3;
            end
            obj.userDemand = c.arrival_scale * max(0.02, base + jit*rand(obj.rs,[c.U,1]));
            % Mobility bookkeeping
            st = struct();
            st.userSpd = c.userSpeed(1) + diff(c.userSpeed)*rand(obj.rs,[c.U,1]);
            st.userWP  = A*rand(obj.rs,[c.U,2]);
            nGroups = max(2, round(c.U/15));
            st.userGroup = randi(obj.rs, nGroups, [c.U,1]);
            st.groupCentroid = A*rand(obj.rs,[nGroups,2]);
            st.groupRef = A*rand(obj.rs,[nGroups,2]);
            st.userAxis = randi(obj.rs,2,[c.U,1]);
            st.userDir  = sign(randn(obj.rs,[c.U,1])); st.userDir(st.userDir==0)=1;
            obj.state = st;
            obj.assign = zeros(c.U,1); obj.prevAssign = zeros(c.U,1);
            obj.t = 0;
            obj.computeNetwork();
            obs = obj.getObs();
        end

        function [obs, reward, metrics, info] = step(obj, action)
            c = obj.cfg;
            % ---- apply UAV actions ----
            if isfield(action,'dv') && ~isempty(action.dv)
                obj.uavVel = obj.uavVel + action.dv;
                sp = sqrt(sum(obj.uavVel.^2,2));
                over = sp > c.vmax;
                obj.uavVel(over,:) = obj.uavVel(over,:) .* (c.vmax./sp(over));
            end
            obj.uavPos(:,1:2) = obj.uavPos(:,1:2) + obj.uavVel*c.dt;
            obj.uavPos(:,1:2) = min(max(obj.uavPos(:,1:2),0), c.area);
            if isfield(action,'dP') && ~isempty(action.dP)
                obj.uavP = min(max(obj.uavP + action.dP, c.Pmin), c.Pmax);
            end
            if isfield(action,'chan') && ~isempty(action.chan)
                obj.uavChan = min(max(round(action.chan),1),c.K);
            end
            % ---- assignment ----
            obj.prevAssign = obj.assign;
            if isfield(action,'assign') && ~isempty(action.assign) && any(action.assign>0)
                obj.assign = action.assign;
            else
                obj.assign = obj.nearestRouter();
            end
            % ---- advance user mobility ----
            [obj.userPos, obj.state] = uwmn_mobility(obj.userPos, obj.state, c, obj.rs);
            obj.t = obj.t + 1;
            % ---- recompute network ----
            obj.computeNetwork();
            metrics = obj.computeMetrics();
            reward  = obj.computeReward(metrics, action);
            obs = obj.getObs();
            info.net = obj.net;
        end

        function a = nearestRouter(obj)
            % Assign each user to the UAV with strongest mean received power.
            d = pdist2(obj.userPos(:,1:2), obj.uavPos(:,1:2));
            [~, a] = min(d, [], 2);
        end

        function computeNetwork(obj)
            c = obj.cfg; N = c.N;
            P = obj.uavPos;
            % ----- UAV-UAV path gains (for mesh backhaul) -----
            Guu = zeros(N,N); PrxUU = -inf(N,N);
            for i=1:N
                [prx, g] = uwmn_channel(repmat(P(i,:),N,1), P, obj.uavP(i), c, obj.rs);
                PrxUU(:,i) = prx; Guu(:,i) = g;    % from i to all
            end
            Guu(1:N+1:end) = 0; PrxUU(1:N+1:end) = -inf;
            adj = PrxUU >= obj.Prx_sens_eff();      % directed edge exists
            adj = adj | adj';                       % symmetric mesh link

            % ----- UAV-gateway gains -----
            nGW = size(obj.gwPos,1); Gug = zeros(N,nGW); PrxUG=-inf(N,nGW);
            for gw=1:nGW
                [prx,g] = uwmn_channel(P, repmat(obj.gwPos(gw,:),N,1), obj.uavP, c, obj.rs);
                PrxUG(:,gw)=prx; Gug(:,gw)=g;
            end
            % ----- shortest path (hop) from each UAV to nearest gateway -----
            % Build graph with UAV mesh + gateway sink; edge weight = ETX proxy.
            gwReach = PrxUG >= obj.Prx_sens_eff();     % N x nGW
            hop = obj.bfsToGateway(adj, gwReach);      % N x 1 hop count (Inf if none)

            % ----- SINR per UAV on its channel (Eq. 3.9) with CCI -----
            % Interference at a UAV's receiver = co-channel UAVs within range.
            sinr = zeros(N,1); rate = zeros(N,1);
            Plin = 10.^((obj.uavP-30)/10);
            for i=1:N
                % desired: best next-hop toward gateway (strongest neighbour with lower hop)
                cand = find(adj(i,:) & (hop(:)' < hop(i)));
                if isempty(cand)
                    [~,nh] = max(Gug(i,:)); sig = Plin(i)*Gug(i,nh);
                     chSelf = obj.uavChan(i);
                else
                    [~,ix] = max(Guu(i,cand)); nh = cand(ix);
                    sig = Plin(i)*Guu(i,nh); chSelf = obj.uavChan(i);
                end
                % interference: other UAVs sharing/overlapping channel, in range of i
                interf = 0;
                for k=1:N
                    if k==i, continue; end
                    if PrxUU(i,k) < obj.Prx_sens_eff()-10, continue; end % out of interfering range
                    kap = obj.overlap(obj.uavChan(k), chSelf);
                    if kap>0
                        interf = interf + Plin(k)*Guu(i,k)*kap;
                    end
                end
                sinr(i) = sig / (c.N0_lin + interf);
                rate(i) = c.BW*log2(1+max(sinr(i),0)) / 1e6;   % Mbps
            end

            % ----- MAC contention per channel -----
            Dmac = zeros(N,1);
            for i=1:N
                nc = sum(obj.uavChan==obj.uavChan(i));   % stations on same channel
                Dmac(i) = uwmn_macdelay(nc, c);
            end

            % ----- per-UAV offered load from assigned users (M/M/1) -----
            load = zeros(N,1);
            for i=1:N
                load(i) = sum(obj.userDemand(obj.assign==i));
            end
            % ----- M/M/1 queueing in packets/second (correct units) -----
            muPps  = max(rate,1e-3)*1e6 / c.Tpacket;         % service rate (pkt/s)
            lamPps = load*1e6 / c.Tpacket;                   % offered load (pkt/s)
            rho    = min(lamPps ./ muPps, 0.98);             % capped utilisation
            Wqueue = 1 ./ max(muPps.*(1-rho), 1e-3);         % M/M/1 sojourn (s)

            % ----- end-to-end delay (ms) = (mac+queue+prop) summed over hops -----
            prop   = sqrt(sum((P - obj.gwPos(1,:)).^2,2)) / c.c0;   % s (one-hop approx)
            connected = isfinite(hop);
            hopEff = hop; hopEff(~connected) = 8;                   % disconnected -> long path
            Dmax   = 2000;                                          % ms timeout cap
            e2e = (Dmac + Wqueue + prop) .* max(hopEff,1) * 1000;   % ms
            e2e(~connected) = Dmax;
            e2e = min(e2e, Dmax);

            % ----- SINR adequacy, delivery fraction, throughput -----
            sinrOK = min(max(sinr ./ 10.^(c.Gamma_min/10), 0), 1); % 1 if above target
            servedFrac = min(1, muPps ./ max(lamPps,1e-6));        % overload -> drop
            deliv = double(connected) .* sinrOK .* servedFrac;
            deliv = min(max(deliv,0),1);

            % store
            n.adj=adj; n.hop=hop; n.sinr=sinr; n.rate=rate; n.Dmac=Dmac;
            n.load=load; n.mu=muPps; n.rateMbps=rate; n.rho=rho; n.Dqueue=Wqueue; n.e2e=e2e;
            n.deliv=deliv; n.PrxUU=PrxUU; n.Guu=Guu; n.Gug=Gug;
            n.deliveredMbps = deliv .* load;
            obj.net = n;
        end

        function s = Prx_sens_eff(obj)
            s = obj.cfg.Prx_sens;
        end

        function k = overlap(obj, c1, c2)
            if c1==c2, k=1;
            elseif abs(c1-c2)==1, k=obj.cfg.kappa_adj;
            else, k=0; end
        end

        function hop = bfsToGateway(obj, adj, gwReach)
            N=size(adj,1); hop=inf(N,1);
            frontier = find(any(gwReach,2));
            hop(frontier)=1; d=1;
            while ~isempty(frontier)
                d=d+1; nxt=[];
                for f=frontier(:)'
                    nb=find(adj(f,:) & ~isfinite(hop'));
                    hop(nb)=d; nxt=[nxt,nb]; %#ok<AGROW>
                end
                frontier=unique(nxt);
            end
        end

        function obs = getObs(obj)
            % Unified per-UAV observation matrix (N x dObs).
            c=obj.cfg; N=c.N; n=obj.net;
            gwdir = obj.gwPos(1,1:2) - obj.uavPos(:,1:2);
            gwdir = gwdir ./ max(sqrt(sum(gwdir.^2,2)),1e-6);
            occ = zeros(N,c.K);
            for k=1:c.K, occ(:,k) = sum(obj.uavChan==k)/N; end
            obs = [ obj.uavPos(:,1:2)/c.area, ...          % 2 normalized pos
                    obj.uavVel/c.vmax, ...                 % 2 velocity
                    (obj.uavP-c.Pmin)/(c.Pmax-c.Pmin), ... % 1 power
                    min(n.rho,1), ...                      % 1 utilisation
                    tanh(n.e2e/100), ...                   % 1 delay proxy
                    gwdir, ...                             % 2 gateway direction
                    min(sum(n.adj,2)/N,1), ...             % 1 degree
                    tanh(10*log10(max(n.sinr,1e-3))/20), ... % 1 SINR
                    min(n.load/10,1), ...                  % 1 load
                    occ ];                                 % K occupancy
        end

        function d = obsDim(obj)
            d = 12 + obj.cfg.K;
        end

        function obs = getObsAug(obj)
            % Graph message-passing with attention (Alg. 3.2 / 3.4 encoder, one
            % layer). Attention weights = row-normalised link gains over the
            % 1-hop mesh neighbourhood (a self-coordination prior, borrowed from
            % B2 Eq.11-13). Returns [own features, attention-aggregated neighbour
            % features] so a plain actor MLP can exploit coordination.
            base = obj.getObs();
            n = obj.net;
            W = n.Guu; W(~n.adj) = 0;
            W = W ./ (sum(W,2) + 1e-9);       % attention (softmax-like) over neighbours
            agg = W * base;                   % message aggregation
            obs = [base, agg];
        end

        function d = obsDimAug(obj)
            d = 2*obj.obsDim();
        end

        function a = loadBalanceAssign(obj)
            % Two-timescale slow assignment (Obj.3): greedily assign each user to
            % the reachable UAV with the lowest projected load -> lower Var(L).
            c = obj.cfg;
            d = pdist2(obj.userPos(:,1:2), obj.uavPos(:,1:2));
            reachR = 0.30*c.area;                 % coarse reach radius
            load = zeros(c.N,1); a = zeros(c.U,1);
            [~, order] = sort(obj.userDemand, 'descend');
            for u = order'
                cand = find(d(u,:) < reachR);
                if isempty(cand), [~,cand] = min(d(u,:)); end
                [~, ix] = min(load(cand)); j = cand(ix);
                a(u) = j; load(j) = load(j) + obj.userDemand(u);
            end
        end

        function a = loadBalanceAssignHyst(obj, h)
            % Load-aware greedy assignment with hysteresis (article revision): a user
            % keeps its current router if that router is within reach and its
            % accumulated load exceeds the least-loaded candidate by at most
            % theta = h * mean user demand (Mbps). h = 0 reduces to loadBalanceAssign.
            c = obj.cfg;
            d = pdist2(obj.userPos(:,1:2), obj.uavPos(:,1:2));
            reachR = 0.30*c.area; theta = h*mean(obj.userDemand);
            load = zeros(c.N,1); a = zeros(c.U,1);
            [~, order] = sort(obj.userDemand, 'descend');
            for u = order'
                cand = find(d(u,:) < reachR);
                if isempty(cand), [~,cand] = min(d(u,:)); end
                [~, ix] = min(load(cand)); j = cand(ix);
                p = obj.assign(u);
                if p > 0 && any(cand == p) && load(p) <= load(j) + theta, j = p; end
                a(u) = j; load(j) = load(j) + obj.userDemand(u);
            end
        end

        function ch = greedyChannels(obj)
            % Classical greedy channel assignment (article revision): UAVs ordered by
            % number of interferers; each takes the channel that minimises the
            % overlap-weighted interference power exchanged with already-assigned UAVs.
            c = obj.cfg; N = c.N; K = c.K; n = obj.net;
            Pr = 10.^((n.PrxUU-30)/10); Pr(~isfinite(Pr)) = 0;   % received power (W), (rx,tx)
            W = Pr + Pr';                                          % symmetric interference weight
            inter = n.PrxUU >= obj.Prx_sens_eff()-10; inter = inter | inter';
            [~, order] = sort(sum(inter,2), 'descend');
            ch = zeros(N,1);
            for i = order'
                cost = zeros(1,K);
                for k = 1:K
                    for j = find(ch' > 0)
                        cost(k) = cost(k) + W(i,j)*obj.overlap(k, ch(j));
                    end
                end
                [~, ch(i)] = min(cost);
            end
        end

        function m = computeMetrics(obj)
            c=obj.cfg; n=obj.net;
            deliv = n.deliv;
            m.delay      = mean(n.e2e);                              % ms
            m.pdr        = mean(deliv);                              % [0,1]
            m.throughput = sum(n.deliveredMbps);                    % Mbps
            act = false(c.N,c.K); for i=1:c.N, act(i,obj.uavChan(i))=true; end
            m.chanUtil   = mean(min(n.rho,1));                       % fraction
            cci=0;
            Plin=10.^((obj.uavP-30)/10);
            for i=1:c.N
                for k=1:c.N
                    if k==i, continue; end
                    kap=obj.overlap(obj.uavChan(k),obj.uavChan(i));
                    if kap>0, cci=cci+Plin(k)*n.Guu(i,k)*kap; end
                end
            end
            m.cci        = 10*log10(max(cci,1e-15))+30;             % dBm
            m.specEff    = m.throughput*1e6 / (c.K*c.BW);           % bps/Hz
            th = n.deliveredMbps;
            m.fairness   = (sum(th)^2) / (c.N*sum(th.^2)+1e-9);     % Jain
            m.loadVar    = var(n.load);                             % load variance
            m.handover   = sum(obj.assign~=obj.prevAssign)/max(c.U,1); % per-user
            totBits = m.throughput*1e6*c.dt;
            energyJ = sum(10.^((obj.uavP-30)/10))*c.dt + 0.1*c.N*c.dt;
            m.energyPerBit = energyJ / max(totBits,1);              % J/bit
            m.sinr_mean  = mean(10*log10(max(n.sinr,1e-3)));
        end

        function r = computeReward(obj, m, action)
            % Integrated per-UAV reward blending Obj1/2/3 terms (Eq.3.8/3.13/3.15).
            c=obj.cfg; n=obj.net; N=c.N;
            Pw = obj.uavP;
            % Objective 1: delay + power + fairness (Eq. 3.8)
            Tnorm = n.e2e/100;
            r1 = c.a1*exp(-Tnorm) - c.b1*log(max(Pw-c.Pmin+1,1))/log(c.Pmax) ...
                 - c.w1*double(Pw>=c.Pmax);
            % Objective 2: interference + Gaussian shaping (Eq. 3.13-3.14)
            sinr_dB = 10*log10(max(n.sinr,1e-3));
            Gshape = exp( -((Pw-c.Pmin).^2)/(2*c.sigP^2) - ((c.Gamma_min - sinr_dB).^2)/(2*c.sigG^2) );
            collide = double(n.sinr < 10.^(c.Gamma_min/10));
            r2 = -c.g2*tanh(max(sinr_dB,0)/20) - c.d2*collide + c.mu2*Gshape;
            r2 = r2 + 0.4*n.rate/max(mean(n.rate),1e-3);   % throughput incentive
            % Objective 3: load fairness (global, broadcast)
            r3 = -c.lam3_var * (n.load - mean(n.load)).^2 / (var(n.load)+1);
            % global fairness/throughput bonus
            gb = 0.3*m.fairness + 0.2*tanh(m.throughput/max(sum(n.rate),1e-3));
            r = r1 + r2 + r3 + gb;
            r = r(:);
        end
    end
end
