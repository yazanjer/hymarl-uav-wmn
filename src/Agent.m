classdef Agent < handle
%AGENT  Unified configurable MARL agent (parameter-shared, CTDE twin critics).
%   One TD3-style core instantiates all methods (thesis Alg. 3.1/3.2/3.3/3.4):
%     .powerMode : 'cont'  -> continuous dP head (Obj.1/B1/Proposed)
%                  'disc'  -> discrete power {-dS,0,+dS} via Gumbel-softmax (B2)
%     .K         : #channels; K>1 adds a Gumbel-softmax channel head (Obj.2/Proposed)
%     .useLyap   : monotonic-ReLU Lyapunov power stabiliser (B1/Proposed, Eq.30-32)
%     .aug       : use graph-attention augmented observations (Proposed/B2)
%   Continuous action layout: [dvx, dvy] (+ dP if 'cont'). Discrete heads appended.
%
%   Author: implementation for Bahaa (2026) thesis, MATLAB R2025a.

    properties
        c            % agent config struct
        actor; c1; c2; tActor; tc1; tc2
        oA; o1; o2   % adam states
        dObs; contDim; chanN; powN; actDim; critIn
        updc = 0
    end

    methods
        function obj = Agent(dObs, aCfg)
            obj.dObs = dObs; obj.c = aCfg;
            obj.contDim = 2 + double(strcmp(aCfg.powerMode,'cont'));  % velocity(+power)
            obj.chanN   = (aCfg.K>1 && ~strcmp(aCfg.chanMode,'greedy')) * aCfg.K;
            obj.powN    = 3 * double(strcmp(aCfg.powerMode,'disc'));
            obj.actDim  = obj.contDim + obj.chanN + obj.powN;
            obj.critIn  = dObs + obj.actDim + dObs;   % obs + action + mean-obs ctx
            obj.actor = Agent.mlp(dObs,[128 128],obj.actDim,'lin');
            obj.c1 = Agent.mlp(obj.critIn,[128 128],1,'lin');
            obj.c2 = Agent.mlp(obj.critIn,[128 128],1,'lin');
            obj.tActor=obj.actor; obj.tc1=obj.c1; obj.tc2=obj.c2;
            z=struct('m',[],'v',[]); obj.oA=z; obj.o1=z; obj.o2=z;
        end

        function [env_act, aVec] = act(obj, obs, env, explore)
            % obs: N x dObs. Returns env action struct + soft action vector (N x actDim).
            N = size(obs,1); cf = obj.c;
            y = Agent.fwd(obj.actor, obs);          % N x actDim (raw)
            cont = tanh(y(:,1:obj.contDim));
            if explore, cont = cont + cf.noise*randn(N,obj.contDim); end
            cont = max(min(cont,1),-1);
            off = obj.contDim;
            % ---- channel head ----
            if obj.chanN>0
                logit = y(:,off+1:off+obj.chanN);
                if explore, s = Agent.gumbel(logit,cf.tau); else, s = logit; end
                chanProb = Agent.softmax(s);
                [~,chanIdx] = max(chanProb,[],2);
                off = off + obj.chanN;
            else
                chanProb = zeros(N,0);
                if strcmp(cf.chanMode,'greedy') && cf.K>1, chanIdx = env.greedyChannels();
                else, chanIdx = mod((0:N-1)',cf.K)+1; end
            end
            % ---- power ----
            if obj.powN>0     % discrete B2: classes {1,2,3} -> {-dS,0,+dS}
                plog = y(:,off+1:off+obj.powN);
                if explore, ps = Agent.gumbel(plog,cf.tau); else, ps=plog; end
                powProb = Agent.softmax(ps); [~,pc] = max(powProb,[],2);
                dP = (pc-2)*cf.dS;
            else              % continuous: last cont dim
                powProb = zeros(N,0);
                dP = cont(:,3)*obj.c.dPmax;
            end
            % ---- Lyapunov monotonic-ReLU stabiliser (Eq.30-32) ----
            if cf.useLyap
                Tproxy = tanh(env.net.e2e/100);         % per-UAV delay error signal
                dP = dP + cf.kL * max(Tproxy - cf.Ttarget, 0);   % monotone in T (nonneg wts)
            end
            env_act.dv   = cont(:,1:2)*2;
            env_act.dP   = dP;
            env_act.chan = chanIdx;
            if cf.assign=='L', env_act.assign = env.loadBalanceAssign();
            elseif cf.assign=='H', env_act.assign = env.loadBalanceAssignHyst(cf.hyst);
            else, env_act.assign=[]; end
            aVec = [cont, chanProb, powProb];
        end

        function push(obj, buf, o,a,r,o2,ctx,ctx2) %#ok<INUSD>
        end

        function update(obj, B)
            % B: struct arrays o,a,r,o2,ctx,ctx2 (rows = samples)
            cf=obj.c; g=cf.gamma; tau=cf.tau_soft;
            a2 = Agent.fwd(obj.tActor, B.o2);
            a2c = tanh(a2(:,1:obj.contDim)) + cf.psmooth*randn(size(a2,1),obj.contDim);
            a2c = max(min(a2c,1),-1);
            off=obj.contDim; parts={a2c};
            if obj.chanN>0, parts{end+1}=Agent.softmax(a2(:,off+1:off+obj.chanN)); off=off+obj.chanN; end
            if obj.powN>0,  parts{end+1}=Agent.softmax(a2(:,off+1:off+obj.powN)); end
            a2v = cat(2, parts{:});
            q1t = Agent.fwd(obj.tc1, [B.o2, a2v, B.ctx2]);
            q2t = Agent.fwd(obj.tc2, [B.o2, a2v, B.ctx2]);
            y = B.r + g*min(q1t,q2t);
            X = [B.o, B.a, B.ctx];
            [gc1,~] = dlfeval(@Agent.criticLoss, obj.c1, X, y);
            [gc2,~] = dlfeval(@Agent.criticLoss, obj.c2, X, y);
            obj.updc=obj.updc+1;
            [obj.c1,obj.o1]=Agent.adam(obj.c1,gc1,obj.o1,cf.lrC,obj.updc);
            [obj.c2,obj.o2]=Agent.adam(obj.c2,gc2,obj.o2,cf.lrC,obj.updc);
            if mod(obj.updc,2)==0
                gA=dlfeval(@Agent.actorLoss, obj.actor, obj.c1, B.o, B.ctx, ...
                           obj.contDim, obj.chanN, obj.powN);
                [obj.actor,obj.oA]=Agent.adam(obj.actor,gA,obj.oA,cf.lrA,obj.updc);
                obj.tActor=Agent.soft(obj.tActor,obj.actor,tau);
                obj.tc1=Agent.soft(obj.tc1,obj.c1,tau); obj.tc2=Agent.soft(obj.tc2,obj.c2,tau);
            end
        end
    end

    methods (Static)
        function net = mlp(inD,hid,outD,outT)
            L=featureInputLayer(inD,'Name','in');
            for h=hid, L=[L; fullyConnectedLayer(h); reluLayer]; end %#ok<AGROW>
            L=[L; fullyConnectedLayer(outD)];
            if strcmp(outT,'tanh'), L=[L; tanhLayer]; end
            net=dlnetwork(L);
        end
        function y=fwd(net,X)      % X: rows=samples -> returns rows=samples
            y = extractdata(forward(net, dlarray(X','CB')))';
        end
        function p=softmax(z), z=z-max(z,[],2); e=exp(z); p=e./sum(e,2); end
        function s=gumbel(logit,tau)
            g=-log(-log(rand(size(logit))+1e-9)+1e-9);
            s=(logit+g)/tau;
        end
        function [g,loss]=criticLoss(net,X,y)
            q=forward(net,dlarray(X','CB'));
            loss=mean((q-dlarray(y','CB')).^2,'all');
            g=dlgradient(loss,net.Learnables);
        end
        function g=actorLoss(actor,critic,O,ctx,contDim,chanN,powN)
            y=forward(actor,dlarray(O','CB'));       % actDim x B
            cont=tanh(y(1:contDim,:));
            parts={cont}; off=contDim;
            if chanN>0, parts{end+1}=Agent.smCol(y(off+1:off+chanN,:)); off=off+chanN; end
            if powN>0,  parts{end+1}=Agent.smCol(y(off+1:off+powN,:)); end
            av=cat(1,parts{:});
            X=[dlarray(O','CB'); av; dlarray(ctx','CB')];
            q=forward(critic,dlarray(stripdims(X),'CB'));
            loss=-mean(q,'all');
            g=dlgradient(loss,actor.Learnables);
        end
        function p=smCol(z), z=z-max(z,[],1); e=exp(z); p=e./sum(e,1); end
        function [net,st]=adam(net,g,st,lr,t)
            [net,st.m,st.v]=adamupdate(net,g,st.m,st.v,t,lr);
        end
        function tnet=soft(tnet,net,tau)
            tnet.Learnables.Value=cellfun(@(a,b)(1-tau)*a+tau*b, ...
                tnet.Learnables.Value, net.Learnables.Value,'uni',0);
        end
    end
end
