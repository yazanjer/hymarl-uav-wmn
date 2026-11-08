function [userPos, mob] = uwmn_mobility(userPos, mob, cfg, rs)
%UWMN_MOBILITY  Advance ground-user positions one step under the scenario model.
%   [userPos, mob] = UWMN_MOBILITY(userPos, mob, cfg, rs)
%   userPos : U x 3 positions (col 3 = 0). mob : bookkeeping struct. rs : RandStream.
%   Supports Random-Waypoint (RWP), Reference-Point Group Mobility (RPGM), and
%   Manhattan Grid (thesis Sec. 3.5.1, 3.5.2.2).

U = cfg.U; A = cfg.area; dt = cfg.dt;
switch upper(cfg.mobility)
    case 'RWP'   % Random Waypoint
        reached = sqrt(sum((userPos(:,1:2)-mob.userWP).^2,2)) < 5;
        mob.userWP(reached,:) = A*rand(rs,[nnz(reached),2]);
        dir = mob.userWP - userPos(:,1:2);
        nrm = max(sqrt(sum(dir.^2,2)),1e-6);
        step = dir./nrm .* min(mob.userSpd*dt, nrm);
        userPos(:,1:2) = userPos(:,1:2) + step;

    case 'RPGM'  % Reference-Point Group Mobility (evacuation toward impact points)
        gdir = mob.groupRef - mob.groupCentroid;
        gnrm = max(sqrt(sum(gdir.^2,2)),1e-6);
        mob.groupCentroid = mob.groupCentroid + gdir./gnrm .* min(2*dt, gnrm);
        target = mob.groupCentroid(mob.userGroup,:) + 20*randn(rs,[U,2]);
        dir = target - userPos(:,1:2);
        nrm = max(sqrt(sum(dir.^2,2)),1e-6);
        userPos(:,1:2) = userPos(:,1:2) + dir./nrm .* min(mob.userSpd*dt, nrm);
        if rand(rs) < 0.02
            mob.groupRef = A*rand(rs,[size(mob.groupRef,1),2]);
        end

    case 'MANHATTAN'  % Manhattan grid crowd flow (axis-aligned moves)
        move = mob.userSpd*dt;
        dx = (mob.userAxis==1) .* mob.userDir .* move;
        dy = (mob.userAxis==2) .* mob.userDir .* move;
        userPos(:,1) = userPos(:,1) + dx;
        userPos(:,2) = userPos(:,2) + dy;
        turn = rand(rs,[U,1]) < 0.05;  mob.userAxis(turn) = 3 - mob.userAxis(turn);
        flip = rand(rs,[U,1]) < 0.03;  mob.userDir(flip)  = -mob.userDir(flip);
end
userPos(:,1:2) = min(max(userPos(:,1:2), 0), A);
end
