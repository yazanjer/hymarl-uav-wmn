function [Prx_dBm, gain_lin] = uwmn_channel(txPos, rxPos, Ptx_dBm, cfg, rng_shadow)
%UWMN_CHANNEL  Friis path loss + log-normal shadowing (thesis Eq. 3.2).
%   [Prx_dBm, gain_lin] = UWMN_CHANNEL(txPos, rxPos, Ptx_dBm, cfg, rng_shadow)
%   txPos, rxPos : (M x 3) node positions (m). Ptx_dBm : (M x 1) or scalar Tx power.
%   Returns received power (dBm) and linear channel power gain g_ij (Eq. 3.9 uses g).
%
%   Pr(i,j) = Pt - PL(d0) - 10 s log10(d/d0) - X_sigma,  X_sigma ~ N(0, sig^2)   (3.2)

d = sqrt(sum((txPos - rxPos).^2, 2));
d = max(d, cfg.d0);                     % clamp to reference distance
PL = cfg.PLd0 + 10*cfg.plExp*log10(d./cfg.d0);
if nargin >= 5 && ~isempty(rng_shadow)
    Xs = cfg.shadowSig * randn(rng_shadow, size(d));
else
    Xs = cfg.shadowSig * randn(size(d));
end
Prx_dBm  = Ptx_dBm - PL - Xs;
% Linear channel power gain g = Prx_lin / Ptx_lin  (dimensionless path gain)
gain_dB  = -PL - Xs;
gain_lin = 10.^(gain_dB/10);
end
