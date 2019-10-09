function Strikes
clc;close all;

% Set of maturitie times

 TiV = [0.5, 1.0, 5.0, 10.0, 20.0, 30.0];
 strikesM = zeros([length(TiV),7]);
 
% Market ZCBs for domestic and foreign markets

P0T_d = @(t) exp(-0.02 * t);
P0T_f = @(t) exp(-0.05 * t);
    
% Spot of the FX rate

y0 = 1.35;
    
 for i = 1: length(TiV)
    ti = TiV(i);
    frwd = y0 * P0T_f(ti) / P0T_d(ti);
    strikesM(i,:) = GenerateStrikes(frwd,ti);
 end
 
 strikesM

function value = GenerateStrikes(frwd,Ti)
c_n = [-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5];
value =  frwd * exp(0.1 * c_n * sqrt(Ti));
