function Convergence_BS_theGreeks
clc;close all;
CP        = 'c'; 
S0        = 1;
r         = 0.06;
sigma     = 0.3;
T         = 1;
K         = S0;
t         = 0.0;

% Range of shock values 

dxV = 0.00001:0.001:0.4;

error_frwdDiff_DELTA  = zeros([length(dxV),1]);
error_frwdDiff_VEGA   = zeros([length(dxV),1]);
error_centDiff_DELTA  = zeros([length(dxV),1]);
error_centDiff_VEGA   = zeros([length(dxV),1]);

exactDelta = BS_Delta(CP,S0,K,sigma,t,T,r);
exactVega  = BS_Vega(S0,K,sigma,t,T,r);
bsPrice    =@(S0,sigma) BS_Call_Put_Option_Price(CP,S0,K,sigma,t,T,r);

for idx = 1:length(dxV)
dx = dxV(idx);

% Shocks will be proportional to a parameter

dS0 =dx *S0;
dsigma =dx *sigma;

% Delta estimation, i.e. dV/dS0

frwdDiffDelta    = (bsPrice(S0+dS0,sigma) - bsPrice(S0,sigma))/dS0;
centralDiffDelta = (bsPrice(S0+dS0,sigma) - bsPrice(S0-dS0,sigma))/(2*dS0);
error_frwdDiff_DELTA(idx) = abs(exactDelta - frwdDiffDelta);
error_centDiff_DELTA(idx) = abs(exactDelta - centralDiffDelta);

% Vega estimation, i.e. dV/dsigma

frwdDiffVega    = (bsPrice(S0,sigma+dsigma) - bsPrice(S0,sigma))/dsigma;
centralDiffVega = (bsPrice(S0,sigma+dsigma) - bsPrice(S0,sigma-dsigma))/(2*dsigma);
error_frwdDiff_VEGA(idx) = abs(frwdDiffVega- exactVega);
error_centDiff_VEGA(idx) = abs(centralDiffVega- exactVega);

end
figure(1)
hold on;
grid on
plot(dxV*S0,error_frwdDiff_DELTA,'linewidth',1.5)
plot(dxV*S0,error_centDiff_DELTA,'--r','linewidth',1.5)
xlabel('\Delta{S0}')
ylabel('error,  \epsilon(\Delta{S0})')
legend('forward diff.','cental diff.')
title('$$\frac{dV}{dS_0}$$, errors for the forward and central difference')

figure(2)
hold on;
grid on
plot(dxV*sigma,error_frwdDiff_VEGA,'linewidth',1.5)
plot(dxV*sigma,error_centDiff_VEGA,'--r','linewidth',1.5)
xlabel('\Delta\sigma')
ylabel('error,  \epsilon(\Delta\sigma)')
legend('forward diff.','cental diff.')
title('$$\frac{dV}{d\sigma}$$, errors for the forward and central difference')

% Black-Scholes call option price

function value = BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
d2    = d1 - sigma * sqrt(T-t);
if lower(CP) == 'c' || lower(CP) == 1
    value = normcdf(d1) .* S_0 - normcdf(d2) .* K * exp(-r * (T-t));
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(-d2) .* K*exp(-r*(T-t)) - normcdf(-d1).*S_0;
end

function value = BS_Delta(CP,S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
if lower(CP) == 'c' || lower(CP) == 1
   value = normcdf(d1);
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(d1)-1;
end

function value = BS_Vega(S_0,K,sigma,t,T,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) ./ (sigma * sqrt(T-t));
value = S_0.*normpdf(d1)*sqrt(T-t);
