function ImpliedVolatility
close all;clc

% Initial parameters and market quotes

CP       = 'c';   % C - call option and P - put option
V_market = 2;     % Market call option price
K        = 120;   % Strike price
tau      = 1;     % Time to maturity
r        = 0.05;  % Interest rate
S_0      = 100;   % Today's stock price

impVol = BSImpliedVolatility(CP,V_market,S_0,K,tau,r);
sprintf('Implied volatility is equal to:%.8f',impVol)

function impVol = BSImpliedVolatility(CP,V_market,S_0,K,tau,r)
error    = 1e10; % initial error
sigma    = 0.1;  % Initial implied volatility

% While the difference between the model and the market price is large
% proceed with the iteration

while error>10e-10
    f         = V_market - BS_Call_Option_Price(CP,S_0,K,sigma,tau,r);
    f_prim    = -dV_dsigma(S_0,K,sigma,tau,r);
    sigma_new = sigma - f / f_prim;

    error=abs(sigma_new-sigma);
    sigma=sigma_new;
end
impVol = sigma;

% Vega, dV/dsigma

function value=dV_dsigma(S_0,K,sigma,tau,r)

% Parameters and value of vega

d2   = (log(S_0 / K) + (r - 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
value = K * exp(-r * tau) * normpdf(d2) * sqrt(tau);

% Closed-form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end
