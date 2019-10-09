function arbitrage_Fig
close all; clc;

% Parameters for Hagan's SABR formula

CP    = 'c';
beta  = 0.5;
alpha = 0.06;
rho   = -0.5;
gamma = 0.5;
f_0   = 0.05;
T     = 7;
r     = 0;

% Figure 1 - dC_dK

figure(1)
grid on 
f_Y  = d2VdK2(CP,beta,alpha,rho,gamma,f_0,T,r);
x_0  = fzero(@(x)f_Y(x),0.04);
x1   = linspace(0.002,x_0,100);
x2   = linspace(x_0,0.15,100);
x    = [x1,x2];
hold on
F_C = dV_dK(CP,beta,alpha,rho,gamma,f_0,T,r);
plot(x1,F_C(x1),'--r','linewidth',2)
plot(x2,F_C(x2),'linewidth',2)
xlabel('K')
ylab = ylabel('$$\frac{d}{dK}V_c(t_0,S_0,K,T)$$');
set(ylab,'interpreter','latex')
set(gca,'xtick',linspace(0,0.15,7))

% Figure 2 - d2C_dK2 - the density computed for given implied volatilities

figure(2)
hold on
set(gca,'xtick',linspace(0,0.15,7))
plot(x1,f_Y(x1),'--r','linewidth',2)
plot(x2,f_Y(x2),'linewidth',2)

plot(x,zeros(length(x),1),'--k')
grid on
xlabel('K')
ylab = ylabel('$$\frac{d^2}{dK^2}V_c(t_0,S_0,K,T)$$');
set(ylab,'interpreter','latex')

% Figure 3 - Calendar spread - option plotted for different maturities

figure(3)
hold on
grid on
K = linspace(0,0.15,100);
T = 10;
iv = HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,gamma);
call = BS_Call_Option_Price(CP,f_0,K,iv,T,r);
plot(K,call,'linewidth',2)

T = 15;

% To illustrate the calendar spread we set rho*2 and choose gamma=0.001

iv = HaganImpliedVolatility(K,T,f_0,alpha,beta,rho*2,0.001);
[call] = BS_Call_Option_Price(CP,f_0,K,iv,T,r);
plot(K,call,'--r','linewidth',2)
set(gca,'xtick',linspace(0,0.15,7))
xlabel('K')
ylabl = ylabel('$$V_c(t_0,S_0,K,T)$$');
set(ylabl,'interpreter','latex')
legend('Call option, T= 10','Call option, T= 15')

% Derivative of an European call/put option wrt strike

function value = dV_dK(CP,beta,alpha,rho,volvol,f_0,T,r)
iv_f      = @(K)HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,volvol);
callPrice = @(K)BS_Call_Option_Price(CP,f_0,K,iv_f(K),T,r);
bump      = 0.00001;
value     = @(x)(callPrice(x+bump)-callPrice(x-bump))./(2 * bump);

% Second-order derivative of European call/put price wrt strike

function value = d2VdK2(CP,beta,alpha,rho,volvol,f_0,T,r)
iv_f       = @(K)HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,volvol);
callPrice  = @(K)BS_Call_Option_Price(CP,f_0,K,iv_f(K),T,r);
bump       = 0.00001;
value      = @(x)(callPrice(x+bump) + callPrice(x-bump)-2*callPrice(x))./ bump^2;

% Implied volatility according to Hagan et al. formula 

function impVol = HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma)
z        = gamma/alpha*(f*K).^((1-beta)/2).*log(f./K);
x_z      = log((sqrt(1-2*rho*z+z.^2)+z-rho)/(1-rho));
A        = alpha./((f*K).^((1-beta)/2).*(1+(1-beta)^2/24*log(f./K).^2+(1-beta)^4/1920*log(f./K).^4));
B1       = 1 + ((1-beta)^2/24*alpha^2./((f*K).^(1-beta))+1/4*(rho*beta*gamma*alpha)./((f*K).^((1-beta)/2))+(2-3*rho^2)/24*gamma^2)*T;
impVol   = A.*(z./x_z).*B1;

B2 = 1 + ((1-beta)^2/24*alpha^2./(f.^(2-2*beta))+1/4*(rho*beta*gamma*alpha)./(f.^(1-beta))+(2-3*rho^2)/24*gamma^2)*T;
impVol(K==f) = alpha/(f^(1-beta))*B2;

% Black-Scholes call option price

function value = BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * tau) ./ (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value = normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end
