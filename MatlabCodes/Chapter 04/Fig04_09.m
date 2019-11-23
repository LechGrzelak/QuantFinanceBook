function CalibrationSABRformula
close all;clc;clf;

% Market given implied volatilities for maturity T=1

K  = [110, 121.3, 131.5, 140.9, 151.4];
iv = [14.2, 11.8, 10.3, 10.0, 10.7]/100;

% Assume r = 0

r = 0;
T = 1;

% Beta parameter is typically fixed

beta = 0.5;
CP   = 'p';

% Forward and ATM volatility

f_0       = 131.5;
iv_atm    = 10.3/100;

% Create a function which will return implied volatility

iv_h = @(K,alpha,rho,gamma)HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,gamma);

% Calibrate Hagan's SABR approximation formula to market implied volatilities

[alpha_est,rho_est,gamma_est] = calibrationHagan(K,T,beta,iv,iv_atm,f_0,f_0);

% Plot the results

figure(1)
plot(K,iv*100,'ok')
hold on
iv_fInterp  = @(x)interp1(K,iv,x,'linear','extrap');
k_ind       = K(1)-10:1:K(end)+10;
plot(k_ind,iv_fInterp(k_ind)*100,'-r','linewidth',1.5)
plot(k_ind,iv_h(k_ind,alpha_est,rho_est,gamma_est)*100,'--k','linewidth',1.5)
grid on;
xlabel('K')
ylabel('Implied volatilities [%]')
title('Interpolation of implied volatilities')
legend('Market Quotes','Linear Interpolation','Parameterization')

% Density from Hagan's formula

f_x  = DensityFromSABRParametric(CP,beta,alpha_est,rho_est,gamma_est,f_0,T,r);

figure(2)
x=K(1)-50:0.17:K(end)+50;
hold on
plot(x,f_x(x),'--k','linewidth',1.5)
trapz(x,f_x(x))

% The case of linearly interpolated implied volatilities

call  = @(K)BS_Call_Option_Price(CP,f_0,K,iv_fInterp(K),T,r);
bump  = 0.0001;
fLinInterp = @(x)(call(x+bump)+call(x-bump)-2*call(x))./bump^2;

% Figure

hold on
plot(x,fLinInterp(x),'-r','linewidth',1.5)
trapz(x,fLinInterp(x))
grid on
xlabel('x')
ylabel('f_S(x)')
title('Implied densities for different implied volatility interpolations')
legend('Parameterization','Linear Interpolation')
axis([80,180,0,0.04])

% Integration results for two cases

sprintf('Integral of density, parametric case = %.4f',trapz(x,f_x(x)))
sprintf('Integral of density, linear case = %.4f',trapz(x,fLinInterp(x)))

% Implied volatility according to the formula 

function impVol = HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma)
z        = gamma/alpha*(f*K).^((1-beta)/2).*log(f./K);
x_z      = log((sqrt(1-2*rho*z+z.^2)+z-rho)/(1-rho));
A        = alpha./((f*K).^((1-beta)/2).*(1+(1-beta)^2/24*log(f./K).^2+(1-beta)^4/1920*log(f./K).^4));
B1       = 1 + ((1-beta)^2/24*alpha^2./((f*K).^(1-beta))+1/4*(rho*beta*gamma*alpha)./((f*K).^((1-beta)/2))+(2-3*rho^2)/24*gamma^2)*T;
impVol   = A.*(z./x_z).*B1;

B2 = 1 + ((1-beta)^2/24*alpha^2./(f.^(2-2*beta))+1/4*(rho*beta*gamma*alpha)./(f.^(1-beta))+(2-3*rho^2)/24*gamma^2)*T;
impVol(K==f) = alpha/(f^(1-beta))*B2;

% Calibration of the SABR Hagan parametrization

% The calibration algorithm is defined as follows: we search for two
% parameters rho and gamma. For any set of these two parameters, alpha is
% determined such that ATM we have an almost perfect fit. 

function [alpha_est,rho_est,gamma_est] = calibrationHagan(K,t,beta,iv,iv_ATM,K_ATM,f)

% x = [rho, gamma];
f_obj = @(x)targetVal(beta,x(1),x(2),iv,K,iv_ATM,K_ATM,f,t);
pars  = fminunc(f_obj,[-0.8,0.4]);
rho_est = pars(1);
gamma_est  = pars(2);

% For a given rho and gamma, re-calculate alpha to ensure the ATM fit

[~,alpha_est] = targetVal(beta,rho_est,gamma_est,iv,K,iv_ATM,K_ATM,f,t);

function [value,alpha_est] = targetVal(beta,rho,gamma,iv,K,iv_ATM,K_ATM,f,t)

% Implied volatilities parametrization

iv_hCAL     = @(alpha)HaganImpliedVolatility(K_ATM,t,f,alpha,beta,rho,gamma);
target      = @(alpha) (iv_hCAL(alpha)-iv_ATM);
[alpha_est] = fzero(target,1.05); % Initial guess does not really matter here
iv_h        = @(K,alpha,rho,gamma)HaganImpliedVolatility(K,t,f,alpha,beta,rho,gamma);
value       = sum((iv_h(K,alpha_est,rho,gamma)- iv).^2);

% Density obtained from Hagan's SABR formula

function value = DensityFromSABRParametric(CP,beta,alpha,rho,volvol,f_0,T,r)
iv_f       = @(K)HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,volvol);
callPrice  = @(K)BS_Call_Option_Price(CP,f_0,K,iv_f(K),T,r);
bump       = 0.0001;
value      = @(x)exp(r*T)*(callPrice(x + bump) + callPrice(x - bump) - 2 * callPrice(x))./ bump^2;

% Black-Scholes call option price

function value = BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * tau) ./ (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value = normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value = normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end
