function BSHW_Comparison
clc;clf;close all;

% Characteristic function parameters 
% HW model parameter settings

lambda = 0.1;
eta   = 0.05;
sigma = 0.2;
rho   = 0.3;
S0    = 100; 
T     = 5;

% Strike equals stock value, thus ATM

K  = linspace(40,220,100)';
CP = 'c';

% We define a ZCB curve (obtained from the market)

P0T = @(T) exp(-0.05*T); 
 
% Settings for the COS method

N = 100;
L = 8;

% Characteristic function of the BSHW model + the COS method

cf        = @(u) ChFBSHW(u, T, P0T, lambda, eta, rho, sigma);
valCOS    = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,T,K,N,L,P0T(T));
exactBSHW = BSHOptionPrice(CP,S0,K,P0T(T),T,eta,sigma,rho,lambda);

% Option prices, the COS method vs. closed-form solution

figure(1)
hold on;grid on
plot(K,valCOS)
plot(K,exactBSHW,'--r')
legend('COS method','Exact')
title('Option price for the BSHW model')
xlabel('strike K')
ylabel('option price')

% Implied volatilities, the COS method vs. closed-form solution

IV = zeros(length(K),1);
for i=1:length(K)
    frwdStock = S0 / P0T(T);
    valCOSFrwd = valCOS(i) / P0T(T);
    IV(i) = ImpliedVolatilityBlack76(CP,valCOSFrwd,K(i),T,frwdStock,0.3);
end

% Implied volatility for the BSHW model

IVexact = BSHWVolatility(T,eta,sigma,rho,lambda);

figure(2)
hold on;grid on
plot(K,IV*100)
plot(K,IVexact*ones(length(K),1)*100,'--r')
legend('COS method','Exact')
title('Implied volatility for the BSHW model')
xlabel('strike K')
ylabel('Implied volatility [%]')
axis([min(K),max(K),20,30])
    

function volBSHW = BSHWVolatility(T,eta,sigma,rho,lambda)
    Br= @(t,T) 1/lambda * (exp(-lambda*(T-t))-1.0);
    sigmaF = @(t) sqrt(sigma * sigma + eta * eta * Br(t,T) .* Br(t,T) - 2.0 * rho * sigma * eta * Br(t,T));
    zGrid = linspace(0.0,T,2500);
    volBSHW = sqrt(1/T*trapz(zGrid,sigmaF(zGrid).*sigmaF(zGrid)));

function optPrice = BSHOptionPrice(CP,S0,K,P0T,T,eta,sigma,rho,lambda)
    frwdS0 = S0 / P0T;
    vol = BSHWVolatility(T,eta,sigma,rho,lambda);

    % As we deal with forward prices we evaluate Black's 76 pricing formula

    r = 0.0;
    BlackPrice = BS_Call_Option_Price(CP,frwdS0,K,vol,T,r);
    optPrice =   P0T * BlackPrice;

function cfV = ChFBSHW(u, T, P0T, lambd, eta, rho, sigma)

% Time step 

dt = 0.0001;

% Complex number

i = complex(0.0,1.0);
f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));  
C = @(u,tau)1.0/lambd*(i*u-1.0)*(1.0-exp(-lambd*tau));
 
 % Define a grid for the numerical integration of function theta

 zGrid = linspace(0.0,T,2500);
 term1 = @(u) 0.5*sigma*sigma *i*u*(i*u-1.0)*T;
 term2 = @(u) i*u*rho*sigma*eta/lambd*(i*u-1.0)*(T+1.0/lambd *(exp(-lambd*T)-1.0));
 term3 = @(u) eta*eta/(4.0*power(lambd,3.0))*power(i+u,2.0)*(3.0+exp(-2.0*lambd*T)-4.0*exp(-lambd*T)-2.0*lambd*T);
 term4 = @(u)  lambd*trapz(zGrid,theta(T-zGrid).*C(u,zGrid));
 A= @(u) term1(u) + term2(u) + term3(u) + term4(u);
 
 % Note that we don't include the B(u)*x0 term as it is included in the COS method

 cf = @(u)exp(A(u) + C(u,T)*r0 );
 
 % Iterate over all u and collect the ChF, iteration is necessary due to the integration in term4

 cfV = zeros(1,length(u));
 idx = 1;
 for ui=u
    cfV(idx)=cf(ui);
    idx = idx +1;
 end

function value = CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,tau,K,N,L,P0T)
i = complex(0,1);

% cf   - Characteristic function, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - Interest rate (constant)
% tau  - Time to maturity
% K    - Vector of strike prices
% N    - Number of expansion terms
% L    - Size of truncation domain (typ.:L=8 or L=10)  

x0 = log(S0 ./ K);   

% Truncation domain

a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

k = 0:N-1;     % row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);  % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % matrix-vector manipulations

% Final output

value = K .* real(mat * temp);

% Use the put-call parity to determine call prices (if needed)

if lower(CP) == 'c' || CP == 1
 value = value + S0 - K * P0T; 
end

% Coefficients H_k for the COS method

function H_k = CallPutCoefficients(CP,a,b,k)
 if lower(CP) == 'c' || CP == 1
  c = 0;
  d = b;
  [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
   if a < b && b < 0.0
   H_k = zeros([length(k),1]);
   else
   H_k = 2.0 / (b - a) * (Chi_k - Psi_k);
   end
 elseif lower(CP) == 'p' || CP == -1
  c = a;
  d = 0.0;
  [Chi_k,Psi_k]  = Chi_Psi(a,b,c,d,k);
   H_k = 2.0 / (b - a) * (- Chi_k + Psi_k);    
 end

function [chi_k,psi_k] = Chi_Psi(a,b,c,d,k)
 psi_k  = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
 psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
 psi_k(1)  = d - c;
 
 chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
 expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d)  - cos(k * pi... 
      * (c - a) / (b - a)) * exp(c);
 expr2 = k * pi / (b - a) .* sin(k * pi * ...
      (d - a) / (b - a))   - k * pi / (b - a) .* sin(k... 
      * pi * (c - a) / (b - a)) * exp(c);
 chi_k = chi_k .* (expr1 + expr2);
 
% Closed-form European call/put options with the Black-Scholes formula

function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1 = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2 = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
 value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
 value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
impliedVol = fzero(func,initialVol);
