function SZHW_MonteCarlo_vs_COSMethod
clc;clf;close all;
CP  = 'c';  
        
% HW model parameter settings

lambda = 0.05;
eta   = 0.01;

% Initial stock and time to maturity

S0    = 100.0;
T     = 5.0;
    
% The SZHW model

sigma0  = 0.1;
gamma   = 0.3;
Rsigmar = 0.32;
Rxsigma = -0.42;
Rxr     = 0.3;
kappa   = 0.4;
sigmaBar= 0.05;
    
% Monte Carlo setting

NoOfPaths = 25000;
NoOfSteps = round(100*T);

% We define the ZCB curve (obtained from the market)

P0T = @(T)exp(-0.025*T);
    
% Forward stock process

frwdStock = S0 / P0T(T);
    
% Strike price range

K = linspace(0.4*frwdStock,1.6*frwdStock,20)';
      
% Settings for the COS method

N = 2000;
L = 10;

% Valuation with the COS method

cf_SZHW = @(u)ChFSZHW(u,P0T,T,kappa,sigmaBar,gamma,lambda,eta,Rxsigma,Rxr,Rsigmar,sigma0);
valCOS_SZHW    = CallPutOptionPriceCOSMthd_StochIR(cf_SZHW,CP,S0,T,K,N,L,P0T(T));

% Euler simulation of the SZHW model- the sumulation is repeated NoOfSeeds times

NoOfSeeds = 10;
optionValueMC = zeros([length(K),NoOfSeeds]);
idx = 1;
for seed =1:NoOfSeeds
    randn('seed',seed)
    [S_Euler,~,M_t] = GeneratePathsSZHW_Euler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmaBar,kappa,gamma,lambda,eta,Rxsigma,Rxr,Rsigmar);
    optValueEuler = EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S_Euler(:,end),K,M_t(:,end));
    optionValueMC(:,idx) = optValueEuler;
    fprintf('Seed number %.0f out of %.0f',idx,NoOfSeeds);
    fprintf('\n')
    idx = idx +1;
end
optionValueMCFinal = mean(optionValueMC,2);

figure(1); hold on;
plot(K,optionValueMCFinal,'-b','linewidth',1.5)
plot(K,valCOS_SZHW,'--r','linewidth',1.5)
xlabel('strike,K')
ylabel('Option price')
grid on
legend('Euler','COS')
title('European option prices for the SZHW model')

% Implied volatilities, the COS method and MC simulation

IVCOS = zeros(length(K),1);
IVMC = zeros(length(K),1);
for i=1:length(K)
    frwdStock = S0 / P0T(T);
    valCOSFrwd = valCOS_SZHW(i) / P0T(T);
    valMCFrwd = optionValueMCFinal(i)/P0T(T);
    IVCOS(i) = ImpliedVolatilityBlack76(CP,valCOSFrwd,K(i),T,frwdStock,0.3)*100.0;
    IVMC(i) = ImpliedVolatilityBlack76(CP,valMCFrwd,K(i),T,frwdStock,0.3)*100.0;
end

figure(2)
plot(K,IVCOS,'--r');hold on
plot(K,IVMC,'-b')
legend('IV-COS method','IV-Monte Carlo')
title('Implied volatilities for the SZHW model: COS method vs. Monte Carlo')
grid on

function [S,time,M_t] = GeneratePathsSZHW_Euler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmaBar,kappa,gamma,lambd,eta,Rxsigma,Rxr,Rsigmar)

% Time step needed 

dt = 0.0001;
f0T = @(t) - (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
    
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);
theta = @(t) 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-exp(-2.0*lambd*t));

% Empty containers for Brownian paths

Wx = zeros([NoOfPaths, NoOfSteps+1]);
Wsigma = zeros([NoOfPaths, NoOfSteps+1]);
Wr = zeros([NoOfPaths, NoOfSteps+1]);

Sigma = zeros([NoOfPaths, NoOfSteps+1]);
X = zeros([NoOfPaths, NoOfSteps+1]);
R = zeros([NoOfPaths, NoOfSteps+1]);
M_t = ones([NoOfPaths,NoOfSteps+1]);
R(:,1)=r0;
Sigma(:,1)=sigma0;
X(:,1)=log(S0);

covM = [[1.0, Rxsigma,Rxr];[Rxsigma,1.0,Rsigmar]; [Rxr,Rsigmar,1.0]];
time = zeros([NoOfSteps+1,1]);
        
dt = T / NoOfSteps;
for i = 1:NoOfSteps
    Z = mvnrnd([0,0,0],covM*dt,NoOfPaths);

    % Making sure that samples from a normal have mean 0 and variance 1

    Z1= Z(:,1);
    Z2= Z(:,2);
    Z3= Z(:,3);
    if NoOfPaths > 1
        Z1 = (Z1 - mean(Z1)) / std(Z1);
        Z2 = (Z2 - mean(Z2)) / std(Z2);
        Z3 = (Z3 - mean(Z3)) / std(Z3);
    end
    
    Wx(:,i+1)  = Wx(:,i)  + sqrt(dt)*Z1;
    Wsigma(:,i+1)  = Wsigma(:,i)  + sqrt(dt)*Z2;
    Wr(:,i+1) = Wr(:,i) + sqrt(dt)*Z3;
    
    R(:,i+1) = R(:,i) + lambd*(theta(time(i)) - R(:,i)) * dt + eta* (Wr(:,i+1)-Wr(:,i));
    M_t(:,i+1) = M_t(:,i) .* exp(0.5*(R(:,i+1) + R(:,i))*dt);
    
    % The volatility process

    Sigma(:,i+1) = Sigma(:,i) + kappa*(sigmaBar-Sigma(:,i))*dt+  gamma*(Wsigma(:,i+1)-Wsigma(:,i));   
     
    % Simulation of the log-stock process

    X(:,i+1) = X(:,i) + (R(:,i) - 0.5*Sigma(:,i).^2)*dt + Sigma(:,i).*(Wx(:,i+1)-Wx(:,i));       
        
    % Moment matching component, i.e. ensure that E(S(T)/M(T))= S0

    a = S0 / mean(exp(X(:,i+1))./M_t(:,i+1));
    X(:,i+1) = X(:,i+1) + log(a);
    time(i+1) = time(i) +dt;
end

    % Compute exponent

    S = exp(X);

function optValue = EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S,K,M_t)
optValue = zeros(length(K),1);
if lower(CP) == 'c' || lower(CP) == 1
    for i =1:length(K)
        optValue(i) = mean(1./M_t.*max(S-K(i),0));
    end
elseif lower(CP) == 'p' || lower(CP) == -1
    for i =1:length(K)
        optValue(i) = mean(1./M_t.*max(K(i)-S,0));
    end
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

k = 0:N-1;             % Row vector, index for expansion terms
u = k * pi / (b - a);  % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

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

 function value = ChFSZHW(u,P0T,tau,kappa, sigmabar,gamma,lambda,eta,Rxsigma,Rxr,Rrsigma,sigma0)
v_D = D(u,tau,kappa,Rxsigma,gamma);
v_E = E(u,tau,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar);
v_A = A(u,tau,P0T,eta,lambda,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar);

% Initial variance 

v0     =sigma0^2;

% Characteristic function of the SZHW model

value = exp(v0*v_D + sigma0*v_E + v_A);
        
function value=C(u,tau,lambda)
    i     = complex(0,1);
    value = 1/lambda*(i*u-1).*(1-exp(-lambda*tau)); 

function value=D(u,tau,kappa,Rxsigma,gamma)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d=sqrt(a_1.^2-4*a_0.*a_2);
    g=(-a_1-d)./(-a_1+d);    
value=(-a_1-d)./(2*a_2*(1-g.*exp(-d*tau))).*(1-exp(-d*tau));

function value=E(u,tau,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d  =sqrt(a_1.^2-4*a_0.*a_2);
    g =(-a_1-d)./(-a_1+d);    
    
    c_1=gamma*Rxsigma*i*u-kappa-1/2*(a_1+d);
    f_1=1./c_1.*(1-exp(-c_1*tau))+1./(c_1+d).*(exp(-(c_1+d)*tau)-1);
    f_2=1./c_1.*(1-exp(-c_1*tau))+1./(c_1+lambda).*(exp(-(c_1+lambda)*tau)-1);
    f_3=(exp(-(c_1+d)*tau)-1)./(c_1+d)+(1-exp(-(c_1+d+lambda)*tau))./(c_1+d+lambda);
    f_4=1./c_1-1./(c_1+d)-1./(c_1+lambda)+1./(c_1+d+lambda);
    f_5=exp(-(c_1+d+lambda).*tau).*(exp(lambda*tau).*(1./(c_1+d)-exp(d*tau)./c_1)+exp(d*tau)./(c_1+lambda)-1./(c_1+d+lambda)); 

    I_1=kappa*sigmabar./a_2.*(-a_1-d).*f_1;
    I_2=eta*Rxr*i*u.*(i*u-1)./lambda.*(f_2+g.*f_3);
    I_3=-Rrsigma*eta*gamma./(lambda.*a_2).*(a_1+d).*(i*u-1).*(f_4+f_5);
    value=exp(c_1*tau).*1./(1-g.*exp(-d*tau)).*(I_1+I_2+I_3);
    
function value=A(u,tau,P0T,eta,lambda,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar)
    i=complex(0,1);
    a_0=-1/2*u.*(i+u);
    a_1=2*(gamma*Rxsigma*i*u-kappa);
    a_2=2*gamma^2;
    d  =sqrt(a_1.^2-4*a_0.*a_2);
    g =(-a_1-d)./(-a_1+d); 
    f_6=eta^2/(4*lambda^3)*(i+u).^2*(3+exp(-2*lambda*tau)-4*exp(-lambda*tau)-2*lambda*tau);
    A_1=1/4*((-a_1-d)*tau-2*log((1-g.*exp(-d*tau))./(1-g)))+f_6;
   
    % Integration within the function A(u,tau)

    value=zeros(1,length(u));   
    N=500;
    arg=linspace(0,tau,N);

    % Solve the integral for A

    for k=1:length(u)
       E_val=E(u(k),arg,lambda,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar);
       C_val=C(u(k),arg,lambda);
       f=(kappa*sigmabar+1/2*gamma^2*E_val+gamma*eta*Rrsigma*C_val).*E_val;
       value(1,k)=trapz(arg,f);
    end
    value = value + A_1;   
    
% Correction for the interest rate term structure    

help = eta^2/(2*lambda^2)*(tau+2/lambda*(exp(-lambda*tau)-1)-1/(2*lambda)*(exp(-2*lambda*tau)-1));
correction = (i*u-1).*(log(1/P0T(tau))+help);
value = value + correction;

 % Closed-form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r)

% Black-Scholes call option price

d1    = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * tau) ./ (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function impliedVol = ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock,initialVol)
func = @(sigma) (BS_Call_Put_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice).^1.0;
impliedVol = fzero(func,initialVol);
