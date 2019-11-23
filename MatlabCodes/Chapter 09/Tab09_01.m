function OptionPrices_EulerAndMilstein
clc;close all;
S0         = 5;
NoOfSteps  = 1000; 
r          = 0.06;
sigma      = 0.3;
T          = 1;
CP         = 'c';
K          = S0;

% Specification of the number of paths

NoOfPathsV = [100,1000,5000,5000,10000,20000];

% Call option pricing and convergence wrt. the number of paths

exactPrice = BS_Call_Option_Price(CP,S0,K,sigma,T,r);
for NoOfPaths= NoOfPathsV
    randn('seed',1)
    S_Euler = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0);
    randn('seed',1)
    S_Milstein = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0);
    priceEuler = EUOptionPriceFromMCPaths(CP,S_Euler(:,end),K,T,r);
    priceMilstein = EUOptionPriceFromMCPaths(CP,S_Milstein(:,end),K,T,r);
    
    sprintf('===========NoOfPaths = %.f=============',NoOfPaths)
    fprintf('Euler call price = %.5f and Mistein call price = %.5f\n',priceEuler,priceMilstein)
    fprintf('Euler error= %.5f, Mistein error = %.5f',priceEuler-exactPrice,priceMilstein-exactPrice)
end

% Cash or nothing pricing and convergence wrt. the number of paths

exactPrice = BS_Cash_Or_Nothing_Price(CP,S0,K,sigma,T,r);
for NoOfPaths= NoOfPathsV
    randn('seed',1)
    S_Euler = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0);
    randn('seed',1)
    S_Milstein = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0);
    priceEuler = CashofNothingPriceFromMCPaths(CP,S_Euler(:,end),K,T,r);
    priceMilstein = CashofNothingPriceFromMCPaths(CP,S_Milstein(:,end),K,T,r);
    
    sprintf('===========NoOfPaths = %.f=============',NoOfPaths)
    fprintf('Euler call price = %.5f and Mistein call price = %.5f\n',priceEuler,priceMilstein)
    fprintf('Euler error= %.5f, Mistein error = %.5f',priceEuler-exactPrice,priceMilstein-exactPrice)
end

function optValue = EUOptionPriceFromMCPaths(CP,S,K,T,r)
if lower(CP) == 'c' || lower(CP) == 1
    optValue = exp(-r*T)*mean(max(S-K,0));
elseif lower(CP) == 'p' || lower(CP) == -1
    optValue = exp(-r*T)*mean(max(K-S,0));
end

function optValue = CashofNothingPriceFromMCPaths(CP,S,K,T,r)
if lower(CP) == 'c' || lower(CP) == 1
    optValue = exp(-r*T)*K*mean(S>K);
elseif lower(CP) == 'p' || lower(CP) == -1
    optValue = exp(-r*T)*K*mean(S<=K);
end

function S = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0)

% Approximation

S=zeros(NoOfPaths,NoOfSteps);
S(:,1) = S0;

% Random noise

Z=random('normal',0,1,[NoOfPaths,NoOfSteps]);
W=zeros([NoOfPaths,NoOfSteps]);

dt = T / NoOfSteps;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps
    if NoOfPaths>1
        Z(:,i)   = (Z(:,i) - mean(Z(:,i))) / std(Z(:,i));
    end
    W(:,i+1)  = W(:,i) + sqrt(dt).*Z(:,i);
    S(:,i+1) = S(:,i) + r * S(:,i)*dt + sigma * S(:,i).*(W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end

function [S] = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0)

% Euler approximation

S=zeros(NoOfPaths,NoOfSteps);
S(:,1) = S0;

% Random noise

Z=random('normal',0,1,[NoOfPaths,NoOfSteps]);
W=zeros([NoOfPaths,NoOfSteps]);

dt = T / NoOfSteps;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps
    if NoOfPaths>1
        Z(:,i)   = (Z(:,i) - mean(Z(:,i))) / std(Z(:,i));
    end
    W(:,i+1)  = W(:,i) + sqrt(dt).*Z(:,i);
    S(:,i+1) = S(:,i) + r * S(:,i)*dt + sigma * S(:,i).*(W(:,i+1)-W(:,i))+...
        + 0.5*sigma^2*S(:,i).*((W(:,i+1)-W(:,i)).^2 - dt);   
    time(i+1) = time(i) + dt;
end

% Closed-form expression of European call/put option with Black-Scholes formula

function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

% Black-Scholes formula for cash or nothing option

function value= BS_Cash_Or_Nothing_Price(CP,S_0,K,sigma,tau,r)
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value = K * exp(-r * tau) * normcdf(d2);
elseif lower(CP) == 'p' || lower(CP) == -1
    value = K * exp(-r * tau) *(1.0 - normcdf(d2));
end
