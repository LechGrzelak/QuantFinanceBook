function PathwiseSens_DeltaVega
clc;close all;
CP        = 'c'; 
S0        = 1;
r         = 0.06;
sigma     = 0.3;
T         = 1;
K         = S0;
t         = 0.0;

NoOfSteps = 1000;

delta_Exact = BS_Delta(CP,S0,K,sigma,t,T,r);
vega_Exact  = BS_Vega(S0,K,sigma,t,T,r);
   
NoOfPathsV = round(linspace(5,20000,50));
deltaPathWiseV = zeros([length(NoOfPathsV),1]);
vegaPathWiseV  = zeros([length(NoOfPathsV),1]);

idx = 1;
for nPaths = NoOfPathsV
     fprintf('Running simulation with %.f paths',nPaths);
     fprintf('\n')
     randn('seed',1);
     
     [S] = GeneratePathsGBMEuler(nPaths,NoOfSteps,T,r,sigma,S0);
     
     % Delta -- pathwise

     delta_pathwise = PathwiseDelta(S0,S,K,r,T);
     deltaPathWiseV(idx)= delta_pathwise;
     
     % Vega -- pathwise

     vega_pathwise = PathwiseVega(S0,S,sigma,K,r,T);
     vegaPathWiseV(idx) =vega_pathwise;
     idx = idx +1;
end

figure(1)
grid on; hold on
plot(NoOfPathsV,deltaPathWiseV,'-or','linewidth',1.5)
plot(NoOfPathsV,delta_Exact*ones([length(NoOfPathsV),1]),'linewidth',1.5)
xlabel('number of paths')
ylabel('Delta')
title('Convergence of pathwise delta w.r.t number of paths')
legend('pathwise est','exact')
ylim([delta_Exact*0.8,delta_Exact*1.2])
    
figure(2)
grid on; hold on
plot(NoOfPathsV,vegaPathWiseV,'-or','linewidth',1.5)
plot(NoOfPathsV,vega_Exact*ones([length(NoOfPathsV),1]),'linewidth',1.5)
xlabel('number of paths')
ylabel('Vega')
title('Convergence of pathwise vega w.r.t number of paths')
legend('pathwise estimator','exact')
ylim([vega_Exact*0.8,vega_Exact*1.2])

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


function optValue = EUOptionPriceFromMCPathsGeneralized(CP,S,K,T,r)
optValue = zeros(length(K),1);
if lower(CP) == 'c' || lower(CP) == 1
    for i =1:length(K)
        optValue(i) = exp(-r*T)*mean(max(S-K(i),0));
    end
elseif lower(CP) == 'p' || lower(CP) == -1
    for i =1:length(K)
        optValue(i) = exp(-r*T)*mean(max(K(i)-S,0));
    end
end

function delta= PathwiseDelta(S0,S,K,r,T)
    temp1 = S(:,end)>K;
    delta =exp(-r*T)*mean(S(:,end)/S0.*temp1); 

function vega= PathwiseVega(S0,S,sigma,K,r,T)
    temp1 = S(:,end)>K;
    temp2 = 1.0/sigma* S(:,end).*(log(S(:,end)/S0)-(r+0.5*sigma^2)*T);
    vega = exp(-r*T)*mean(temp1.*temp2);

function [S,time] = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0)
X=zeros(NoOfPaths,NoOfSteps);
X(:,1) = log(S0);

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
    X(:,i+1) = X(:,i) + (r -0.5*sigma^2)*dt + sigma *(W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end
S=exp(X);
