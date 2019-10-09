function LikelihoodSens_DeltaVega
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
   
NoOfPathsV = round(linspace(5,25000,25));
deltaPathWiseV = zeros([length(NoOfPathsV),1]);
deltaLikeliWiseV = zeros([length(NoOfPathsV),1]);
vegaPathWiseV  = zeros([length(NoOfPathsV),1]);
vegaLikelihoodV= zeros([length(NoOfPathsV),1]);

idx = 1;
for nPaths = NoOfPathsV
     fprintf('Running simulation with %.f paths',nPaths);
     fprintf('\n')
     randn('seed',1);
     
     [S] = GeneratePathsGBMEuler(nPaths,NoOfSteps,T,r,sigma,S0);
     
     % Delta -- pathwise

     delta_pathwise   = PathwiseDelta(S0,S,K,r,T);
     delta_likelihood = LikelihoodDelta(S0,S,K,r,T,sigma);
     deltaPathWiseV(idx)   = delta_pathwise;
     deltaLikeliWiseV(idx) = delta_likelihood; 
     
     % Vega -- pathwise

     vega_pathwise = PathwiseVega(S0,S,sigma,K,r,T);
     vegaPathWiseV(idx) =vega_pathwise;
     vegaLikelihoodV(idx)= LikelihoodVega(S0,S,K,r,T,sigma);
     idx = idx +1;
end

figure(1)
grid on; hold on
plot(NoOfPathsV,deltaPathWiseV,'-or','linewidth',1.5)
plot(NoOfPathsV,deltaLikeliWiseV,'o-k','linewidth',1.5)
plot(NoOfPathsV,delta_Exact*ones([length(NoOfPathsV),1]),'linewidth',1.5)
xlabel('number of paths')
ylabel('Delta')
title('Convergence of delta for pathwise and likelihood methods')
legend('pathwise est','likelihood ratio est','exact')
ylim([delta_Exact*0.8,delta_Exact*1.2])


figure(2)
grid on; hold on
plot(NoOfPathsV,vegaPathWiseV,'-or','linewidth',1.5)
plot(NoOfPathsV,vegaLikelihoodV,'o-k','linewidth',1.5)
plot(NoOfPathsV,vega_Exact*ones([length(NoOfPathsV),1]),'linewidth',1.5)
xlabel('number of paths')
ylabel('Vega')
title('Convergence of vega for pathwise and likelihood methods')
legend('pathwise est','likelihood ratio est','exact')
ylim([vega_Exact*0.8,vega_Exact*1.2])

% Delta and vega as functions of maturity time

T         = 10;
NoOfSteps = 1000; 
randn('seed',2);
[S,time] = GeneratePathsGBMEuler(25000,NoOfSteps,T,r,sigma,S0);

deltaPathWiseV   = zeros([length(time),1]);
deltaLikeliWiseV = zeros([length(time),1]);
vegaPathWiseV    = zeros([length(time),1]);
vegaLikelihoodV  = zeros([length(time),1]);
vegaExactV       = zeros([length(time),1]);
deltaExactV      = zeros([length(time),1]);
for i=2:length(time)
  T = time(i);
  S_t = S(:,1:i);
  % Delta- pathwise
  delta_pathwise     = PathwiseDelta(S0,S_t,K,r,T);
  delta_likelihood   = LikelihoodDelta(S0,S_t,K,r,T,sigma);
  deltaPathWiseV(i)  = delta_pathwise;
  deltaLikeliWiseV(i)= delta_likelihood; 
  delta_Exact        = BS_Delta(CP,S0,K,sigma,t,T,r);
  deltaExactV(i)     = delta_Exact;
  
  % Vega -- pathwise

  vega_pathwise     = PathwiseVega(S0,S_t,sigma,K,r,T);
  vegaPathWiseV(i)  = vega_pathwise;
  vegaLikelihoodV(i)= LikelihoodVega(S0,S_t,K,r,T,sigma);  
  vega_Exact        = BS_Vega(S0,K,sigma,t,T,r);
  vegaExactV(i)     = vega_Exact;
end

figure(3)
grid on;
hold on;
plot(time,deltaExactV,'linewidth',1.5)
plot(time,deltaPathWiseV,'linewidth',1.5)
plot(time,deltaLikeliWiseV,'linewidth',1.5)
legend('Exact','Pathwise est','Likelihood est')
title('Estimation of Delta')
xlabel('Time')
ylabel('Delta')

figure(4)
grid on;
hold on;
plot(time,vegaExactV,'linewidth',1.5)
plot(time,vegaPathWiseV,'linewidth',1.5)
plot(time,vegaLikelihoodV,'linewidth',1.5)
legend('Exact','Pathwise est','Likelihood est')
title('Estimation of Vega')
xlabel('Time')
ylabel('Vega')

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

function delta= PathwiseDelta(S0,S,K,r,T)
    temp1 = S(:,end)>K;
    delta =exp(-r*T)*mean(S(:,end)/S0.*temp1); 
    
function delta= LikelihoodDelta(S0,S,K,r,T,sigma)
    temp1 = 1/(S0*sigma^2*T)*max(S(:,end)-K,0);
    beta = log(S(:,end)/S0)-(r-0.5*sigma^2)*T;
    delta = exp(-r*T)*mean(temp1.*beta);
    
function vega= LikelihoodVega(S0,S,K,r,T,sigma)
    beta = log(S(:,end)/S0)-(r-0.5*sigma^2)*T;
    temp1 = -1/sigma + 1/(sigma^3*T)*beta.^2-1/sigma*beta;
    vega = exp(-r*T)*mean(max(S(:,end)-K,0).*temp1);

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
