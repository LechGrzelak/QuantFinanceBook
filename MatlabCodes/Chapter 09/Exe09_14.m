function PathwiseSens_TwoStocks
clc;close all;
S10       = 1;
S20       = 1;
r         = 0.06;
sigma1    = 0.3;
sigma2    = 0.2;
T         = 1;
K         = S10;
rho       = 0.7;

% Estimator of rho computed with finite differences

drho = 1e-5;
randn('seed',1);
[S1,S2] = GeneratePathsTwoStocksEuler(20000,1000,T,r,S10,S20,rho-drho,sigma1,sigma2);     
optValue1 = AssetOfNothingPayoff(S1,S2,K,T,r);
randn('seed',1);
[S1,S2] = GeneratePathsTwoStocksEuler(20000,1000,T,r,S10,S20,rho+drho,sigma1,sigma2);     
optValue2 = AssetOfNothingPayoff(S1,S2,K,T,r);
exact_rho = (optValue2- optValue1)/(2*drho); 

% Preparation for the simulation

NoOfSteps = 1000;
NoOfPathsV = round(linspace(5,20000,50));
rhoPathWiseV = zeros([length(NoOfPathsV),1]);

idx = 1;
for nPaths = NoOfPathsV
     fprintf('Running simulation with %.f paths',nPaths);
     fprintf('\n')
     
     randn('seed',1);
     [S1,S2] = GeneratePathsTwoStocksEuler(nPaths,NoOfSteps,T,r,S10,S20,rho,sigma1,sigma2);         

     % dVdrho -- pathwise

     rho_pathwise = PathwiseRho(S10,S20,sigma1,sigma2,rho,S1,S2,K,r,T);
     rhoPathWiseV(idx)= rho_pathwise;
     idx = idx +1;
end

figure(1)
grid on; hold on
plot(NoOfPathsV,rhoPathWiseV,'-or','linewidth',1.5)
plot(NoOfPathsV,exact_rho*ones([length(NoOfPathsV),1]),'linewidth',1.5)
xlabel('number of paths')
ylabel('$$dV/d\rho$$')
title('Convergence of pathwise sensitivity to \rho w.r.t number of paths')
legend('pathwise est','exact')
ylim([exact_rho*0.8,exact_rho*1.2])
   

function optValue = AssetOfNothingPayoff(S1,S2,K,T,r)
optValue = zeros(length(K),1);
for i =1:length(K)
    optValue(i) = exp(-r*T)*mean(S2(:,end).*(S1(:,end)>K(i)));
end

function dVdrho = PathwiseRho(S10,S20,sigma1,sigma2,rho,S1,S2,K,r,T)
W1 = 1/sigma1*(log(S1(:,end)/S10)-(r-0.5*sigma1^2)*T);
W2 = 1/(sigma2*sqrt(1-rho^2))*(log(S2(:,end)/S20)-(r-0.5*sigma2^2)*T- sigma2*rho*W1);
dVdrho = exp(-r*T)*mean((S1(:,end)>K).*S2(:,end).*(sigma2*W1-sigma2*rho/(sqrt(1-rho^2))*W2));

    function [S1,S2,time] = GeneratePathsTwoStocksEuler(NoOfPaths,NoOfSteps,T,r,S10,S20,rho,sigma1,sigma2)
X1=zeros(NoOfPaths,NoOfSteps);
X1(:,1) = log(S10);
X2=zeros(NoOfPaths,NoOfSteps);
X2(:,1) = log(S20);

% Random noise

Z1=random('normal',0,1,[NoOfPaths,NoOfSteps]);
Z2=random('normal',0,1,[NoOfPaths,NoOfSteps]);
W1=zeros([NoOfPaths,NoOfSteps]);
W2=zeros([NoOfPaths,NoOfSteps]);

dt = T / NoOfSteps;
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps
    if NoOfPaths>1
        Z1(:,i)   = (Z1(:,i) - mean(Z1(:,i))) / std(Z1(:,i));
        Z2(:,i)   = (Z2(:,i) - mean(Z2(:,i))) / std(Z2(:,i));
    end
    Z2(:,i)= rho * Z1(:,i) + sqrt(1-rho^2)*Z2(:,i);
    W1(:,i+1)  = W1(:,i) + sqrt(dt).*Z1(:,i);
    W2(:,i+1)  = W2(:,i) + sqrt(dt).*Z2(:,i);
    
    X1(:,i+1) = X1(:,i) + (r -0.5*sigma1^2)*dt + sigma1 *(W1(:,i+1)-W1(:,i));
    X2(:,i+1) = X2(:,i) + (r -0.5*sigma2^2)*dt + sigma2 *(W2(:,i+1)-W2(:,i));
    time(i+1) = time(i) + dt;
end
S1= exp(X1);
S2= exp(X2);
