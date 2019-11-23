function PathwiseSens_Asian_Delta
clc;close all;

NoOfPathsMax = 25000;
NoOfSteps    = 1000;

% Black-Scholes model parameters

S0     = 100;
CP     = 'c';
sigma  = 0.15;

T      = 1;
r      = 0.05;
K      = S0;

% Time grid for averaging

nPeriods = 10;
Ti= linspace(0,T,nPeriods);

% Delta estimated by central differences

dS0 = 1e-04;
randn('seed',2)
[S1,time] =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma,S0-dS0);
randn('seed',2)
[S2,time] =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma,S0+dS0);
value1    =AsianOption(S1,time,Ti,r,T,CP,K);
value2    =AsianOption(S2,time,Ti,r,T,CP,K);
delta_Exact = (value2-value1)/(2*dS0);

% Vega estimated by central differences

dsigma = 1e-04;
randn('seed',2)
[S1,time] =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma-dsigma,S0);
randn('seed',2)
[S2,time] =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma+dsigma,S0);
value1    =AsianOption(S1,time,Ti,r,T,CP,K);
value2    =AsianOption(S2,time,Ti,r,T,CP,K);
vega_Exact = (value2-value1)/(2*dsigma);

NoOfPathsV = round(linspace(5,NoOfPathsMax,25));
deltaPathWiseV = zeros([length(NoOfPathsV),1]);
vegaPathWiseV  = zeros([length(NoOfPathsV),1]);

idx = 1;
for nPaths = NoOfPathsV
     fprintf('Running simulation with %.f paths',nPaths);
     fprintf('\n')
     
     randn('seed',2);
     [S,time] = GeneratePathsGBMEuler(nPaths,NoOfSteps,T,r,sigma,S0);
     [~,A]    = AsianOption(S,time,Ti,r,T,CP,K);

     % Delta -- pathwise

     delta_pathwise = PathwiseDelta(S0,A,K,r,T);
     deltaPathWiseV(idx)= delta_pathwise;
     
     % Vega -- pathwise

     vega_pathwise = PathwiseVegaAsian(S0,S,time,sigma,K,r,T,Ti);
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

% Value Asian option, given the stock paths

function [value,A]= AsianOption(S,time,Ti,r,T,cp,K)
[~,idx] =min(abs(time-Ti));
S_i = S(:,idx);
A = mean(S_i,2);
if cp =='c'
    value = exp(-r*T)*mean(max(A-K,0.0));
elseif cp =='p'
    value = exp(-r*T)*mean(max(K-A,0.0));
end

function delta= PathwiseDelta(S0,S,K,r,T)
    temp1 = S(:,end)>K;
    delta =exp(-r*T)*mean(S(:,end)/S0.*temp1); 

function vega = PathwiseVegaAsian(S0,S,time,sigma,K,r,T,Ti)
[~,idx] =min(abs(time-Ti));
S = S(:,idx);
A = mean(S,2);

Sum = zeros(length(A),1);
for i = 1:length(Ti)
    temp1 = 1.0/sigma* S(:,i).*(log(S(:,i)/S0)-(r+0.5*sigma^2)*Ti(i));
    Sum = Sum + temp1;
end
vega  = 1/length(Ti)*exp(-r*T)*mean((A>K).*Sum);

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
