function HW_paths
clc;close all;

% CIR model parameter settings

NoOfPaths = 1; 
NoOfSteps = 500; 
T         = 50;
lambda    = 0.5;
eta       = 0.01;

% We define a ZCB curve (obtained from the market)

P0T = @(T)exp(-0.05*T);

% Effect of lambda on Monte Carlo paths

figure(1); hold on; grid on;
lambdaV = [-0.01, 0.2, 5.0];
argLegend = cell(3,1);
idx = 1;
for lambdaTemp= lambdaV
    randn('seed',1)
    [R,timeGrid] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambdaTemp, eta);
    plot(timeGrid,R,'linewidth',1.5)
    argLegend{idx} = sprintf('lambda=%.2f',lambdaTemp);
    idx = idx + 1;
end
legend(argLegend)
xlim([0,T])
xlabel('time')
ylabel('r(t)')

% Effect of eta on Monte Carlo paths

figure(2); hold on; grid on;
etaV = [0.1, 0.2, 0.3];
argLegend = cell(3,1);
idx = 1;
for etaTemp= etaV
    randn('seed',1)
    [R,timeGrid] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, etaTemp);
    plot(timeGrid,R,'linewidth',1.5)
    argLegend{idx} = sprintf('eta=%.2f',etaTemp);
    idx = idx + 1;
end
legend(argLegend)
xlim([0,T])
xlabel('time')
ylabel('r(t)')

function [R,time] = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambda, eta)

% Time step

dt = 0.0001;

% Complex number

f0T = @(t)- (log(P0T(t+dt))-log(P0T(t-dt)))/(2*dt);
   
% Initial interest rate is forward rate at time t->0

r0 = f0T(0.00001);  
theta = @(t) 1.0/lambda * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambda*lambda)*(1.0-exp(-2.0*lambda*t));  

% Define initial values

R=zeros(NoOfPaths,NoOfSteps);
R(:,1) = r0;

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
    R(:,i+1)  = R(:,i) + lambda*(theta(time(i))-R(:,i))*dt+  eta* (W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end
