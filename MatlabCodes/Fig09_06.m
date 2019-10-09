function MilsteinConvergence_GBM
clc;close all;
S0        = 50;
NoOfPaths = 25; 
NoOfSteps = 250; 
r         = 0.06;
sigma     = 0.3;
T         = 1;

% Generate Monte Carlo paths for approximation and exact simulation

[Saprox,Sexact,gridTime] = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0);

% Plot the paths

figure(1)
hold on;grid on
plot(gridTime,Saprox,'--r')
plot(gridTime,Sexact,'-b')
xlabel('time')
ylabel('stock value')
title('Monte Carlo paths for exact and approximated solution')
xlim([0,T])

% Weak and strong convergence

NoOfPaths = 1000;
NoOfStepsV  = 1:1:500;
dtV = T./NoOfStepsV;
errorWeak   = zeros([length(NoOfStepsV),1]);
errorStrong = zeros([length(NoOfStepsV),1]);
idx = 1;
for NoOfSteps = NoOfStepsV
    [Saprox,Sexact,~] = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0);
    errorWeak(idx)    = abs(mean(Saprox(:,end))- mean(Sexact(:,end)));
    errorStrong(idx)  = mean(abs(Saprox(:,end)- Sexact(:,end)));
    idx = idx +1;
end
figure(2);hold on;grid on
plot(dtV,errorWeak,'--r','linewidth',1.5)
plot(dtV,errorStrong,'b','linewidth',1.5)
xlabel('$$\Delta t$$','interpreter','latex')
ylabel('error, $$\epsilon(\Delta t)$$','interpreter','latex')
grid on;
title('Error analysis for Euler scheme')
legend('weak conv.','strong conv.')

function [S1,S2,time] = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0)

% Approximation

S1=zeros(NoOfPaths,NoOfSteps);
S1(:,1) = S0;

% Exact

S2=zeros(NoOfPaths,NoOfSteps);
S2(:,1) = S0;

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
    S1(:,i+1) = S1(:,i) + r * S1(:,i)*dt + sigma * S1(:,i).*(W(:,i+1)-W(:,i))+...
        + 0.5*sigma*sigma*S1(:,i).*((W(:,i+1)-W(:,i)).^2 - dt);   
    S2(:,i+1) = S2(:,i) .* exp((r -0.5*sigma^2)*dt + sigma * (W(:,i+1)-W(:,i)));
    time(i+1) = time(i) + dt;
end
