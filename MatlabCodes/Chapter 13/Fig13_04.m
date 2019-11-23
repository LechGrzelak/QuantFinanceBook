function SZHWvsHestonDensity
clc;close all;

% Heston model parameter settings

NoOfPaths = 25000; 
NoOfSteps = 500; 
T         = 2.0;
v0        = 0.063;
vbar      = 0.063;

% Set 1

kappa     = 1.2;
gamma     = 0.1;

% Set 2

%kappa     = 0.25;
%gamma     = 0.63;

% Effect of kappa on Monte Carlo paths

figure(1); hold on; grid on;
randn('seed',7)
[V,timeGrid] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,v0,kappa,vbar,gamma,T);

% Volatility for the Heston model, sqrt(v(t))

volHeston = sqrt(V);

% Plot paths for the volatility

plot(timeGrid,volHeston(1:50,:),'linewidth',1.5)
xlim([0,T])
xlabel('time')
ylabel('sqrt(V(t))')

% The Ornstein-Uhlenbec process for the volatility for the SZ model

kappa    = 1;
sigma0   = sqrt(v0);
sigmaBar = (mean(volHeston(:,end))-sigma0*exp(-T))/(1.0 - exp(-T));
gamma    = sqrt(2.0*var(volHeston(:,end))/(1.0-exp(-2.0*T)));
[sigma,~] = GeneratePathsOrnsteinUhlenbecEuler(NoOfPaths,NoOfSteps,T,sigma0,sigmaBar, kappa, gamma);

% Plotting of the densities for both processes

[yHes,xHes]=ksdensity(volHeston(:,end));
[ySZ,xSZ]=ksdensity(sigma(:,end));

figure(2)
plot(xHes,yHes,'b','linewidth',1.5);
hold on
plot(xSZ,ySZ,'.-r','linewidth',1.5);
grid on
legend('Heston vol.: sqrt(V(T)','Schobel-Zhu vol.: sigma(T)')

function [V,time] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,v0,kappa,vbar,gamma,T)

% Define initial values

V=zeros(NoOfPaths,NoOfSteps);
V(:,1) = v0;

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
    V(:,i+1)  = V(:,i) + kappa*(vbar-V(:,i))*dt+  gamma* sqrt(V(:,i)).* (W(:,i+1)-W(:,i));
        
    % We apply here the truncation scheme to deal with negative values

    V(:,i+1) = max(V(:,i+1),0);
    time(i+1) = time(i) + dt;
end

function [sigma,time] = GeneratePathsOrnsteinUhlenbecEuler(NoOfPaths,NoOfSteps,T,sigma0,sigmaBar, kappa, gamma)  

% Define initial values

sigma=zeros(NoOfPaths,NoOfSteps);
sigma(:,1) = sigma0;

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
    sigma(:,i+1)  = sigma(:,i) + kappa*(sigmaBar-sigma(:,i))*dt + gamma* (W(:,i+1)-W(:,i));
    time(i+1) = time(i) + dt;
end
