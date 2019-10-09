function ErrorHistograms
clc;close all;
S0        = 50;
NoOfSteps = 10; 
NoOfPaths = 10000;
r         = 0.06;
sigma     = 0.3;
T         = 1;

% Generate Monte Carlo paths for approximation and exact solution

randn('seed',1)
[SEuler,Sexact1,~] = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0);
randn('seed',1)
[SMilstein,Sexact2,~] = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S0);

% Make histograms- Euler scheme 

figure(1)
h1 = histogram(SEuler(:,end)-Sexact1(:,end),50);
h1.FaceColor=[1,1,1];
grid on
xlabel('Error at time T')
title('Euler scheme')
grid on;

% Make histograms- Milstein scheme 

figure(2)
h1 = histogram(SMilstein(:,end)-Sexact2(:,end),50);
h1.FaceColor=[1,1,1];
grid on
xlabel('Error at time T')
title('Milstein scheme')
grid on;

function [S1,S2,time] = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0)

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
    S1(:,i+1) = S1(:,i) + r * S1(:,i)*dt + sigma * S1(:,i).*(W(:,i+1)-W(:,i));
    S2(:,i+1) = S2(:,i) .* exp((r -0.5*sigma^2)*dt + sigma * (W(:,i+1)-W(:,i)));
    time(i+1) = time(i) + dt;
end

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

