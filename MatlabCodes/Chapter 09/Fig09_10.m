function Jacobi_Paths
clc;close all;
NoOfPaths  =5000;
NoOfSteps  = 10; 
T          = 5;
rho0      = 0.0;
kappa     = 0.1;
gamma     = 0.6;
mu        = 0.5;

figure(1)
randn('seed',1)
[Rho,timeGrid] = GeneratePathsJackobi(NoOfPaths,NoOfSteps,T,rho0,mu,kappa,gamma);
plot(timeGrid,Rho,'color',[0,0.45,0.75],'linewidth',1.5)
axis([0,T,min(min(Rho)),max(max(Rho))])
grid on
xlabel('time')
ylabel('Rho(t)')

figure(2)
h1 = histogram(Rho(:,end),50);
h1.FaceColor=[1,1,1];
grid on
xlabel('Error at time T')
title('Rho')
grid on;
function [Rho,time] = GeneratePathsJackobi(NoOfPaths,NoOfSteps,T,rho0,mu,kappa,gamma)
Rho=zeros(NoOfPaths,NoOfSteps);
Rho(:,1) = rho0;

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
    Rho(:,i+1) = Rho(:,i) + kappa*(mu - Rho(:,i)) * dt + gamma* sqrt(1.0 -Rho(:,i).*Rho(:,i)) .* (W(:,i+1)-W(:,i));
        
    % Handling of the boundary conditions to ensure that paths stay in the (-1,1] range

    Rho(:,i+1) = max(Rho(:,i+1),-1.0);
    Rho(:,i+1) = min(Rho(:,i+1),1.0);
    time(i+1) = time(i) + dt;
end
