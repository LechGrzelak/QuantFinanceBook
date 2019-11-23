function MertonProcess_paths
clc;close all;

% Monte Carlo settings

NoOfPaths = 10;
NoOfSteps = 500;
T         = 5;

% Model parameters

r     = 0.05;
S0    = 100;
sigma = 0.2;
sigmaJ= 0.5;
muJ   = 0;
xiP   = 1;

[S,time,X] = GeneratePathsMerton(NoOfPaths,NoOfSteps,S0,T,xiP,muJ,sigmaJ,r,sigma);

figure(1)
plot(time,X,'color',[0 0.45 0.74],'linewidth',1.2)
xlabel('time')
ylabel('$$X(t)$$','interpreter','latex')
grid on

figure(2)
plot(time,S,'color',[0 0.45 0.74],'linewidth',1.2)
xlabel('time')
ylabel('$$S(t)$$','interpreter','latex')
grid on

% Martingale error

error = mean(S(:,end))-S0*exp(r*T)

function [S,time,X] =GeneratePathsMerton(NoOfPaths,NoOfSteps,S0,T,xiP,muJ,sigmaJ,r,sigma)

% Empty matrices for the Poisson process and stock paths

Xp     = zeros(NoOfPaths,NoOfSteps);
X      = zeros(NoOfPaths,NoOfSteps);
W      = zeros(NoOfPaths,NoOfSteps);
X(:,1) = log(S0);
S      = zeros(NoOfPaths,NoOfSteps);
S(:,1) = S0;
dt        = T/NoOfSteps;

% Random noise

Z1 = random('poisson',xiP*dt,[NoOfPaths,NoOfSteps]);
Z2 = random('normal',0,1,[NoOfPaths,NoOfSteps]);
J  = random('normal',muJ,sigmaJ,[NoOfPaths,NoOfSteps]);

% Creation of the paths

% Expectation E(exp(J))

EeJ = exp(muJ + 0.5*sigmaJ^2);
time = zeros([NoOfSteps+1,1]);
for i=1:NoOfSteps  
    if NoOfPaths>1
        Z2(:,i) = (Z2(:,i)-mean(Z2(:,i)))/std(Z2(:,i));
    end
    Xp(:,i+1) = Xp(:,i) + Z1(:,i);
    W(:,i+1)  = W(:,i)  + sqrt(dt)* Z2(:,i);
        
    X(:,i+1)  = X(:,i) + (r- xiP*(EeJ-1)-0.5*sigma^2)*dt +  sigma* (W(:,i+1)-W(:,i)) + J(:,i).*(Xp(:,i+1)-Xp(:,i));
    S(:,i+1) = exp(X(:,i));
    time(i+1) = time(i) + dt;
end
