function PoissonProcess_paths
clc;close all;

% Monte Carlo settings

NoOfPaths = 25;
NoOfSteps = 500;
T         = 30;
dt        = T/NoOfSteps;
time      = 0:dt:T;

% Poisson process settings

xiP       = 1;

% Empty matrices for the Poisson paths

X  = zeros(NoOfPaths,NoOfSteps);
Xc =zeros(NoOfPaths,NoOfSteps);

% Random noise

Z = random('poisson',xiP*dt,[NoOfPaths,NoOfSteps]);

% Creation of the paths

for i=1:NoOfSteps
    X(:,i+1)  = X(:,i) + Z(:,i);
    Xc(:,i+1) = Xc(:,i) - xiP * dt + Z(:,i);
end
figure(1)
plot(time,X,'color',[0 0.45 0.74],'linewidth',1.2)
xlabel('time')
ylabel('$$X_\mathcal{P}(t)$$','interpreter','latex')
grid on

figure(2)
plot(time,Xc,'color',[0 0.45 0.74],'linewidth',1.2)
xlabel('time')
ylabel('$$X_\mathcal{P}(t)-\xi_pdt$$','interpreter','latex')
grid on
