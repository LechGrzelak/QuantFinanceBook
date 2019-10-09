function m_g_and_gn
clc;close all;

% Monte Carlo setting for generation of paths

randn('seed',43) 
S0        = 100;
NoOfPaths = 1;
NoOfSteps = 500;
r         = 0.05;
sigma     = 0.4;
T         = 1;
dt        = T/NoOfSteps;
X         = zeros(NoOfPaths,NoOfSteps);
X(:,1)    = log(S0);
Z         = randn([NoOfPaths,NoOfSteps]);

% Simulation of GBM process

for i=1:NoOfSteps
    if NoOfPaths>1
        Z(:,i)   = (Z(:,i)-mean(Z(:,i)))./std(Z(:,i));
    end
    X(:,i+1) = X(:,i) + (r-0.5*sigma^2)*dt + sigma* sqrt(dt)*Z(:,i);
end
S    = exp(X);
time = 0:dt:T;

% Number of intervals

m          = 20;
pathNumber = 1;

figure(1);hold on;
plot(time,S,'linewidth',2)
f = @(arg)interp1(time,S(pathNumber,:),arg);
for k = 1:m*T
    tGrid = linspace((k-1)/m, k/m, 100);
    g_m   = m * trapz(tGrid,f(tGrid));
    plot(tGrid,g_m.*ones(1,length(tGrid)),'-r','linewidth',3)
    plot(tGrid(1),g_m,'marker','o','MarkerFaceColor',[1 0 0 ],'Color',[1 0 0])
    plot(tGrid(end),g_m,'marker','o','Color',[1 0 0 ])
end
xlabel('t')
ylabel('paths for g(t) and g_n(t)')
title(strcat('n = ','',num2str(m)))
grid on
