function CIR_paths_boundary
clc;close all;

% CIR model parameter settings

v0        = 0.1;
kappa     = 0.5;
vbar      = 0.1;
gamma     = 0.8;
NoOfPaths = 1; 
NoOfSteps = 200; 
T         = 1;

% Generation of the CIR paths with truncation scheme

%randn('seed',8)
randn('seed',26)
[Vtruncated,Vreflecting,timeGrid] = GeneratePathsCIREuler2Schemes(NoOfPaths,NoOfSteps,v0,kappa,vbar,gamma,T);

% Generate figure

figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Density plot for S(t)

plot(timeGrid,Vreflecting,'-k','linewidth',1.5)
plot(timeGrid,Vtruncated,'--r','linewidth',1.5)

grid on;
legend('truncated scheme','reflecting scheme')
axis([0,T,0,max(max(Vtruncated))])
xlabel('time')
ylabel('v(t)')
title('reflection vs. truncation scheme')

function [V1,V2,time] = GeneratePathsCIREuler2Schemes(NoOfPaths,NoOfSteps,V0,kappa,vbar,gamma,T)

% Define initial value

V1=zeros(NoOfPaths,NoOfSteps);
V1(:,1) = V0;

V2=zeros(NoOfPaths,NoOfSteps);
V2(:,1) = V0;

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
    V1(:,i+1) = V1(:,i) + kappa*(vbar-V1(:,i))*dt+  gamma* sqrt(V1(:,i)).* (W(:,i+1)-W(:,i));

    % We apply here the truncation scheme to deal with negative values

    V1(:,i+1) = max(V1(:,i+1),0);
    V2(:,i+1) = V2(:,i) + kappa*(vbar-V2(:,i))*dt+  gamma* sqrt(V2(:,i)).* (W(:,i+1)-W(:,i));

    % We apply here the reflection scheme to deal with negative values

    V2(:,i+1) = abs(V2(:,i+1));
    time(i+1) = time(i) + dt;
end
