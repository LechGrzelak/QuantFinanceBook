function CIR_IR_paths
clc;close all;

% CIR model parameter settings

NoOfPaths = 1; 
NoOfSteps = 500; 
T         = 50.0;
lambda     = 0.1;
gamma     = 0.1;
r0        = 0.05;
theta     = 0.05;

% Effect of lambda on Monte Carlo paths

figure(1); hold on; grid on;
lambdaV = [0.01, 0.2, 5.0];
argLegend = cell(3,1);
idx = 1;
for lambdaTemp= lambdaV
    randn('seed',7)
    [R,timeGrid] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,r0,lambdaTemp,theta,gamma,T);
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
gammaV = [0.1, 0.2, 0.3];
argLegend = cell(3,1);
idx = 1;
for gammaTemp= gammaV
    randn('seed',1)
    [R,timeGrid] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,r0,lambda,theta,gammaTemp,T);
    plot(timeGrid,R,'linewidth',1.5)
    argLegend{idx} = sprintf('gamma=%.2f',gammaTemp);
    idx = idx + 1;
end
legend(argLegend)
xlim([0,T])
xlabel('time')
ylabel('r(t)')

function [R,time] = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,r0,lambda,theta,gamma,T)

% Define initial value

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
    R(:,i+1)  = R(:,i) + lambda*(theta-R(:,i))*dt+  gamma* sqrt(R(:,i)).* (W(:,i+1)-W(:,i));
        
    % We apply here the truncation scheme to deal with negative values

    R(:,i+1) = max(R(:,i+1),0);
    time(i+1) = time(i) + dt;
end
